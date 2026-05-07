#!/usr/bin/env python3
"""Replay MoE router event traces through expert cache/offload policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ExpertRef:
    ref_index: int
    event_index: int
    request_id: str
    layer: int
    token_pos: int
    expert: int


@dataclass
class CacheState:
    resident: dict[int, set[int]]
    last_touch: dict[int, dict[int, int]]
    prefetched_unused: set[tuple[int, int]]


@dataclass(frozen=True)
class WrecStats:
    p_window_use: dict[int, dict[int, float]]
    expected_routed_tokens: dict[int, dict[int, float]]
    base_score: dict[int, dict[int, float]]
    train_frequency: dict[int, dict[int, float]]
    cross_layer_transition: dict[int, dict[int, dict[int, float]]]


@dataclass
class WrecOnlineState:
    recent_refs: dict[int, deque[int]]
    recent_counts: dict[int, Counter[int]]
    current_request_id: str | None
    request_counts: dict[int, Counter[int]]
    request_totals: dict[int, int]
    token_layer_experts: dict[tuple[int, int], tuple[int, ...]]
    current_event_key: tuple[int, int] | None
    current_event_experts: list[int]
    decisions: Any | None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_event_trace(path: Path) -> tuple[list[ExpertRef], dict[str, Any]]:
    refs: list[ExpertRef] = []
    event_index = -1
    max_layer = -1
    max_expert = -1
    request_tokens: set[tuple[str, int]] = set()
    token_layer_events: set[tuple[str, int, int]] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event_index += 1
            request_id = str(row["request_id"])
            layer = int(row["layer"])
            token_pos = int(row["token_pos"])
            selected = row.get("selected_experts")
            if not isinstance(selected, list) or not selected:
                raise ValueError(f"{path}:{line_no} missing selected_experts")

            max_layer = max(max_layer, layer)
            request_tokens.add((request_id, token_pos))
            token_layer_events.add((request_id, layer, token_pos))
            for expert in selected:
                expert_id = int(expert)
                max_expert = max(max_expert, expert_id)
                refs.append(
                    ExpertRef(
                        ref_index=len(refs),
                        event_index=event_index,
                        request_id=request_id,
                        layer=layer,
                        token_pos=token_pos,
                        expert=expert_id,
                    )
                )

    if not refs:
        raise ValueError(f"no expert refs loaded from {path}")

    metadata = {
        "trace_path": str(path),
        "num_expert_refs": len(refs),
        "num_router_events": event_index + 1,
        "num_input_tokens": len(request_tokens),
        "num_token_layer_events": len(token_layer_events),
        "num_layers": max_layer + 1,
        "num_experts": max_expert + 1,
    }
    return refs, metadata


def infer_expert_bytes(model_path: Path | None, dtype: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    if model_path is None:
        raise ValueError("--expert-bytes is required when --model-path is not provided")
    config = read_json(model_path / "config.json" if model_path.is_dir() else model_path)
    hidden = int(config["hidden_size"])
    intermediate = int(config.get("moe_intermediate_size", config.get("intermediate_size")))
    return 3.0 * hidden * intermediate * DTYPE_BYTES[dtype]


def parse_cache_values(text: str, num_experts: int) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.endswith("%"):
            pct = float(item[:-1]) / 100.0
            values.append(max(0, math.ceil(num_experts * pct)))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("at least one cache capacity is required")
    return sorted(set(values))


def build_future_queues(refs: Iterable[ExpertRef]) -> dict[tuple[int, int], deque[int]]:
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for ref in refs:
        future[(ref.layer, ref.expert)].append(ref.ref_index)
    return future


def build_static_hot(refs: Iterable[ExpertRef], capacity: int) -> dict[int, set[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for ref in refs:
        counts[ref.layer][ref.expert] += 1
    return {
        layer: {expert for expert, _ in counter.most_common(capacity)}
        for layer, counter in counts.items()
    }


def build_layer_event_experts(refs: Iterable[ExpertRef]) -> dict[int, list[tuple[int, ...]]]:
    events: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    current_event: tuple[str, int, int] | None = None
    current_layer = 0
    current_experts: list[int] = []
    for ref in refs:
        event_key = (ref.request_id, ref.layer, ref.token_pos)
        if current_event is None:
            current_event = event_key
            current_layer = ref.layer
        elif event_key != current_event:
            events[current_layer].append(tuple(current_experts))
            current_event = event_key
            current_layer = ref.layer
            current_experts = []
        current_experts.append(ref.expert)
    if current_event is not None:
        events[current_layer].append(tuple(current_experts))
    return events


def build_request_token_layer_events(
    refs: Iterable[ExpertRef],
) -> dict[str, dict[int, dict[int, tuple[int, ...]]]]:
    events: dict[str, dict[int, dict[int, tuple[int, ...]]]] = defaultdict(lambda: defaultdict(dict))
    current_event: tuple[str, int, int] | None = None
    current_experts: list[int] = []
    for ref in refs:
        event_key = (ref.request_id, ref.layer, ref.token_pos)
        if current_event is None:
            current_event = event_key
        elif event_key != current_event:
            request_id, layer, token_pos = current_event
            events[request_id][token_pos][layer] = tuple(current_experts)
            current_event = event_key
            current_experts = []
        current_experts.append(ref.expert)
    if current_event is not None:
        request_id, layer, token_pos = current_event
        events[request_id][token_pos][layer] = tuple(current_experts)
    return events


def build_cross_layer_transition(
    refs: list[ExpertRef],
    *,
    num_layers: int,
    num_experts: int,
) -> dict[int, dict[int, dict[int, float]]]:
    event_map = build_request_token_layer_events(refs)
    pair_counts: dict[int, dict[int, Counter[int]]] = {
        layer: {expert: Counter() for expert in range(num_experts)}
        for layer in range(1, num_layers)
    }
    for token_layers_by_request in event_map.values():
        for token_layers in token_layers_by_request.values():
            for layer in range(1, num_layers):
                previous = token_layers.get(layer - 1)
                current = token_layers.get(layer)
                if not previous or not current:
                    continue
                for previous_expert in previous:
                    pair_counts[layer][previous_expert].update(current)

    transition: dict[int, dict[int, dict[int, float]]] = defaultdict(dict)
    for layer, by_previous in pair_counts.items():
        for previous_expert, counts in by_previous.items():
            total = sum(counts.values())
            if total <= 0:
                transition[layer][previous_expert] = {
                    expert: 1.0 / num_experts for expert in range(num_experts)
                }
                continue
            transition[layer][previous_expert] = {
                expert: counts[expert] / total for expert in range(num_experts)
            }
    return transition


def build_wrec_stats(
    refs: list[ExpertRef],
    *,
    num_layers: int,
    num_experts: int,
    window_size: int,
    expert_bytes: float,
    bandwidth_gbps: float,
) -> WrecStats:
    layer_events = build_layer_event_experts(refs)
    p_window_use: dict[int, dict[int, float]] = defaultdict(dict)
    expected_routed_tokens: dict[int, dict[int, float]] = defaultdict(dict)
    train_frequency: dict[int, dict[int, float]] = defaultdict(dict)
    base_score: dict[int, dict[int, float]] = defaultdict(dict)
    cross_layer_transition = build_cross_layer_transition(
        refs,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    transfer_ms = expert_bytes / (bandwidth_gbps * 1e9) * 1000.0
    miss_stall_ms = transfer_ms

    for layer in range(num_layers):
        events = layer_events.get(layer, [])
        counts = Counter(expert for experts in events for expert in experts)
        total_refs = sum(counts.values())
        window_hits = Counter()
        window_counts = Counter()
        windows = max(0, len(events) - window_size + 1)
        if windows:
            for start in range(windows):
                window = events[start : start + window_size]
                present: set[int] = set()
                for experts in window:
                    present.update(experts)
                    window_counts.update(experts)
                for expert in present:
                    window_hits[expert] += 1
        for expert in range(num_experts):
            p_use = window_hits[expert] / windows if windows else 0.0
            expected_tokens = window_counts[expert] / windows if windows else 0.0
            freq = counts[expert] / total_refs if total_refs else 0.0
            p_window_use[layer][expert] = p_use
            expected_routed_tokens[layer][expert] = expected_tokens
            train_frequency[layer][expert] = freq
            base_score[layer][expert] = p_use * expected_tokens * miss_stall_ms - transfer_ms
    return WrecStats(
        p_window_use=p_window_use,
        expected_routed_tokens=expected_routed_tokens,
        base_score=base_score,
        train_frequency=train_frequency,
        cross_layer_transition=cross_layer_transition,
    )


def init_wrec_online_state(num_layers: int, history_size: int, decisions: Any | None) -> WrecOnlineState:
    return WrecOnlineState(
        recent_refs={layer: deque(maxlen=history_size) for layer in range(num_layers)},
        recent_counts={layer: Counter() for layer in range(num_layers)},
        current_request_id=None,
        request_counts={layer: Counter() for layer in range(num_layers)},
        request_totals={layer: 0 for layer in range(num_layers)},
        token_layer_experts={},
        current_event_key=None,
        current_event_experts=[],
        decisions=decisions,
    )


def write_decision(decisions: Any | None, payload: dict[str, Any]) -> None:
    if decisions is None:
        return
    decisions.write(json.dumps(payload, ensure_ascii=False) + "\n")


def update_wrec_history(online: WrecOnlineState, ref: ExpertRef) -> None:
    if online.current_request_id != ref.request_id:
        online.current_request_id = ref.request_id
        for counter in online.request_counts.values():
            counter.clear()
        for layer in online.request_totals:
            online.request_totals[layer] = 0
        online.token_layer_experts.clear()
        online.current_event_key = None
        online.current_event_experts = []

    event_key = (ref.layer, ref.token_pos)
    if online.current_event_key is None:
        online.current_event_key = event_key
    elif online.current_event_key != event_key:
        online.token_layer_experts[online.current_event_key] = tuple(online.current_event_experts)
        online.current_event_key = event_key
        online.current_event_experts = []

    recent = online.recent_refs[ref.layer]
    counts = online.recent_counts[ref.layer]
    if recent.maxlen is not None and len(recent) >= recent.maxlen:
        expired = recent[0]
        counts[expired] -= 1
        if counts[expired] <= 0:
            counts.pop(expired, None)
    recent.append(ref.expert)
    counts[ref.expert] += 1
    online.request_counts[ref.layer][ref.expert] += 1
    online.request_totals[ref.layer] += 1
    online.current_event_experts.append(ref.expert)


def wrec_score(
    *,
    layer: int,
    token_pos: int,
    expert: int,
    timestamp: int,
    state: CacheState,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> float:
    recent = online.recent_refs[layer]
    recent_count = online.recent_counts[layer].get(expert, 0)
    recent_prob = recent_count / len(recent) if recent else 0.0
    request_total = online.request_totals.get(layer, 0)
    request_count = online.request_counts[layer].get(expert, 0)
    request_prob = request_count / request_total if request_total else 0.0
    score = stats.base_score[layer].get(expert, 0.0)
    score += recent_weight * recent_prob
    score += request_weight * request_prob
    if cross_layer_weight > 0.0 and layer > 0:
        previous_experts = online.token_layer_experts.get((layer - 1, token_pos), ())
        if previous_experts:
            transition = stats.cross_layer_transition.get(layer, {})
            cross_prob = sum(
                transition.get(previous_expert, {}).get(expert, 0.0)
                for previous_expert in previous_experts
            ) / len(previous_experts)
            score += cross_layer_weight * cross_prob
    if expert in state.resident[layer]:
        age = max(0, timestamp - state.last_touch[layer].get(expert, timestamp))
        history_len = max(1, recent.maxlen or 1)
        score -= contention_penalty * min(1.0, age / history_len)
    return score


def evict_wrec(
    state: CacheState,
    *,
    layer: int,
    token_pos: int,
    timestamp: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> tuple[int | None, bool]:
    if not state.resident[layer]:
        return None, False
    victim = min(
        state.resident[layer],
        key=lambda expert: wrec_score(
            layer=layer,
            token_pos=token_pos,
            expert=expert,
            timestamp=timestamp,
            state=state,
            online=online,
            stats=stats,
            recent_weight=recent_weight,
            request_weight=request_weight,
            cross_layer_weight=cross_layer_weight,
            contention_penalty=contention_penalty,
        ),
    )
    state.resident[layer].remove(victim)
    state.last_touch[layer].pop(victim, None)
    was_wasted = (layer, victim) in state.prefetched_unused
    state.prefetched_unused.discard((layer, victim))
    return victim, was_wasted


def admit_wrec_or_bypass(
    state: CacheState,
    *,
    ref: ExpertRef,
    capacity: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
    prefetched: bool,
) -> bool:
    if capacity <= 0:
        return False
    if ref.expert in state.resident[ref.layer]:
        state.last_touch[ref.layer][ref.expert] = ref.ref_index
        return False
    incoming_score = wrec_score(
        layer=ref.layer,
        token_pos=ref.token_pos,
        expert=ref.expert,
        timestamp=ref.ref_index,
        state=state,
        online=online,
        stats=stats,
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    )
    victim = None
    wasted = False
    if len(state.resident[ref.layer]) >= capacity:
        victim = min(
            state.resident[ref.layer],
            key=lambda expert: wrec_score(
                layer=ref.layer,
                token_pos=ref.token_pos,
                expert=expert,
                timestamp=ref.ref_index,
                state=state,
                online=online,
                stats=stats,
                recent_weight=recent_weight,
                request_weight=request_weight,
                cross_layer_weight=cross_layer_weight,
                contention_penalty=contention_penalty,
            ),
        )
        victim_score = wrec_score(
            layer=ref.layer,
            token_pos=ref.token_pos,
            expert=victim,
            timestamp=ref.ref_index,
            state=state,
            online=online,
            stats=stats,
            recent_weight=recent_weight,
            request_weight=request_weight,
            cross_layer_weight=cross_layer_weight,
            contention_penalty=contention_penalty,
        )
        if incoming_score <= victim_score:
            write_decision(
                online.decisions,
                {
                    "policy": "wrec_h",
                    "action": "bypass",
                    "ref_index": ref.ref_index,
                    "request_id": ref.request_id,
                    "layer": ref.layer,
                    "token_pos": ref.token_pos,
                    "expert": ref.expert,
                    "incoming_score": incoming_score,
                    "victim": victim,
                    "victim_score": victim_score,
                },
            )
            return False
        state.resident[ref.layer].remove(victim)
        state.last_touch[ref.layer].pop(victim, None)
        wasted = (ref.layer, victim) in state.prefetched_unused
        state.prefetched_unused.discard((ref.layer, victim))

    state.resident[ref.layer].add(ref.expert)
    state.last_touch[ref.layer][ref.expert] = ref.ref_index
    if prefetched:
        state.prefetched_unused.add((ref.layer, ref.expert))
    write_decision(
        online.decisions,
        {
            "policy": "wrec_h",
            "action": "admit_prefetch" if prefetched else "admit_demand",
            "ref_index": ref.ref_index,
            "request_id": ref.request_id,
            "layer": ref.layer,
            "token_pos": ref.token_pos,
            "expert": ref.expert,
            "incoming_score": incoming_score,
            "victim": victim,
            "wasted_victim": wasted,
        },
    )
    return wasted


def init_cache(
    *,
    num_layers: int,
    num_experts: int,
    capacity: int,
    policy: str,
    static_hot: dict[int, set[int]],
) -> CacheState:
    resident: dict[int, set[int]] = {layer: set() for layer in range(num_layers)}
    last_touch: dict[int, dict[int, int]] = {layer: {} for layer in range(num_layers)}

    if capacity >= num_experts:
        for layer in range(num_layers):
            resident[layer] = set(range(num_experts))
            last_touch[layer] = {expert: 0 for expert in resident[layer]}
    elif policy in {"static_hot", "belady_oracle"} and capacity > 0:
        for layer in range(num_layers):
            resident[layer] = set(static_hot.get(layer, set()))
            last_touch[layer] = {expert: 0 for expert in resident[layer]}

    return CacheState(resident=resident, last_touch=last_touch, prefetched_unused=set())


def next_use(future: dict[tuple[int, int], deque[int]], layer: int, expert: int) -> float:
    queue = future.get((layer, expert))
    if not queue:
        return math.inf
    return float(queue[0])


def evict_one(
    state: CacheState,
    *,
    layer: int,
    policy: str,
    future: dict[tuple[int, int], deque[int]],
) -> tuple[int | None, bool]:
    if not state.resident[layer]:
        return None, False
    if policy == "belady_oracle":
        victim = max(state.resident[layer], key=lambda expert: next_use(future, layer, expert))
    else:
        victim = min(state.resident[layer], key=lambda expert: state.last_touch[layer].get(expert, -1))
    state.resident[layer].remove(victim)
    state.last_touch[layer].pop(victim, None)
    was_wasted = (layer, victim) in state.prefetched_unused
    state.prefetched_unused.discard((layer, victim))
    return victim, was_wasted


def load_into_cache(
    state: CacheState,
    *,
    layer: int,
    expert: int,
    capacity: int,
    policy: str,
    future: dict[tuple[int, int], deque[int]],
    timestamp: int,
    prefetched: bool,
) -> bool:
    if capacity <= 0:
        return False
    if expert in state.resident[layer]:
        state.last_touch[layer][expert] = timestamp
        return False
    if len(state.resident[layer]) >= capacity:
        _, wasted = evict_one(state, layer=layer, policy=policy, future=future)
    else:
        wasted = False
    state.resident[layer].add(expert)
    state.last_touch[layer][expert] = timestamp
    if prefetched:
        state.prefetched_unused.add((layer, expert))
    return wasted


def admit_with_belady_or_bypass(
    state: CacheState,
    *,
    layer: int,
    expert: int,
    capacity: int,
    future: dict[tuple[int, int], deque[int]],
    timestamp: int,
) -> bool:
    if capacity <= 0:
        return False
    if expert in state.resident[layer]:
        state.last_touch[layer][expert] = timestamp
        return False
    if len(state.resident[layer]) < capacity:
        state.resident[layer].add(expert)
        state.last_touch[layer][expert] = timestamp
        return False

    victim = max(state.resident[layer], key=lambda cached: next_use(future, layer, cached))
    victim_next = next_use(future, layer, victim)
    current_next = next_use(future, layer, expert)
    if current_next >= victim_next:
        return False

    state.resident[layer].remove(victim)
    state.last_touch[layer].pop(victim, None)
    was_wasted = (layer, victim) in state.prefetched_unused
    state.prefetched_unused.discard((layer, victim))
    state.resident[layer].add(expert)
    state.last_touch[layer][expert] = timestamp
    return was_wasted


def advance_future(future: dict[tuple[int, int], deque[int]], ref: ExpertRef) -> None:
    queue = future[(ref.layer, ref.expert)]
    if queue and queue[0] == ref.ref_index:
        queue.popleft()


def prefetch_for_ref(
    state: CacheState,
    *,
    refs: list[ExpertRef],
    current_index: int,
    capacity: int,
    future: dict[tuple[int, int], deque[int]],
    window_size: int,
    queue_depth: int,
    timestamp: int,
) -> tuple[int, int]:
    if capacity <= 0 or queue_depth <= 0 or window_size <= 0:
        return 0, 0
    current = refs[current_index]
    loaded = 0
    waste_evictions = 0
    seen: set[tuple[int, int]] = set()
    upper = min(len(refs), current_index + 1 + window_size)
    for future_ref in refs[current_index + 1 : upper]:
        key = (future_ref.layer, future_ref.expert)
        if key in seen or future_ref.layer != current.layer:
            continue
        seen.add(key)
        if future_ref.expert in state.resident[future_ref.layer]:
            continue
        wasted = load_into_cache(
            state,
            layer=future_ref.layer,
            expert=future_ref.expert,
            capacity=capacity,
            policy="lru",
            future=future,
            timestamp=timestamp,
            prefetched=True,
        )
        loaded += 1
        waste_evictions += int(wasted)
        if loaded >= queue_depth:
            break
    return loaded, waste_evictions


def prefetch_wrec_h(
    state: CacheState,
    *,
    ref: ExpertRef,
    capacity: int,
    queue_depth: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> tuple[int, int]:
    if capacity <= 0 or queue_depth <= 0:
        return 0, 0
    if len(state.resident[ref.layer]) >= capacity and capacity <= 1:
        return 0, 0
    candidates = []
    for expert in stats.base_score[ref.layer].keys():
        if expert in state.resident[ref.layer]:
            continue
        score = wrec_score(
            layer=ref.layer,
            token_pos=ref.token_pos,
            expert=expert,
            timestamp=ref.ref_index,
            state=state,
            online=online,
            stats=stats,
            recent_weight=recent_weight,
            request_weight=request_weight,
            cross_layer_weight=cross_layer_weight,
            contention_penalty=contention_penalty,
        )
        candidates.append((score, expert))
    candidates.sort(reverse=True)

    loaded = 0
    waste_evictions = 0
    for score, expert in candidates:
        if loaded >= queue_depth:
            break
        if len(state.resident[ref.layer]) >= capacity:
            victim = min(
                state.resident[ref.layer],
                key=lambda cached: wrec_score(
                    layer=ref.layer,
                    token_pos=ref.token_pos,
                    expert=cached,
                    timestamp=ref.ref_index,
                    state=state,
                    online=online,
                    stats=stats,
                    recent_weight=recent_weight,
                    request_weight=request_weight,
                    cross_layer_weight=cross_layer_weight,
                    contention_penalty=contention_penalty,
                ),
            )
            victim_score = wrec_score(
                layer=ref.layer,
                token_pos=ref.token_pos,
                expert=victim,
                timestamp=ref.ref_index,
                state=state,
                online=online,
                stats=stats,
                recent_weight=recent_weight,
                request_weight=request_weight,
                cross_layer_weight=cross_layer_weight,
                contention_penalty=contention_penalty,
            )
            if score <= victim_score:
                break
            _, wasted = evict_wrec(
                state,
                layer=ref.layer,
                token_pos=ref.token_pos,
                timestamp=ref.ref_index,
                online=online,
                stats=stats,
                recent_weight=recent_weight,
                request_weight=request_weight,
                cross_layer_weight=cross_layer_weight,
                contention_penalty=contention_penalty,
            )
        else:
            wasted = False
        state.resident[ref.layer].add(expert)
        state.last_touch[ref.layer][expert] = ref.ref_index
        state.prefetched_unused.add((ref.layer, expert))
        loaded += 1
        waste_evictions += int(wasted)
        write_decision(
            online.decisions,
            {
                "policy": "wrec_h",
                "action": "prefetch",
                "ref_index": ref.ref_index,
                "request_id": ref.request_id,
                "layer": ref.layer,
                "token_pos": ref.token_pos,
                "expert": expert,
                "score": score,
                "wasted_victim": wasted,
            },
        )
    return loaded, waste_evictions


def simulate_policy(
    refs: list[ExpertRef],
    *,
    metadata: dict[str, Any],
    policy: str,
    capacity: int,
    num_experts: int,
    expert_bytes: float,
    bandwidth_gbps: float,
    static_hot_refs: list[ExpertRef],
    window_size: int,
    prefetch_queue_depth: int,
    wrec_stats: WrecStats | None = None,
    wrec_recent_weight: float = 1.0,
    wrec_request_weight: float = 0.0,
    wrec_cross_layer_weight: float = 0.0,
    wrec_contention_penalty: float = 0.0,
    wrec_history_size: int = 64,
    decisions: Any | None = None,
) -> dict[str, Any]:
    static_hot = build_static_hot(static_hot_refs, capacity)
    init_hot = build_static_hot(refs, capacity) if policy == "belady_oracle" else static_hot
    wrec_policies = {"wrec_h", "wrec_h2"}
    init_policy = "static_hot" if policy in wrec_policies else policy
    state = init_cache(
        num_layers=int(metadata["num_layers"]),
        num_experts=num_experts,
        capacity=capacity,
        policy=init_policy,
        static_hot=init_hot,
    )
    future = build_future_queues(refs)
    if policy in wrec_policies and wrec_stats is None:
        raise ValueError(f"{policy} requires wrec_stats")
    effective_request_weight = wrec_request_weight if policy == "wrec_h2" else 0.0
    effective_cross_layer_weight = wrec_cross_layer_weight if policy == "wrec_h2" else 0.0
    wrec_online = (
        init_wrec_online_state(int(metadata["num_layers"]), wrec_history_size, decisions)
        if policy in wrec_policies
        else None
    )

    demand_hits = 0
    demand_misses = 0
    demand_loads = 0
    prefetch_loads = 0
    prefetch_waste_evictions = 0
    full_resident = capacity >= num_experts

    for ref in refs:
        advance_future(future, ref)
        if policy == "route_window_prefetch" and not full_resident:
            loaded, wasted = prefetch_for_ref(
                state,
                refs=refs,
                current_index=ref.ref_index,
                capacity=capacity,
                future=future,
                window_size=window_size,
                queue_depth=prefetch_queue_depth,
                timestamp=ref.ref_index,
            )
            prefetch_loads += loaded
            prefetch_waste_evictions += wasted
        is_event_tail = ref.ref_index == len(refs) - 1 or refs[ref.ref_index + 1].event_index != ref.event_index
        resident_hit = ref.expert in state.resident[ref.layer]
        if resident_hit:
            demand_hits += 1
            state.last_touch[ref.layer][ref.expert] = ref.ref_index
            state.prefetched_unused.discard((ref.layer, ref.expert))
            if policy in wrec_policies and wrec_online is not None:
                update_wrec_history(wrec_online, ref)
                if is_event_tail and not full_resident and wrec_stats is not None:
                    loaded, wasted = prefetch_wrec_h(
                        state,
                        ref=ref,
                        capacity=capacity,
                        queue_depth=prefetch_queue_depth,
                        online=wrec_online,
                        stats=wrec_stats,
                        recent_weight=wrec_recent_weight,
                        request_weight=effective_request_weight,
                        cross_layer_weight=effective_cross_layer_weight,
                        contention_penalty=wrec_contention_penalty,
                    )
                    prefetch_loads += loaded
                    prefetch_waste_evictions += wasted
            continue

        demand_misses += 1
        if policy == "on_demand":
            demand_loads += 1
            continue
        if policy == "static_hot":
            demand_loads += 1
            continue
        if policy == "belady_oracle":
            demand_loads += 1
            wasted = admit_with_belady_or_bypass(
                state,
                layer=ref.layer,
                expert=ref.expert,
                capacity=capacity,
                future=future,
                timestamp=ref.ref_index,
            )
            prefetch_waste_evictions += int(wasted)
            continue
        if policy in wrec_policies and wrec_online is not None and wrec_stats is not None:
            demand_loads += 1
            update_wrec_history(wrec_online, ref)
            wasted = admit_wrec_or_bypass(
                state,
                ref=ref,
                capacity=capacity,
                online=wrec_online,
                stats=wrec_stats,
                recent_weight=wrec_recent_weight,
                request_weight=effective_request_weight,
                cross_layer_weight=effective_cross_layer_weight,
                contention_penalty=wrec_contention_penalty,
                prefetched=False,
            )
            prefetch_waste_evictions += int(wasted)
            if is_event_tail and not full_resident:
                loaded, wasted = prefetch_wrec_h(
                    state,
                    ref=ref,
                    capacity=capacity,
                    queue_depth=prefetch_queue_depth,
                    online=wrec_online,
                    stats=wrec_stats,
                    recent_weight=wrec_recent_weight,
                    request_weight=effective_request_weight,
                    cross_layer_weight=effective_cross_layer_weight,
                    contention_penalty=wrec_contention_penalty,
                )
                prefetch_loads += loaded
                prefetch_waste_evictions += wasted
            continue

        demand_loads += 1
        wasted = load_into_cache(
            state,
            layer=ref.layer,
            expert=ref.expert,
            capacity=capacity,
            policy=policy,
            future=future,
            timestamp=ref.ref_index,
            prefetched=False,
        )
        prefetch_waste_evictions += int(wasted)

    prefetch_waste_evictions += len(state.prefetched_unused)
    transfer_bytes = (demand_loads + prefetch_loads) * expert_bytes
    demand_transfer_bytes = demand_loads * expert_bytes
    prefetch_transfer_bytes = prefetch_loads * expert_bytes
    stall_ms = demand_transfer_bytes / (bandwidth_gbps * 1e9) * 1000.0
    total_refs = len(refs)
    total_input_tokens = max(1, int(metadata["num_input_tokens"]))
    total_token_layer_events = max(1, int(metadata["num_token_layer_events"]))

    return {
        "policy": policy,
        "cache_capacity_per_layer": capacity,
        "num_experts": num_experts,
        "expert_bytes": expert_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "window_size": window_size if policy in {"route_window_prefetch", "wrec_h", "wrec_h2"} else 0,
        "prefetch_queue_depth": prefetch_queue_depth if policy in {"route_window_prefetch", "wrec_h", "wrec_h2"} else 0,
        "wrec_recent_weight": wrec_recent_weight if policy in wrec_policies else 0.0,
        "wrec_request_weight": effective_request_weight if policy in wrec_policies else 0.0,
        "wrec_cross_layer_weight": effective_cross_layer_weight if policy in wrec_policies else 0.0,
        "wrec_contention_penalty": wrec_contention_penalty if policy in wrec_policies else 0.0,
        "wrec_history_size": wrec_history_size if policy in wrec_policies else 0,
        "total_expert_refs": total_refs,
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "cache_hit_rate": demand_hits / total_refs if total_refs else 0.0,
        "workload_weighted_miss_rate": demand_misses / total_refs if total_refs else 0.0,
        "demand_loads": demand_loads,
        "prefetch_loads": prefetch_loads,
        "prefetch_waste_count": prefetch_waste_evictions,
        "transfer_bytes": transfer_bytes,
        "demand_transfer_bytes": demand_transfer_bytes,
        "prefetch_transfer_bytes": prefetch_transfer_bytes,
        "waste_bytes": prefetch_waste_evictions * expert_bytes,
        "stall_ms": stall_ms,
        "stall_ms_per_input_token": stall_ms / total_input_tokens,
        "stall_ms_per_token_layer_event": stall_ms / total_token_layer_events,
        "transfer_bytes_per_input_token": transfer_bytes / total_input_tokens,
        "transfer_bytes_per_token_layer_event": transfer_bytes / total_token_layer_events,
    }


def add_oracle_gap(rows: list[dict[str, Any]]) -> None:
    by_capacity: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_capacity[int(row["cache_capacity_per_layer"])][str(row["policy"])] = row
    for policies in by_capacity.values():
        belady = policies.get("belady_oracle")
        if belady is None:
            continue
        oracle_stall = float(belady["stall_ms"])
        for row in policies.values():
            row["oracle_gap_ms"] = float(row["stall_ms"]) - oracle_stall
            row["oracle_gap_ratio"] = (
                (float(row["stall_ms"]) - oracle_stall) / float(row["stall_ms"])
                if float(row["stall_ms"]) > 0
                else 0.0
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sanity_tests() -> None:
    refs = [
        ExpertRef(i, i, "sanity", 0, i, expert)
        for i, expert in enumerate([0, 1, 0, 2, 0, 3, 0, 1, 0])
    ]
    meta = {
        "num_layers": 1,
        "num_experts": 4,
        "num_input_tokens": len(refs),
        "num_token_layer_events": len(refs),
    }
    kwargs = {
        "metadata": meta,
        "num_experts": 4,
        "expert_bytes": 100.0,
        "bandwidth_gbps": 10.0,
        "static_hot_refs": refs,
        "window_size": 4,
        "prefetch_queue_depth": 2,
    }
    full = simulate_policy(refs, policy="lru", capacity=4, **kwargs)
    zero = simulate_policy(refs, policy="lru", capacity=0, **kwargs)
    lru = simulate_policy(refs, policy="lru", capacity=1, **kwargs)
    belady = simulate_policy(refs, policy="belady_oracle", capacity=1, **kwargs)
    static_hot = simulate_policy(refs, policy="static_hot", capacity=1, **kwargs)
    prefetch = simulate_policy(refs, policy="route_window_prefetch", capacity=2, **kwargs)
    wrec_stats = build_wrec_stats(
        refs,
        num_layers=1,
        num_experts=4,
        window_size=4,
        expert_bytes=100.0,
        bandwidth_gbps=10.0,
    )
    wrec = simulate_policy(refs, policy="wrec_h", capacity=2, wrec_stats=wrec_stats, **kwargs)

    assert full["demand_misses"] == 0, full
    assert zero["demand_misses"] == len(refs), zero
    assert belady["stall_ms"] <= lru["stall_ms"], (belady, lru)
    assert static_hot["stall_ms"] < lru["stall_ms"], (static_hot, lru)
    assert prefetch["waste_bytes"] >= 0, prefetch
    assert prefetch["transfer_bytes"] >= prefetch["demand_transfer_bytes"], prefetch
    assert wrec["demand_misses"] <= zero["demand_misses"], wrec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--static-hot-trace", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument("--bandwidth-gbps", type=float, default=6.4984144931787124)
    parser.add_argument("--cache-capacities", default="1,2,4,25%,50%")
    parser.add_argument(
        "--policies",
        default="on_demand,lru,static_hot,belady_oracle,route_window_prefetch",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--prefetch-queue-depth", type=int, default=2)
    parser.add_argument("--wrec-prefetch-queue-depth", type=int, default=None)
    parser.add_argument("--wrec-recent-weight", type=float, default=1.0)
    parser.add_argument("--wrec-request-weight", type=float, default=0.0)
    parser.add_argument("--wrec-cross-layer-weight", type=float, default=0.0)
    parser.add_argument("--wrec-contention-penalty", type=float, default=0.0)
    parser.add_argument("--wrec-history-size", type=int, default=64)
    parser.add_argument("--decision-log", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-sanity-tests", action="store_true")
    args = parser.parse_args()

    if args.run_sanity_tests:
        run_sanity_tests()

    refs, metadata = load_event_trace(args.trace)
    static_refs = refs
    if args.static_hot_trace is not None:
        static_refs, _ = load_event_trace(args.static_hot_trace)

    num_experts = int(metadata["num_experts"])
    expert_bytes = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    capacities = parse_cache_values(args.cache_capacities, num_experts)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    wrec_policies = {"wrec_h", "wrec_h2"}
    wrec_stats = None
    if any(policy in wrec_policies for policy in policies):
        wrec_stats = build_wrec_stats(
            static_refs,
            num_layers=int(metadata["num_layers"]),
            num_experts=num_experts,
            window_size=args.window_size,
            expert_bytes=expert_bytes,
            bandwidth_gbps=args.bandwidth_gbps,
        )

    rows: list[dict[str, Any]] = []
    decision_handle = None
    try:
        if args.decision_log is not None:
            args.decision_log.parent.mkdir(parents=True, exist_ok=True)
            decision_handle = args.decision_log.open("w", encoding="utf-8")
        for capacity in capacities:
            for policy in policies:
                policy_prefetch_queue_depth = (
                    args.wrec_prefetch_queue_depth
                    if policy in wrec_policies and args.wrec_prefetch_queue_depth is not None
                    else args.prefetch_queue_depth
                )
                rows.append(
                    simulate_policy(
                        refs,
                        metadata=metadata,
                        policy=policy,
                        capacity=capacity,
                        num_experts=num_experts,
                        expert_bytes=expert_bytes,
                        bandwidth_gbps=args.bandwidth_gbps,
                        static_hot_refs=static_refs,
                        window_size=args.window_size,
                        prefetch_queue_depth=policy_prefetch_queue_depth,
                        wrec_stats=wrec_stats,
                        wrec_recent_weight=args.wrec_recent_weight,
                        wrec_request_weight=args.wrec_request_weight,
                        wrec_cross_layer_weight=args.wrec_cross_layer_weight,
                        wrec_contention_penalty=args.wrec_contention_penalty,
                        wrec_history_size=args.wrec_history_size,
                        decisions=decision_handle if policy in wrec_policies else None,
                    )
                )
    finally:
        if decision_handle is not None:
            decision_handle.close()
    add_oracle_gap(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_metadata": metadata,
        "simulator": {
            "dtype": args.dtype,
            "expert_bytes": expert_bytes,
            "bandwidth_gbps": args.bandwidth_gbps,
            "cache_capacities": capacities,
            "policies": policies,
            "window_size": args.window_size,
            "prefetch_queue_depth": args.prefetch_queue_depth,
            "wrec_prefetch_queue_depth": args.wrec_prefetch_queue_depth,
            "wrec_recent_weight": args.wrec_recent_weight,
            "wrec_request_weight": args.wrec_request_weight,
            "wrec_cross_layer_weight": args.wrec_cross_layer_weight,
            "wrec_contention_penalty": args.wrec_contention_penalty,
            "wrec_history_size": args.wrec_history_size,
            "static_hot_trace": str(args.static_hot_trace or args.trace),
            "belady_initial_cache": "oracle_frequency_from_eval_trace",
            "belady_admission": "future_next_use_or_bypass",
            "wrec_h_reference_trace": (
                str(args.static_hot_trace or args.trace)
                if any(policy in wrec_policies for policy in policies)
                else None
            ),
            "decision_log": str(args.decision_log) if args.decision_log is not None else None,
        },
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(args.output_json)
    print(args.output_csv)


if __name__ == "__main__":
    main()
