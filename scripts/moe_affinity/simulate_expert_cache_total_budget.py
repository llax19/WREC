#!/usr/bin/env python3
"""Replay MoE expert cache policies under a fixed total expert-cache budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from simulate_expert_cache_offload import (
    ExpertRef,
    build_future_queues,
    infer_expert_bytes,
    load_event_trace,
    next_use,
)
from wrec import (
    WrecExpertCachePolicy,
    WrecOnlineState,
    WrecPolicyConfig,
    WrecStats,
    build_wrec_stats,
    init_wrec_online_state,
    update_wrec_history,
)


@dataclass
class GlobalCache:
    resident: set[tuple[int, int]]
    last_touch: dict[tuple[int, int], int]
    lru_order: OrderedDict[tuple[int, int], None]
    prefetched_unused: set[tuple[int, int]]


@dataclass
class LayeredCache:
    resident: dict[int, set[int]]
    last_touch: dict[int, dict[int, int]]
    lru_order: dict[int, OrderedDict[int, None]]
    prefetched_unused: set[tuple[int, int]]


def make_wrec_policy(
    *,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> WrecExpertCachePolicy:
    return WrecExpertCachePolicy(
        WrecPolicyConfig(
            recent_weight=recent_weight,
            request_weight=request_weight,
            cross_layer_weight=cross_layer_weight,
            contention_penalty=contention_penalty,
        )
    )


def parse_fractions(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.endswith("%"):
            values.append(float(item[:-1]) / 100.0)
        else:
            value = float(item)
            values.append(value / 100.0 if value > 1.0 else value)
    if not values:
        raise ValueError("at least one budget fraction is required")
    return sorted(set(values))


def total_slots_for_fraction(num_layers: int, num_experts: int, fraction: float) -> int:
    return max(0, min(num_layers * num_experts, round(num_layers * num_experts * fraction)))


def uniform_layer_capacities(total_slots: int, num_layers: int, num_experts: int) -> dict[int, int]:
    base = min(num_experts, total_slots // num_layers)
    remainder = max(0, total_slots - base * num_layers)
    capacities = {layer: base for layer in range(num_layers)}
    for layer in range(num_layers):
        if remainder <= 0:
            break
        if capacities[layer] < num_experts:
            capacities[layer] += 1
            remainder -= 1
    return capacities


def layer_expert_counts(refs: list[ExpertRef]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for ref in refs:
        counts[ref.layer][ref.expert] += 1
    return counts


def top_global_hot(refs: list[ExpertRef], total_slots: int) -> set[tuple[int, int]]:
    counts = Counter((ref.layer, ref.expert) for ref in refs)
    return {key for key, _ in counts.most_common(total_slots)}


def init_global_cache(initial: set[tuple[int, int]]) -> GlobalCache:
    order = OrderedDict((key, None) for key in initial)
    return GlobalCache(
        resident=set(initial),
        last_touch={key: 0 for key in initial},
        lru_order=order,
        prefetched_unused=set(),
    )


def init_layered_cache(num_layers: int) -> LayeredCache:
    return LayeredCache(
        resident={layer: set() for layer in range(num_layers)},
        last_touch={layer: {} for layer in range(num_layers)},
        lru_order={layer: OrderedDict() for layer in range(num_layers)},
        prefetched_unused=set(),
    )


def layered_touch(cache: LayeredCache, layer: int, expert: int, timestamp: int) -> None:
    cache.last_touch[layer][expert] = timestamp
    if expert in cache.lru_order[layer]:
        cache.lru_order[layer].move_to_end(expert)
    else:
        cache.lru_order[layer][expert] = None


def evict_layer_lru(cache: LayeredCache, layer: int) -> tuple[int | None, bool]:
    if not cache.lru_order[layer]:
        return None, False
    victim, _ = cache.lru_order[layer].popitem(last=False)
    cache.resident[layer].remove(victim)
    cache.last_touch[layer].pop(victim, None)
    key = (layer, victim)
    wasted = key in cache.prefetched_unused
    cache.prefetched_unused.discard(key)
    return victim, wasted


def load_layer_lru(
    cache: LayeredCache,
    *,
    layer: int,
    expert: int,
    capacity: int,
    timestamp: int,
    prefetched: bool,
) -> bool:
    if capacity <= 0:
        return False
    if expert in cache.resident[layer]:
        layered_touch(cache, layer, expert, timestamp)
        return False
    wasted = False
    if len(cache.resident[layer]) >= capacity:
        _, wasted = evict_layer_lru(cache, layer)
    cache.resident[layer].add(expert)
    layered_touch(cache, layer, expert, timestamp)
    if prefetched:
        cache.prefetched_unused.add((layer, expert))
    return wasted


def lru_touch(cache: GlobalCache, key: tuple[int, int], timestamp: int) -> None:
    cache.last_touch[key] = timestamp
    if key in cache.lru_order:
        cache.lru_order.move_to_end(key)
    else:
        cache.lru_order[key] = None


def evict_lru(cache: GlobalCache) -> tuple[tuple[int, int] | None, bool]:
    if not cache.lru_order:
        return None, False
    victim, _ = cache.lru_order.popitem(last=False)
    cache.resident.remove(victim)
    cache.last_touch.pop(victim, None)
    wasted = victim in cache.prefetched_unused
    cache.prefetched_unused.discard(victim)
    return victim, wasted


def load_lru(
    cache: GlobalCache,
    *,
    key: tuple[int, int],
    total_slots: int,
    timestamp: int,
    prefetched: bool,
) -> bool:
    if total_slots <= 0:
        return False
    if key in cache.resident:
        lru_touch(cache, key, timestamp)
        return False
    wasted = False
    if len(cache.resident) >= total_slots:
        _, wasted = evict_lru(cache)
    cache.resident.add(key)
    lru_touch(cache, key, timestamp)
    if prefetched:
        cache.prefetched_unused.add(key)
    return wasted


def make_uniform_initial(refs: list[ExpertRef], total_slots: int, num_layers: int, num_experts: int) -> set[tuple[int, int]]:
    capacities = uniform_layer_capacities(total_slots, num_layers, num_experts)
    counts = layer_expert_counts(refs)
    initial: set[tuple[int, int]] = set()
    for layer, capacity in capacities.items():
        for expert, _ in counts[layer].most_common(capacity):
            initial.add((layer, expert))
    return initial


def pop_current_future(future: dict[tuple[int, int], deque[int]], ref: ExpertRef) -> None:
    queue = future[(ref.layer, ref.expert)]
    if queue and queue[0] == ref.ref_index:
        queue.popleft()


def admit_belady_or_bypass(
    cache: GlobalCache,
    *,
    key: tuple[int, int],
    total_slots: int,
    future: dict[tuple[int, int], deque[int]],
    timestamp: int,
) -> bool:
    if total_slots <= 0:
        return False
    if key in cache.resident:
        lru_touch(cache, key, timestamp)
        return False
    if len(cache.resident) < total_slots:
        cache.resident.add(key)
        lru_touch(cache, key, timestamp)
        return False

    victim = max(cache.resident, key=lambda item: next_use(future, item[0], item[1]))
    if next_use(future, key[0], key[1]) >= next_use(future, victim[0], victim[1]):
        return False
    cache.resident.remove(victim)
    cache.lru_order.pop(victim, None)
    cache.last_touch.pop(victim, None)
    wasted = victim in cache.prefetched_unused
    cache.prefetched_unused.discard(victim)
    cache.resident.add(key)
    lru_touch(cache, key, timestamp)
    return wasted


def wrec_total_score(
    *,
    layer: int,
    token_pos: int,
    expert: int,
    key: tuple[int, int],
    timestamp: int,
    cache: GlobalCache,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> float:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).score(
        layer=layer,
        token_pos=token_pos,
        expert=expert,
        timestamp=timestamp,
        cache=cache,
        online=online,
        stats=stats,
    )


def evict_wrec_global(
    cache: GlobalCache,
    *,
    ref: ExpertRef,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> tuple[tuple[int, int] | None, bool]:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).evict(
        cache,
        token_pos=ref.token_pos,
        timestamp=ref.ref_index,
        online=online,
        stats=stats,
    )


def wrec_c_target_set(
    *,
    ref: ExpertRef,
    cache: GlobalCache,
    total_slots: int,
    num_layers: int,
    num_experts: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
    min_slots_per_layer: int,
    max_slots_per_layer: int,
    candidate_keys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).constrained_target_set(
        ref=ref,
        cache=cache,
        total_slots=total_slots,
        num_layers=num_layers,
        num_experts=num_experts,
        online=online,
        stats=stats,
        min_slots_per_layer=min_slots_per_layer,
        max_slots_per_layer=max_slots_per_layer,
        candidate_keys=candidate_keys,
    )


def evict_wrec_c_global(
    cache: GlobalCache,
    *,
    ref: ExpertRef,
    target: set[tuple[int, int]],
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> tuple[tuple[int, int] | None, bool]:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).evict_constrained(
        cache,
        ref=ref,
        target=target,
        online=online,
        stats=stats,
    )


def admit_wrec_c_global(
    cache: GlobalCache,
    *,
    ref: ExpertRef,
    total_slots: int,
    num_layers: int,
    num_experts: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
    min_slots_per_layer: int,
    max_slots_per_layer: int,
) -> bool:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).admit_constrained(
        cache,
        ref=ref,
        total_slots=total_slots,
        num_layers=num_layers,
        num_experts=num_experts,
        online=online,
        stats=stats,
        min_slots_per_layer=min_slots_per_layer,
        max_slots_per_layer=max_slots_per_layer,
    )


def replan_wrec_c_prefetch(
    cache: GlobalCache,
    *,
    ref: ExpertRef,
    total_slots: int,
    num_layers: int,
    num_experts: int,
    queue_depth: int,
    overlap_ms: float,
    expert_bytes: float,
    bandwidth_gbps: float,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
    min_slots_per_layer: int,
    max_slots_per_layer: int,
    candidates_per_layer: int,
) -> tuple[int, int]:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).replan_constrained_prefetch(
        cache,
        ref=ref,
        total_slots=total_slots,
        num_layers=num_layers,
        num_experts=num_experts,
        queue_depth=queue_depth,
        overlap_ms=overlap_ms,
        expert_bytes=expert_bytes,
        bandwidth_gbps=bandwidth_gbps,
        online=online,
        stats=stats,
        min_slots_per_layer=min_slots_per_layer,
        max_slots_per_layer=max_slots_per_layer,
        candidates_per_layer=candidates_per_layer,
    )


def admit_wrec_global(
    cache: GlobalCache,
    *,
    ref: ExpertRef,
    total_slots: int,
    online: WrecOnlineState,
    stats: WrecStats,
    recent_weight: float,
    request_weight: float,
    cross_layer_weight: float,
    contention_penalty: float,
) -> bool:
    return make_wrec_policy(
        recent_weight=recent_weight,
        request_weight=request_weight,
        cross_layer_weight=cross_layer_weight,
        contention_penalty=contention_penalty,
    ).admit_or_bypass(
        cache,
        ref=ref,
        total_slots=total_slots,
        online=online,
        stats=stats,
    )


def prefetch_future_global(
    cache: GlobalCache,
    *,
    refs: list[ExpertRef],
    current_index: int,
    total_slots: int,
    window_size: int,
    queue_depth: int,
) -> tuple[int, int]:
    if total_slots <= 0 or window_size <= 0 or queue_depth <= 0:
        return 0, 0
    loaded = 0
    wasted = 0
    seen: set[tuple[int, int]] = set()
    upper = min(len(refs), current_index + 1 + window_size)
    for future_ref in refs[current_index + 1 : upper]:
        key = (future_ref.layer, future_ref.expert)
        if key in seen or key in cache.resident:
            continue
        seen.add(key)
        wasted += int(load_lru(cache, key=key, total_slots=total_slots, timestamp=refs[current_index].ref_index, prefetched=True))
        loaded += 1
        if loaded >= queue_depth:
            break
    return loaded, wasted


def prefetch_future_layered(
    cache: LayeredCache,
    *,
    refs: list[ExpertRef],
    current_index: int,
    capacities: dict[int, int],
    window_size: int,
    queue_depth: int,
) -> tuple[int, int]:
    if window_size <= 0 or queue_depth <= 0:
        return 0, 0
    current = refs[current_index]
    loaded = 0
    wasted = 0
    seen: set[int] = set()
    upper = min(len(refs), current_index + 1 + window_size)
    for future_ref in refs[current_index + 1 : upper]:
        if future_ref.layer != current.layer:
            continue
        if future_ref.expert in seen or future_ref.expert in cache.resident[future_ref.layer]:
            continue
        seen.add(future_ref.expert)
        wasted += int(
            load_layer_lru(
                cache,
                layer=future_ref.layer,
                expert=future_ref.expert,
                capacity=capacities.get(future_ref.layer, 0),
                timestamp=current.ref_index,
                prefetched=True,
            )
        )
        loaded += 1
        if loaded >= queue_depth:
            break
    return loaded, wasted


def simulate_uniform_layer_policy(
    refs: list[ExpertRef],
    *,
    metadata: dict[str, Any],
    policy: str,
    total_slots: int,
    total_budget_fraction: float,
    expert_bytes: float,
    bandwidth_gbps: float,
    window_size: int,
    prefetch_queue_depth: int,
) -> dict[str, Any]:
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])
    capacities = uniform_layer_capacities(total_slots, num_layers, num_experts)
    actual_total_slots = sum(capacities.values())
    cache = init_layered_cache(num_layers)
    demand_hits = 0
    demand_misses = 0
    demand_loads = 0
    prefetch_loads = 0
    prefetch_waste_count = 0

    for ref in refs:
        if policy == "route_window_prefetch":
            loaded, wasted = prefetch_future_layered(
                cache,
                refs=refs,
                current_index=ref.ref_index,
                capacities=capacities,
                window_size=window_size,
                queue_depth=prefetch_queue_depth,
            )
            prefetch_loads += loaded
            prefetch_waste_count += wasted

        if ref.expert in cache.resident[ref.layer]:
            demand_hits += 1
            layered_touch(cache, ref.layer, ref.expert, ref.ref_index)
            cache.prefetched_unused.discard((ref.layer, ref.expert))
            continue

        demand_misses += 1
        demand_loads += 1
        prefetch_waste_count += int(
            load_layer_lru(
                cache,
                layer=ref.layer,
                expert=ref.expert,
                capacity=capacities.get(ref.layer, 0),
                timestamp=ref.ref_index,
                prefetched=False,
            )
        )

    prefetch_waste_count += len(cache.prefetched_unused)
    demand_transfer_bytes = demand_loads * expert_bytes
    prefetch_transfer_bytes = prefetch_loads * expert_bytes
    transfer_bytes = demand_transfer_bytes + prefetch_transfer_bytes
    stall_ms = demand_transfer_bytes / (bandwidth_gbps * 1e9) * 1000.0
    total_input_tokens = max(1, int(metadata["num_input_tokens"]))
    total_token_layer_events = max(1, int(metadata["num_token_layer_events"]))
    total_refs = len(refs)
    return {
        "policy": policy,
        "allocation_mode": "uniform_per_layer",
        "total_cache_slots": actual_total_slots,
        "total_budget_fraction": total_budget_fraction,
        "total_cache_bytes": actual_total_slots * expert_bytes,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_bytes": expert_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "window_size": window_size if policy == "route_window_prefetch" else 0,
        "prefetch_queue_depth": prefetch_queue_depth if policy == "route_window_prefetch" else 0,
        "wrec_recent_weight": 0.0,
        "wrec_request_weight": 0.0,
        "wrec_cross_layer_weight": 0.0,
        "wrec_contention_penalty": 0.0,
        "wrec_history_size": 0,
        "total_expert_refs": total_refs,
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "cache_hit_rate": demand_hits / total_refs if total_refs else 0.0,
        "workload_weighted_miss_rate": demand_misses / total_refs if total_refs else 0.0,
        "demand_loads": demand_loads,
        "prefetch_loads": prefetch_loads,
        "prefetch_waste_count": prefetch_waste_count,
        "transfer_bytes": transfer_bytes,
        "demand_transfer_bytes": demand_transfer_bytes,
        "prefetch_transfer_bytes": prefetch_transfer_bytes,
        "waste_bytes": prefetch_waste_count * expert_bytes,
        "stall_ms": stall_ms,
        "stall_ms_per_input_token": stall_ms / total_input_tokens,
        "stall_ms_per_token_layer_event": stall_ms / total_token_layer_events,
        "transfer_bytes_per_input_token": transfer_bytes / total_input_tokens,
        "transfer_bytes_per_token_layer_event": transfer_bytes / total_token_layer_events,
        "waste_bytes_per_input_token": (prefetch_waste_count * expert_bytes) / total_input_tokens,
    }


def simulate_total_budget_policy(
    refs: list[ExpertRef],
    *,
    metadata: dict[str, Any],
    static_refs: list[ExpertRef],
    policy: str,
    total_slots: int,
    total_budget_fraction: float,
    expert_bytes: float,
    bandwidth_gbps: float,
    window_size: int,
    prefetch_queue_depth: int,
    wrec_stats: WrecStats | None,
    wrec_recent_weight: float,
    wrec_request_weight: float,
    wrec_cross_layer_weight: float,
    wrec_contention_penalty: float,
    wrec_history_size: int,
    wrec_c_prefetch_queue_depth: int,
    wrec_c_overlap_ms: float,
    wrec_c_min_slots_per_layer: int,
    wrec_c_max_slots_per_layer: int | None,
    wrec_c_candidates_per_layer: int,
    wrec_c_replan_interval: int,
) -> dict[str, Any]:
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])
    total_possible_slots = num_layers * num_experts
    total_slots = max(0, min(total_slots, total_possible_slots))

    allocation_mode = {
        "on_demand": "none",
        "lru": "uniform_per_layer",
        "static_hot": "global_static_hot",
        "route_window_prefetch": "uniform_per_layer_future_window",
        "belady_oracle": "global_belady",
        "wrec_h": "global_wrec_adaptive",
        "wrec_h2": "global_wrec_adaptive",
        "wrec_c": "global_wrec_constrained",
    }[policy]

    if policy in {"lru", "route_window_prefetch"}:
        return simulate_uniform_layer_policy(
            refs,
            metadata=metadata,
            policy=policy,
            total_slots=total_slots,
            total_budget_fraction=total_budget_fraction,
            expert_bytes=expert_bytes,
            bandwidth_gbps=bandwidth_gbps,
            window_size=window_size,
            prefetch_queue_depth=prefetch_queue_depth,
        )

    if policy == "on_demand":
        initial: set[tuple[int, int]] = set()
    elif policy == "lru":
        initial = make_uniform_initial(static_refs, total_slots, num_layers, num_experts)
    elif policy == "static_hot":
        initial = top_global_hot(static_refs, total_slots)
    elif policy == "belady_oracle":
        initial = top_global_hot(refs, total_slots)
    elif policy == "route_window_prefetch":
        initial = make_uniform_initial(static_refs, total_slots, num_layers, num_experts)
    elif policy in {"wrec_h", "wrec_h2", "wrec_c"}:
        initial = top_global_hot(static_refs, total_slots)
    else:
        raise ValueError(f"unknown policy: {policy}")

    cache = init_global_cache(initial)
    future = build_future_queues(refs)
    wrec_online = None
    wrec_policies = {"wrec_h", "wrec_h2", "wrec_c"}
    if policy in wrec_policies:
        if wrec_stats is None:
            raise ValueError(f"{policy} requires WREC stats")
        wrec_online = init_wrec_online_state(num_layers, wrec_history_size, decisions=None)
    wrec_policy = make_wrec_policy(
        recent_weight=wrec_recent_weight,
        request_weight=wrec_request_weight if policy in {"wrec_h2", "wrec_c"} else 0.0,
        cross_layer_weight=wrec_cross_layer_weight if policy in {"wrec_h2", "wrec_c"} else 0.0,
        contention_penalty=wrec_contention_penalty,
    ) if policy in wrec_policies else None
    wrec_c_max_slots = num_experts if wrec_c_max_slots_per_layer is None else wrec_c_max_slots_per_layer
    wrec_c_interval = max(1, wrec_c_replan_interval)

    demand_hits = 0
    demand_misses = 0
    demand_loads = 0
    prefetch_loads = 0
    prefetch_waste_count = 0

    for ref in refs:
        key = (ref.layer, ref.expert)
        pop_current_future(future, ref)
        is_event_tail = ref.ref_index == len(refs) - 1 or refs[ref.ref_index + 1].event_index != ref.event_index
        if policy == "route_window_prefetch" and total_slots < total_possible_slots:
            loaded, wasted = prefetch_future_global(
                cache,
                refs=refs,
                current_index=ref.ref_index,
                total_slots=total_slots,
                window_size=window_size,
                queue_depth=prefetch_queue_depth,
            )
            prefetch_loads += loaded
            prefetch_waste_count += wasted

        if key in cache.resident:
            demand_hits += 1
            lru_touch(cache, key, ref.ref_index)
            cache.prefetched_unused.discard(key)
            if wrec_online is not None:
                update_wrec_history(wrec_online, ref)
                if (
                    policy == "wrec_c"
                    and is_event_tail
                    and ref.event_index % wrec_c_interval == 0
                    and wrec_stats is not None
                    and wrec_policy is not None
                ):
                    loaded, wasted = wrec_policy.replan_constrained_prefetch(
                        cache,
                        ref=ref,
                        total_slots=total_slots,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        queue_depth=wrec_c_prefetch_queue_depth,
                        overlap_ms=wrec_c_overlap_ms,
                        expert_bytes=expert_bytes,
                        bandwidth_gbps=bandwidth_gbps,
                        online=wrec_online,
                        stats=wrec_stats,
                        min_slots_per_layer=wrec_c_min_slots_per_layer,
                        max_slots_per_layer=wrec_c_max_slots,
                        candidates_per_layer=wrec_c_candidates_per_layer,
                    )
                    prefetch_loads += loaded
                    prefetch_waste_count += wasted
            continue

        demand_misses += 1
        demand_loads += 1
        if policy in {"on_demand", "static_hot"}:
            continue
        if policy in {"lru", "route_window_prefetch"}:
            prefetch_waste_count += int(load_lru(cache, key=key, total_slots=total_slots, timestamp=ref.ref_index, prefetched=False))
            continue
        if policy == "belady_oracle":
            prefetch_waste_count += int(
                admit_belady_or_bypass(
                    cache,
                    key=key,
                    total_slots=total_slots,
                    future=future,
                    timestamp=ref.ref_index,
                )
            )
            continue
        if policy in wrec_policies and wrec_online is not None and wrec_stats is not None and wrec_policy is not None:
            update_wrec_history(wrec_online, ref)
            if policy == "wrec_c":
                prefetch_waste_count += int(
                    wrec_policy.admit_constrained(
                        cache,
                        ref=ref,
                        total_slots=total_slots,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        online=wrec_online,
                        stats=wrec_stats,
                        min_slots_per_layer=wrec_c_min_slots_per_layer,
                        max_slots_per_layer=wrec_c_max_slots,
                    )
                )
                if is_event_tail and ref.event_index % wrec_c_interval == 0:
                    loaded, wasted = wrec_policy.replan_constrained_prefetch(
                        cache,
                        ref=ref,
                        total_slots=total_slots,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        queue_depth=wrec_c_prefetch_queue_depth,
                        overlap_ms=wrec_c_overlap_ms,
                        expert_bytes=expert_bytes,
                        bandwidth_gbps=bandwidth_gbps,
                        online=wrec_online,
                        stats=wrec_stats,
                        min_slots_per_layer=wrec_c_min_slots_per_layer,
                        max_slots_per_layer=wrec_c_max_slots,
                        candidates_per_layer=wrec_c_candidates_per_layer,
                    )
                    prefetch_loads += loaded
                    prefetch_waste_count += wasted
            else:
                prefetch_waste_count += int(
                    wrec_policy.admit_or_bypass(
                        cache,
                        ref=ref,
                        total_slots=total_slots,
                        online=wrec_online,
                        stats=wrec_stats,
                    )
                )
            continue

    prefetch_waste_count += len(cache.prefetched_unused)
    demand_transfer_bytes = demand_loads * expert_bytes
    prefetch_transfer_bytes = prefetch_loads * expert_bytes
    transfer_bytes = demand_transfer_bytes + prefetch_transfer_bytes
    stall_ms = demand_transfer_bytes / (bandwidth_gbps * 1e9) * 1000.0
    total_input_tokens = max(1, int(metadata["num_input_tokens"]))
    total_token_layer_events = max(1, int(metadata["num_token_layer_events"]))
    total_refs = len(refs)
    return {
        "policy": policy,
        "allocation_mode": allocation_mode,
        "total_cache_slots": total_slots,
        "total_budget_fraction": total_budget_fraction,
        "total_cache_bytes": total_slots * expert_bytes,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_bytes": expert_bytes,
        "bandwidth_gbps": bandwidth_gbps,
        "window_size": window_size if policy in {"route_window_prefetch", "wrec_c"} else 0,
        "prefetch_queue_depth": (
            wrec_c_prefetch_queue_depth if policy == "wrec_c"
            else prefetch_queue_depth if policy == "route_window_prefetch"
            else 0
        ),
        "wrec_recent_weight": wrec_recent_weight if policy in wrec_policies else 0.0,
        "wrec_request_weight": wrec_request_weight if policy in {"wrec_h2", "wrec_c"} else 0.0,
        "wrec_cross_layer_weight": wrec_cross_layer_weight if policy in {"wrec_h2", "wrec_c"} else 0.0,
        "wrec_contention_penalty": wrec_contention_penalty if policy in wrec_policies else 0.0,
        "wrec_history_size": wrec_history_size if policy in wrec_policies else 0,
        "wrec_c_overlap_ms": wrec_c_overlap_ms if policy == "wrec_c" else 0.0,
        "wrec_c_min_slots_per_layer": wrec_c_min_slots_per_layer if policy == "wrec_c" else 0,
        "wrec_c_max_slots_per_layer": wrec_c_max_slots if policy == "wrec_c" else 0,
        "wrec_c_candidates_per_layer": wrec_c_candidates_per_layer if policy == "wrec_c" else 0,
        "wrec_c_replan_interval": wrec_c_interval if policy == "wrec_c" else 0,
        "total_expert_refs": total_refs,
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "cache_hit_rate": demand_hits / total_refs if total_refs else 0.0,
        "workload_weighted_miss_rate": demand_misses / total_refs if total_refs else 0.0,
        "demand_loads": demand_loads,
        "prefetch_loads": prefetch_loads,
        "prefetch_waste_count": prefetch_waste_count,
        "transfer_bytes": transfer_bytes,
        "demand_transfer_bytes": demand_transfer_bytes,
        "prefetch_transfer_bytes": prefetch_transfer_bytes,
        "waste_bytes": prefetch_waste_count * expert_bytes,
        "stall_ms": stall_ms,
        "stall_ms_per_input_token": stall_ms / total_input_tokens,
        "stall_ms_per_token_layer_event": stall_ms / total_token_layer_events,
        "transfer_bytes_per_input_token": transfer_bytes / total_input_tokens,
        "transfer_bytes_per_token_layer_event": transfer_bytes / total_token_layer_events,
        "waste_bytes_per_input_token": (prefetch_waste_count * expert_bytes) / total_input_tokens,
    }


def add_oracle_gap(rows: list[dict[str, Any]]) -> None:
    by_budget: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_budget[int(row["total_cache_slots"])][str(row["policy"])] = row
    for policies in by_budget.values():
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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--static-hot-trace", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bandwidth-gbps", type=float, default=41.37220609315469)
    parser.add_argument("--budget-fractions", default="12.5%,25%,37.5%,50%,75%")
    parser.add_argument(
        "--policies",
        default="on_demand,lru,static_hot,route_window_prefetch,belady_oracle,wrec_h,wrec_h2,wrec_c",
    )
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--prefetch-queue-depth", type=int, default=1)
    parser.add_argument("--wrec-recent-weight", type=float, default=512.0)
    parser.add_argument("--wrec-request-weight", type=float, default=1024.0)
    parser.add_argument("--wrec-cross-layer-weight", type=float, default=1024.0)
    parser.add_argument("--wrec-contention-penalty", type=float, default=0.0)
    parser.add_argument("--wrec-history-size", type=int, default=8)
    parser.add_argument("--wrec-c-prefetch-queue-depth", type=int, default=0)
    parser.add_argument("--wrec-c-overlap-ms", type=float, default=0.0)
    parser.add_argument("--wrec-c-min-slots-per-layer", type=int, default=0)
    parser.add_argument("--wrec-c-max-slots-per-layer", type=int, default=None)
    parser.add_argument("--wrec-c-candidates-per-layer", type=int, default=4)
    parser.add_argument("--wrec-c-replan-interval", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    refs, metadata = load_event_trace(args.trace)
    static_refs, _ = load_event_trace(args.static_hot_trace)
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])
    expert_bytes = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    fractions = parse_fractions(args.budget_fractions)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    wrec_stats = None
    if any(policy in {"wrec_h", "wrec_h2", "wrec_c"} for policy in policies):
        wrec_stats = build_wrec_stats(
            static_refs,
            num_layers=num_layers,
            num_experts=num_experts,
            window_size=args.window_size,
            expert_bytes=expert_bytes,
            bandwidth_gbps=args.bandwidth_gbps,
        )

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        total_slots = total_slots_for_fraction(num_layers, num_experts, fraction)
        for policy in policies:
            rows.append(
                simulate_total_budget_policy(
                    refs,
                    metadata=metadata,
                    static_refs=static_refs,
                    policy=policy,
                    total_slots=total_slots,
                    total_budget_fraction=fraction,
                    expert_bytes=expert_bytes,
                    bandwidth_gbps=args.bandwidth_gbps,
                    window_size=args.window_size,
                    prefetch_queue_depth=args.prefetch_queue_depth,
                    wrec_stats=wrec_stats,
                    wrec_recent_weight=args.wrec_recent_weight,
                    wrec_request_weight=args.wrec_request_weight,
                    wrec_cross_layer_weight=args.wrec_cross_layer_weight,
                    wrec_contention_penalty=args.wrec_contention_penalty,
                    wrec_history_size=args.wrec_history_size,
                    wrec_c_prefetch_queue_depth=args.wrec_c_prefetch_queue_depth,
                    wrec_c_overlap_ms=args.wrec_c_overlap_ms,
                    wrec_c_min_slots_per_layer=args.wrec_c_min_slots_per_layer,
                    wrec_c_max_slots_per_layer=args.wrec_c_max_slots_per_layer,
                    wrec_c_candidates_per_layer=args.wrec_c_candidates_per_layer,
                    wrec_c_replan_interval=args.wrec_c_replan_interval,
                )
            )
    add_oracle_gap(rows)

    payload = {
        "trace_metadata": metadata,
        "simulator": {
            "mode": "fixed_total_expert_cache_budget",
            "static_hot_trace": str(args.static_hot_trace),
            "dtype": args.dtype,
            "expert_bytes": expert_bytes,
            "bandwidth_gbps": args.bandwidth_gbps,
            "budget_fractions": fractions,
            "policies": policies,
            "window_size": args.window_size,
            "prefetch_queue_depth": args.prefetch_queue_depth,
            "wrec_recent_weight": args.wrec_recent_weight,
            "wrec_request_weight": args.wrec_request_weight,
            "wrec_cross_layer_weight": args.wrec_cross_layer_weight,
            "wrec_contention_penalty": args.wrec_contention_penalty,
            "wrec_history_size": args.wrec_history_size,
            "wrec_c_prefetch_queue_depth": args.wrec_c_prefetch_queue_depth,
            "wrec_c_overlap_ms": args.wrec_c_overlap_ms,
            "wrec_c_min_slots_per_layer": args.wrec_c_min_slots_per_layer,
            "wrec_c_max_slots_per_layer": args.wrec_c_max_slots_per_layer,
            "wrec_c_candidates_per_layer": args.wrec_c_candidates_per_layer,
            "wrec_c_replan_interval": args.wrec_c_replan_interval,
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(args.output_json)
    print(args.output_csv)


if __name__ == "__main__":
    main()
