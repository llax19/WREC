#!/usr/bin/env python3
"""Reusable WREC expert cache policy primitives.

This module keeps WREC policy logic independent from one-off replay drivers.
Callers provide lightweight cache/state objects with the attributes used here:
`resident`, `last_touch`, `lru_order`, and `prefetched_unused`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class WrecPolicyConfig:
    recent_weight: float
    request_weight: float
    cross_layer_weight: float
    contention_penalty: float = 0.0


@dataclass(frozen=True)
class WrecConstrainedConfig:
    min_slots_per_layer: int
    max_slots_per_layer: int
    candidates_per_layer: int
    prefetch_queue_depth: int = 0
    overlap_ms: float = 0.0


@dataclass(frozen=True)
class WrecScoreBreakdown:
    base: float
    recent: float
    request: float
    cross_layer: float
    contention: float

    @property
    def total(self) -> float:
        return self.base + self.recent + self.request + self.cross_layer - self.contention


class WrecExpertCachePolicy:
    """WREC admission, eviction, and constrained planning policy.

    The policy is intentionally runtime-agnostic: it scores `(layer, expert)`
    residency candidates from WREC stats plus online route history, while the
    caller owns transfer execution and metric accounting.
    """

    def __init__(self, config: WrecPolicyConfig) -> None:
        self.config = config

    def score_breakdown(
        self,
        *,
        layer: int,
        token_pos: int,
        expert: int,
        timestamp: int,
        cache: Any,
        online: Any,
        stats: Any,
    ) -> WrecScoreBreakdown:
        recent = online.recent_refs[layer]
        recent_count = online.recent_counts[layer].get(expert, 0)
        recent_prob = recent_count / len(recent) if recent else 0.0

        request_total = online.request_totals.get(layer, 0)
        request_count = online.request_counts[layer].get(expert, 0)
        request_prob = request_count / request_total if request_total else 0.0

        base = stats.base_score[layer].get(expert, 0.0)
        recent_score = self.config.recent_weight * recent_prob
        request_score = self.config.request_weight * request_prob
        cross_score = 0.0
        if self.config.cross_layer_weight > 0.0 and layer > 0:
            previous_experts = online.token_layer_experts.get((layer - 1, token_pos), ())
            if previous_experts:
                transition = stats.cross_layer_transition.get(layer, {})
                cross_prob = sum(
                    transition.get(previous, {}).get(expert, 0.0)
                    for previous in previous_experts
                ) / len(previous_experts)
                cross_score = self.config.cross_layer_weight * cross_prob

        contention_score = 0.0
        key = (layer, expert)
        if key in cache.resident:
            age = max(0, timestamp - cache.last_touch.get(key, timestamp))
            history_len = max(1, recent.maxlen or 1)
            contention_score = self.config.contention_penalty * min(1.0, age / history_len)

        return WrecScoreBreakdown(
            base=base,
            recent=recent_score,
            request=request_score,
            cross_layer=cross_score,
            contention=contention_score,
        )

    def score(
        self,
        *,
        layer: int,
        token_pos: int,
        expert: int,
        timestamp: int,
        cache: Any,
        online: Any,
        stats: Any,
    ) -> float:
        return self.score_breakdown(
            layer=layer,
            token_pos=token_pos,
            expert=expert,
            timestamp=timestamp,
            cache=cache,
            online=online,
            stats=stats,
        ).total

    def touch(self, cache: Any, key: ExpertKey, timestamp: int) -> None:
        cache.last_touch[key] = timestamp
        lru_order = getattr(cache, "lru_order", None)
        if lru_order is None:
            return
        if key in lru_order:
            lru_order.move_to_end(key)
        else:
            lru_order[key] = None

    def remove_resident(self, cache: Any, key: ExpertKey) -> bool:
        cache.resident.remove(key)
        cache.last_touch.pop(key, None)
        lru_order = getattr(cache, "lru_order", None)
        if lru_order is not None:
            lru_order.pop(key, None)
        wasted = key in cache.prefetched_unused
        cache.prefetched_unused.discard(key)
        return wasted

    def add_resident(self, cache: Any, key: ExpertKey, timestamp: int, *, prefetched: bool) -> None:
        cache.resident.add(key)
        self.touch(cache, key, timestamp)
        if prefetched:
            cache.prefetched_unused.add(key)

    def choose_victim(
        self,
        cache: Any,
        *,
        token_pos: int,
        timestamp: int,
        online: Any,
        stats: Any,
        candidates: set[ExpertKey] | list[ExpertKey] | None = None,
    ) -> ExpertKey | None:
        victim_candidates = cache.resident if candidates is None else candidates
        if not victim_candidates:
            return None
        return min(
            victim_candidates,
            key=lambda key: self.score(
                layer=key[0],
                token_pos=token_pos,
                expert=key[1],
                timestamp=timestamp,
                cache=cache,
                online=online,
                stats=stats,
            ),
        )

    def evict(
        self,
        cache: Any,
        *,
        token_pos: int,
        timestamp: int,
        online: Any,
        stats: Any,
        candidates: set[ExpertKey] | list[ExpertKey] | None = None,
    ) -> tuple[ExpertKey | None, bool]:
        victim = self.choose_victim(
            cache,
            token_pos=token_pos,
            timestamp=timestamp,
            online=online,
            stats=stats,
            candidates=candidates,
        )
        if victim is None:
            return None, False
        wasted = self.remove_resident(cache, victim)
        return victim, wasted

    def admit_or_bypass(
        self,
        cache: Any,
        *,
        ref: Any,
        total_slots: int,
        online: Any,
        stats: Any,
        prefetched: bool = False,
    ) -> bool:
        key = (ref.layer, ref.expert)
        if total_slots <= 0:
            return False
        if key in cache.resident:
            self.touch(cache, key, ref.ref_index)
            return False

        incoming_score = self.score(
            layer=ref.layer,
            token_pos=ref.token_pos,
            expert=ref.expert,
            timestamp=ref.ref_index,
            cache=cache,
            online=online,
            stats=stats,
        )
        wasted = False
        if len(cache.resident) >= total_slots:
            victim = self.choose_victim(
                cache,
                token_pos=ref.token_pos,
                timestamp=ref.ref_index,
                online=online,
                stats=stats,
            )
            if victim is None:
                return False
            victim_score = self.score(
                layer=victim[0],
                token_pos=ref.token_pos,
                expert=victim[1],
                timestamp=ref.ref_index,
                cache=cache,
                online=online,
                stats=stats,
            )
            if incoming_score <= victim_score:
                return False
            wasted = self.remove_resident(cache, victim)

        self.add_resident(cache, key, ref.ref_index, prefetched=prefetched)
        return wasted

    def constrained_target_set(
        self,
        *,
        ref: Any,
        cache: Any,
        total_slots: int,
        num_layers: int,
        num_experts: int,
        online: Any,
        stats: Any,
        min_slots_per_layer: int,
        max_slots_per_layer: int,
        candidate_keys: set[ExpertKey],
    ) -> set[ExpertKey]:
        if total_slots <= 0:
            return set()
        min_slots = max(0, min(min_slots_per_layer, num_experts))
        max_slots = max(min_slots, min(max_slots_per_layer, num_experts))
        if min_slots * num_layers > total_slots:
            min_slots = total_slots // num_layers

        scored: list[tuple[float, ExpertKey]] = []
        by_layer: dict[int, list[tuple[float, ExpertKey]]] = {}
        bounded_candidates = {
            (layer, expert)
            for layer, expert in candidate_keys
            if 0 <= layer < num_layers and 0 <= expert < num_experts
        }
        for layer, expert in bounded_candidates:
            key = (layer, expert)
            score = self.score(
                layer=layer,
                token_pos=ref.token_pos,
                expert=expert,
                timestamp=ref.ref_index,
                cache=cache,
                online=online,
                stats=stats,
            )
            item = (score, key)
            scored.append(item)
            by_layer.setdefault(layer, []).append(item)

        target: set[ExpertKey] = set()
        layer_counts: Counter[int] = Counter()
        if min_slots > 0:
            for layer in range(num_layers):
                for _, key in sorted(by_layer.get(layer, []), reverse=True)[:min_slots]:
                    if len(target) >= total_slots:
                        break
                    target.add(key)
                    layer_counts[layer] += 1

        for _, key in sorted(scored, reverse=True):
            if len(target) >= total_slots:
                break
            if key in target:
                continue
            layer = key[0]
            if layer_counts[layer] >= max_slots:
                continue
            target.add(key)
            layer_counts[layer] += 1
        return target

    def evict_constrained(
        self,
        cache: Any,
        *,
        ref: Any,
        target: set[ExpertKey],
        online: Any,
        stats: Any,
    ) -> tuple[ExpertKey | None, bool]:
        candidates = [key for key in cache.resident if key not in target]
        if not candidates:
            candidates = list(cache.resident)
        return self.evict(
            cache,
            token_pos=ref.token_pos,
            timestamp=ref.ref_index,
            online=online,
            stats=stats,
            candidates=candidates,
        )

    def admit_constrained(
        self,
        cache: Any,
        *,
        ref: Any,
        total_slots: int,
        num_layers: int,
        num_experts: int,
        online: Any,
        stats: Any,
        min_slots_per_layer: int,
        max_slots_per_layer: int,
    ) -> bool:
        key = (ref.layer, ref.expert)
        if total_slots <= 0:
            return False
        if key in cache.resident:
            self.touch(cache, key, ref.ref_index)
            return False

        target = self.constrained_target_set(
            ref=ref,
            cache=cache,
            total_slots=total_slots,
            num_layers=num_layers,
            num_experts=num_experts,
            online=online,
            stats=stats,
            min_slots_per_layer=min_slots_per_layer,
            max_slots_per_layer=max_slots_per_layer,
            candidate_keys=set(cache.resident) | {key},
        )
        if key not in target:
            return False
        if len(cache.resident) >= total_slots:
            _, wasted = self.evict_constrained(
                cache,
                ref=ref,
                target=target,
                online=online,
                stats=stats,
            )
        else:
            wasted = False

        self.add_resident(cache, key, ref.ref_index, prefetched=False)
        return wasted

    def build_constrained_candidates(
        self,
        *,
        ref: Any,
        cache: Any,
        num_layers: int,
        num_experts: int,
        online: Any,
        stats: Any,
        candidates_per_layer: int,
    ) -> set[ExpertKey]:
        candidate_keys = set(cache.resident)
        train_top_k = max(0, min(candidates_per_layer, num_experts))
        if train_top_k > 0:
            for layer in range(num_layers):
                ranked = sorted(
                    stats.train_frequency[layer].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                candidate_keys.update((layer, expert) for expert, _ in ranked[:train_top_k])
        for layer, counts in online.recent_counts.items():
            candidate_keys.update((layer, expert) for expert in counts)
        for layer, counts in online.request_counts.items():
            candidate_keys.update((layer, expert) for expert in counts)
        if self.config.cross_layer_weight > 0.0 and ref.layer > 0:
            previous_experts = online.token_layer_experts.get((ref.layer - 1, ref.token_pos), ())
            transition = stats.cross_layer_transition.get(ref.layer, {})
            for previous in previous_experts:
                ranked = sorted(
                    transition.get(previous, {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                candidate_keys.update((ref.layer, expert) for expert, _ in ranked[:train_top_k])
        return candidate_keys

    def replan_constrained_prefetch(
        self,
        cache: Any,
        *,
        ref: Any,
        total_slots: int,
        num_layers: int,
        num_experts: int,
        queue_depth: int,
        overlap_ms: float,
        expert_bytes: float,
        bandwidth_gbps: float,
        online: Any,
        stats: Any,
        min_slots_per_layer: int,
        max_slots_per_layer: int,
        candidates_per_layer: int,
    ) -> tuple[int, int]:
        if total_slots <= 0 or queue_depth <= 0 or overlap_ms <= 0.0:
            return 0, 0
        transfer_ms = expert_bytes / (bandwidth_gbps * 1e9) * 1000.0
        overlap_limited = int(overlap_ms // transfer_ms) if transfer_ms > 0.0 else queue_depth
        allowed = min(queue_depth, overlap_limited)
        if allowed <= 0:
            return 0, 0

        candidate_keys = self.build_constrained_candidates(
            ref=ref,
            cache=cache,
            num_layers=num_layers,
            num_experts=num_experts,
            online=online,
            stats=stats,
            candidates_per_layer=candidates_per_layer,
        )
        target = self.constrained_target_set(
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
        candidates = [key for key in target if key not in cache.resident]
        candidates.sort(
            key=lambda key: self.score(
                layer=key[0],
                token_pos=ref.token_pos,
                expert=key[1],
                timestamp=ref.ref_index,
                cache=cache,
                online=online,
                stats=stats,
            ),
            reverse=True,
        )

        loaded = 0
        wasted_count = 0
        for key in candidates:
            if loaded >= allowed:
                break
            if len(cache.resident) >= total_slots:
                _, wasted = self.evict_constrained(
                    cache,
                    ref=ref,
                    target=target,
                    online=online,
                    stats=stats,
                )
            else:
                wasted = False
            self.add_resident(cache, key, ref.ref_index, prefetched=True)
            loaded += 1
            wasted_count += int(wasted)
        return loaded, wasted_count
