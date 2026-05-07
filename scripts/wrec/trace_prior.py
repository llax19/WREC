#!/usr/bin/env python3
"""Offline WREC statistics builders from router event traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class WrecStats:
    p_window_use: dict[int, dict[int, float]]
    expected_routed_tokens: dict[int, dict[int, float]]
    base_score: dict[int, dict[int, float]]
    train_frequency: dict[int, dict[int, float]]
    cross_layer_transition: dict[int, dict[int, dict[int, float]]]


def build_layer_event_experts(refs: Iterable[Any]) -> dict[int, list[tuple[int, ...]]]:
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
    refs: Iterable[Any],
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
    refs: list[Any],
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
    refs: list[Any],
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
