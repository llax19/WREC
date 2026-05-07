#!/usr/bin/env python3
"""Online state for WREC expert cache decisions."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


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


def update_wrec_history(online: WrecOnlineState, ref: Any) -> None:
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
