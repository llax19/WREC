#!/usr/bin/env python3
"""Analyze MoE expert workload statistics from router event traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from simulate_expert_cache_offload import ExpertRef, build_static_hot, load_event_trace, parse_cache_values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(materialized),
        "min": float(min(materialized)),
        "p50": percentile(materialized, 0.50),
        "p90": percentile(materialized, 0.90),
        "p99": percentile(materialized, 0.99),
        "max": float(max(materialized)),
        "mean": float(sum(materialized) / len(materialized)),
    }


def layer_expert_counts(refs: Iterable[ExpertRef]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for ref in refs:
        counts[ref.layer][ref.expert] += 1
    return counts


def frequency_summary(refs: list[ExpertRef], num_layers: int, num_experts: int) -> dict[str, Any]:
    counts = layer_expert_counts(refs)
    layers: dict[str, Any] = {}
    for layer in range(num_layers):
        total = sum(counts[layer].values())
        ranked = counts[layer].most_common()
        layers[str(layer)] = {
            "total_refs": total,
            "experts": [
                {
                    "expert": expert,
                    "count": count,
                    "probability": count / total if total else 0.0,
                }
                for expert, count in ranked
            ],
            "entropy_bits": entropy_bits(counts[layer].values()),
            "top1_share": ranked[0][1] / total if ranked and total else 0.0,
            "top2_share": sum(count for _, count in ranked[:2]) / total if total else 0.0,
            "top4_share": sum(count for _, count in ranked[:4]) / total if total else 0.0,
        }
    global_counts = Counter()
    for counter in counts.values():
        global_counts.update(counter)
    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "global_expert_counts": [
            {"expert": expert, "count": count, "probability": count / len(refs)}
            for expert, count in global_counts.most_common()
        ],
        "layers": layers,
    }


def entropy_bits(counts: Iterable[int]) -> float:
    values = [count for count in counts if count > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def static_hot_summary(refs: list[ExpertRef], capacities: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for capacity in capacities:
        hot = build_static_hot(refs, capacity)
        result[str(capacity)] = {str(layer): sorted(experts) for layer, experts in sorted(hot.items())}
    return result


def reuse_distance_summary(refs: list[ExpertRef]) -> dict[str, Any]:
    last_ref_index: dict[tuple[int, int], int] = {}
    last_event_index: dict[tuple[int, int], int] = {}
    ref_distances: list[float] = []
    event_distances: list[float] = []
    by_layer_ref_distances: dict[int, list[float]] = defaultdict(list)

    for ref in refs:
        key = (ref.layer, ref.expert)
        if key in last_ref_index:
            ref_distance = ref.ref_index - last_ref_index[key]
            event_distance = ref.event_index - last_event_index[key]
            ref_distances.append(float(ref_distance))
            event_distances.append(float(event_distance))
            by_layer_ref_distances[ref.layer].append(float(ref_distance))
        last_ref_index[key] = ref.ref_index
        last_event_index[key] = ref.event_index

    return {
        "same_layer_expert_ref_distance": summarize_values(ref_distances),
        "same_layer_expert_router_event_distance": summarize_values(event_distances),
        "by_layer_ref_distance": {
            str(layer): summarize_values(values) for layer, values in sorted(by_layer_ref_distances.items())
        },
    }


def working_set_by_request(refs: list[ExpertRef]) -> dict[str, Any]:
    sets: dict[tuple[str, int], set[int]] = defaultdict(set)
    for ref in refs:
        sets[(ref.request_id, ref.layer)].add(ref.expert)
    sizes = [float(len(experts)) for experts in sets.values()]
    by_layer: dict[int, list[float]] = defaultdict(list)
    for (_, layer), experts in sets.items():
        by_layer[layer].append(float(len(experts)))
    return {
        "per_request_layer_unique_experts": summarize_values(sizes),
        "by_layer": {str(layer): summarize_values(values) for layer, values in sorted(by_layer.items())},
    }


def window_working_set(refs: list[ExpertRef], window_sizes: list[int]) -> dict[str, Any]:
    by_layer_events: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    current_event: tuple[str, int, int] | None = None
    current_experts: list[int] = []
    current_layer = 0
    current_index = 0

    for ref in refs:
        event_key = (ref.request_id, ref.layer, ref.token_pos)
        if current_event is None:
            current_event = event_key
            current_layer = ref.layer
            current_index = ref.event_index
        elif event_key != current_event:
            by_layer_events[current_layer].append((current_index, tuple(current_experts)))
            current_event = event_key
            current_layer = ref.layer
            current_index = ref.event_index
            current_experts = []
        current_experts.append(ref.expert)
    if current_event is not None:
        by_layer_events[current_layer].append((current_index, tuple(current_experts)))

    result: dict[str, Any] = {}
    for window_size in window_sizes:
        sizes: list[float] = []
        hit_probabilities: list[float] = []
        for events in by_layer_events.values():
            expert_lists = [experts for _, experts in events]
            for start in range(0, len(expert_lists)):
                window = expert_lists[start : start + window_size]
                if len(window) < window_size:
                    continue
                unique = set()
                refs_in_window = 0
                for experts in window:
                    unique.update(experts)
                    refs_in_window += len(experts)
                sizes.append(float(len(unique)))
                hit_probabilities.append(len(unique) / refs_in_window if refs_in_window else 0.0)
        result[str(window_size)] = {
            "unique_experts_per_layer_window": summarize_values(sizes),
            "expert_use_probability_per_window": summarize_values(hit_probabilities),
        }
    return result


def distribution(counts: Counter[int], num_experts: int) -> list[float]:
    total = sum(counts.values())
    if total <= 0:
        return [0.0 for _ in range(num_experts)]
    return [counts[expert] / total for expert in range(num_experts)]


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def hotness_shift(
    train_refs: list[ExpertRef],
    eval_refs: list[ExpertRef] | None,
    *,
    num_layers: int,
    num_experts: int,
    topks: list[int],
) -> dict[str, Any] | None:
    if eval_refs is None:
        return None
    train_counts = layer_expert_counts(train_refs)
    eval_counts = layer_expert_counts(eval_refs)
    layer_rows: dict[str, Any] = {}
    tv_values: list[float] = []
    overlap_by_k: dict[int, list[float]] = {topk: [] for topk in topks}

    for layer in range(num_layers):
        p = distribution(train_counts[layer], num_experts)
        q = distribution(eval_counts[layer], num_experts)
        tv = total_variation(p, q)
        tv_values.append(tv)
        layer_entry: dict[str, Any] = {"total_variation": tv}
        for topk in topks:
            k = min(topk, num_experts)
            train_top = {expert for expert, _ in train_counts[layer].most_common(k)}
            eval_top = {expert for expert, _ in eval_counts[layer].most_common(k)}
            overlap = len(train_top & eval_top) / k if k else 0.0
            overlap_by_k[topk].append(overlap)
            layer_entry[f"top{topk}_overlap"] = overlap
        layer_rows[str(layer)] = layer_entry

    return {
        "total_variation_by_layer": summarize_values(tv_values),
        "topk_overlap_by_layer": {
            str(topk): summarize_values(values) for topk, values in sorted(overlap_by_k.items())
        },
        "layers": layer_rows,
    }


def analyze(
    *,
    train_trace: Path,
    eval_trace: Path | None,
    window_sizes: list[int],
    hot_capacities_text: str,
) -> dict[str, Any]:
    train_refs, train_meta = load_event_trace(train_trace)
    eval_refs: list[ExpertRef] | None = None
    eval_meta: dict[str, Any] | None = None
    if eval_trace is not None:
        eval_refs, eval_meta = load_event_trace(eval_trace)

    num_experts = int(train_meta["num_experts"])
    capacities = sorted(set(parse_cache_values(hot_capacities_text, num_experts)))
    num_layers = int(train_meta["num_layers"])
    topks = sorted(set(capacities + [1, 2, 4]))

    selected_per_event = train_meta["num_expert_refs"] / train_meta["num_router_events"]
    events_per_input_token = train_meta["num_router_events"] / train_meta["num_input_tokens"]
    refs_per_input_token = train_meta["num_expert_refs"] / train_meta["num_input_tokens"]

    return {
        "inputs": {
            "train_trace": str(train_trace),
            "eval_trace": str(eval_trace) if eval_trace is not None else None,
            "window_sizes": window_sizes,
            "static_hot_capacities": capacities,
        },
        "train_trace_metadata": train_meta,
        "eval_trace_metadata": eval_meta,
        "expected_routed_tokens": {
            "selected_experts_per_router_event": selected_per_event,
            "router_events_per_input_token": events_per_input_token,
            "expert_refs_per_input_token": refs_per_input_token,
        },
        "train_expert_frequency": frequency_summary(train_refs, num_layers, num_experts),
        "static_hot_by_capacity": static_hot_summary(train_refs, capacities),
        "train_reuse_distance": reuse_distance_summary(train_refs),
        "train_working_set": {
            "by_request": working_set_by_request(train_refs),
            "by_layer_window": window_working_set(train_refs, window_sizes),
        },
        "train_eval_hotness_shift": hotness_shift(
            train_refs,
            eval_refs,
            num_layers=num_layers,
            num_experts=num_experts,
            topks=topks,
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["train_trace_metadata"]
    routed = payload["expected_routed_tokens"]
    reuse = payload["train_reuse_distance"]["same_layer_expert_ref_distance"]
    ws_req = payload["train_working_set"]["by_request"]["per_request_layer_unique_experts"]
    shift = payload["train_eval_hotness_shift"]

    lines = [
        "# Mixtral Train Workload Statistics",
        "",
        "## Inputs",
        "",
        f"- Train trace: `{payload['inputs']['train_trace']}`",
        f"- Eval trace: `{payload['inputs']['eval_trace']}`",
        f"- Static-hot capacities: `{payload['inputs']['static_hot_capacities']}`",
        f"- Window sizes: `{payload['inputs']['window_sizes']}`",
        "",
        "## Summary",
        "",
        f"- Requests: `{meta.get('num_input_tokens')}` input-token units across `{meta.get('num_router_events')}` router events.",
        f"- Expert refs: `{meta.get('num_expert_refs')}`.",
        f"- Selected experts per router event: `{routed['selected_experts_per_router_event']:.4f}`.",
        f"- Router events per input token: `{routed['router_events_per_input_token']:.4f}`.",
        f"- Expert refs per input token: `{routed['expert_refs_per_input_token']:.4f}`.",
        f"- Same-layer expert reuse distance p50/p90/p99: `{reuse['p50']:.2f}` / `{reuse['p90']:.2f}` / `{reuse['p99']:.2f}` expert refs.",
        f"- Per-request-layer working set p50/p90/p99: `{ws_req['p50']:.2f}` / `{ws_req['p90']:.2f}` / `{ws_req['p99']:.2f}` unique experts.",
    ]

    if shift is not None:
        tv = shift["total_variation_by_layer"]
        top2 = shift["topk_overlap_by_layer"].get("2")
        top4 = shift["topk_overlap_by_layer"].get("4")
        lines.extend(
            [
                f"- Train/eval hotness total variation p50/p90: `{tv['p50']:.4f}` / `{tv['p90']:.4f}`.",
                f"- Train/eval top-2 overlap mean: `{top2['mean']:.4f}`." if top2 else "",
                f"- Train/eval top-4 overlap mean: `{top4['mean']:.4f}`." if top4 else "",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `static_hot_by_capacity` in the JSON is computed only from train trace.",
            "- Eval trace is used here only for train/eval hotness shift diagnostics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line for line in lines if line != "") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--eval-trace", type=Path, default=None)
    parser.add_argument("--window-sizes", default="4,8,16")
    parser.add_argument("--static-hot-capacities", default="1,2,4,25%,50%")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = analyze(
        train_trace=args.train_trace,
        eval_trace=args.eval_trace,
        window_sizes=parse_int_list(args.window_sizes),
        hot_capacities_text=args.static_hot_capacities,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
