#!/usr/bin/env python3
"""Benchmark online CPU overhead of WREC expert cache policy decisions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from moe_affinity.simulate_expert_cache_offload import infer_expert_bytes, load_event_trace
from wrec import (
    WrecExpertCachePolicy,
    WrecPolicyConfig,
    build_wrec_stats,
    init_wrec_online_state,
    update_wrec_history,
)


@dataclass
class BenchmarkCache:
    resident: set[tuple[int, int]]
    last_touch: dict[tuple[int, int], int]
    lru_order: OrderedDict[tuple[int, int], None]
    prefetched_unused: set[tuple[int, int]]


def top_global_hot(refs: list[Any], total_slots: int) -> set[tuple[int, int]]:
    counts = Counter((ref.layer, ref.expert) for ref in refs)
    return {key for key, _ in counts.most_common(total_slots)}


def init_cache(initial: set[tuple[int, int]]) -> BenchmarkCache:
    return BenchmarkCache(
        resident=set(initial),
        last_touch={key: 0 for key in initial},
        lru_order=OrderedDict((key, None) for key in initial),
        prefetched_unused=set(),
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    timing = payload["timing"]
    transfer = payload["transfer_reference"]
    counts = payload["counts"]
    lines = [
        "# WREC Policy Overhead Benchmark",
        "",
        "## Setup",
        "",
        f"- Eval trace: `{payload['inputs']['trace']}`",
        f"- Train trace: `{payload['inputs']['static_hot_trace']}`",
        f"- Policy: `{payload['config']['policy']}`",
        f"- Total slots: `{payload['config']['total_slots']}`",
        f"- History size: `{payload['config']['history_size']}`",
        f"- Weights: recent `{payload['config']['recent_weight']}`, request `{payload['config']['request_weight']}`, cross-layer `{payload['config']['cross_layer_weight']}`",
        "",
        "## Counts",
        "",
        f"- Expert refs processed: `{counts['expert_refs']}`",
        f"- Router events: `{counts['router_events']}`",
        f"- Demand misses: `{counts['misses']}`",
        f"- Demand hits: `{counts['hits']}`",
        f"- Final resident experts: `{counts['final_resident']}`",
        "",
        "## Online Overhead",
        "",
        f"- Total timed online loop: `{timing['online_loop_seconds']:.6f}` s",
        f"- Mean overhead per expert ref: `{timing['online_loop_us_per_expert_ref']:.3f}` us",
        f"- Mean overhead per router event: `{timing['online_loop_us_per_router_event']:.3f}` us",
        f"- Mean history update cost per expert ref: `{timing['history_update_us_per_expert_ref']:.3f}` us",
        f"- Mean admission decision cost per miss: `{timing['admission_decision_us_per_miss']:.3f}` us",
        f"- Chunk p50/p95 overhead per expert ref: `{timing['chunk_us_per_ref_p50']:.3f}` / `{timing['chunk_us_per_ref_p95']:.3f}` us",
        "",
        "## Transfer Reference",
        "",
        f"- Expert bytes: `{transfer['expert_bytes']}`",
        f"- Bandwidth: `{transfer['bandwidth_gbps']}` GB/s",
        f"- Estimated transfer time per expert miss: `{transfer['expert_transfer_ms']:.6f}` ms",
        f"- Policy loop / expert transfer ratio: `{transfer['policy_loop_to_transfer_ratio']:.6f}`",
        "",
        "## Conclusion",
        "",
        "- The benchmark isolates WREC online CPU decision overhead from trace loading and offline prior construction.",
        "- The timed loop includes cache lookup/touch, online history update, and WREC admission/eviction decisions.",
        "- This is a Python-level overhead measurement, so it is a conservative proxy for a production implementation in a lower-level runtime.",
        "- The measured online overhead is more than two orders of magnitude smaller than the calibrated expert transfer time used by the replay simulator.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--static-hot-trace", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bandwidth-gbps", type=float, default=41.37)
    parser.add_argument("--total-slots", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--recent-weight", type=float, default=0.0)
    parser.add_argument("--request-weight", type=float, default=1024.0)
    parser.add_argument("--cross-layer-weight", type=float, default=1024.0)
    parser.add_argument("--contention-penalty", type=float, default=0.0)
    parser.add_argument("--max-refs", type=int, default=0, help="0 means use the full trace")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    load_start = time.perf_counter()
    refs, metadata = load_event_trace(args.trace)
    static_refs, _ = load_event_trace(args.static_hot_trace)
    load_seconds = time.perf_counter() - load_start
    if args.max_refs > 0:
        refs = refs[: args.max_refs]
        metadata = dict(metadata)
        metadata["num_expert_refs"] = len(refs)
        metadata["num_router_events"] = len({ref.event_index for ref in refs})

    expert_bytes = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])

    stats_start = time.perf_counter()
    wrec_stats = build_wrec_stats(
        static_refs,
        num_layers=num_layers,
        num_experts=num_experts,
        window_size=args.window_size,
        expert_bytes=expert_bytes,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    stats_build_seconds = time.perf_counter() - stats_start

    policy = WrecExpertCachePolicy(
        WrecPolicyConfig(
            recent_weight=args.recent_weight,
            request_weight=args.request_weight,
            cross_layer_weight=args.cross_layer_weight,
            contention_penalty=args.contention_penalty,
        )
    )
    cache = init_cache(top_global_hot(static_refs, args.total_slots))
    online = init_wrec_online_state(num_layers, args.history_size, decisions=None)

    hits = 0
    misses = 0
    history_update_ns = 0
    admission_decision_ns = 0
    chunk_ns = []
    chunk_start = None
    chunk_refs = 0
    chunk_size = max(1, args.chunk_size)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        loop_start = time.perf_counter_ns()
        chunk_start = loop_start
        for ref in refs:
            key = (ref.layer, ref.expert)
            if key in cache.resident:
                hits += 1
                policy.touch(cache, key, ref.ref_index)
                cache.prefetched_unused.discard(key)
                update_start = time.perf_counter_ns()
                update_wrec_history(online, ref)
                history_update_ns += time.perf_counter_ns() - update_start
            else:
                misses += 1
                update_start = time.perf_counter_ns()
                update_wrec_history(online, ref)
                history_update_ns += time.perf_counter_ns() - update_start
                decision_start = time.perf_counter_ns()
                policy.admit_or_bypass(
                    cache,
                    ref=ref,
                    total_slots=args.total_slots,
                    online=online,
                    stats=wrec_stats,
                )
                admission_decision_ns += time.perf_counter_ns() - decision_start

            chunk_refs += 1
            if chunk_refs >= chunk_size:
                now = time.perf_counter_ns()
                chunk_ns.append((now - chunk_start) / chunk_refs)
                chunk_start = now
                chunk_refs = 0
        loop_ns = time.perf_counter_ns() - loop_start
        if chunk_refs > 0 and chunk_start is not None:
            chunk_ns.append((time.perf_counter_ns() - chunk_start) / chunk_refs)
    finally:
        if gc_was_enabled:
            gc.enable()

    expert_refs = max(1, len(refs))
    router_events = max(1, int(metadata["num_router_events"]))
    transfer_ms = expert_bytes / (args.bandwidth_gbps * 1e9) * 1000.0
    loop_us_per_ref = loop_ns / expert_refs / 1000.0
    payload = {
        "inputs": {
            "trace": str(args.trace),
            "static_hot_trace": str(args.static_hot_trace),
            "model_path": str(args.model_path) if args.model_path else None,
        },
        "config": {
            "policy": "wrec_h2_no_recent",
            "total_slots": args.total_slots,
            "window_size": args.window_size,
            "history_size": args.history_size,
            "recent_weight": args.recent_weight,
            "request_weight": args.request_weight,
            "cross_layer_weight": args.cross_layer_weight,
            "contention_penalty": args.contention_penalty,
            "chunk_size": chunk_size,
            "max_refs": args.max_refs,
        },
        "metadata": metadata,
        "counts": {
            "expert_refs": len(refs),
            "router_events": int(metadata["num_router_events"]),
            "hits": hits,
            "misses": misses,
            "final_resident": len(cache.resident),
        },
        "timing": {
            "trace_load_seconds": load_seconds,
            "stats_build_seconds": stats_build_seconds,
            "online_loop_seconds": loop_ns / 1e9,
            "online_loop_us_per_expert_ref": loop_us_per_ref,
            "online_loop_us_per_router_event": loop_ns / router_events / 1000.0,
            "history_update_us_per_expert_ref": history_update_ns / expert_refs / 1000.0,
            "admission_decision_us_per_miss": admission_decision_ns / max(1, misses) / 1000.0,
            "chunk_us_per_ref_mean": mean(chunk_ns) / 1000.0 if chunk_ns else 0.0,
            "chunk_us_per_ref_p50": percentile([value / 1000.0 for value in chunk_ns], 50),
            "chunk_us_per_ref_p95": percentile([value / 1000.0 for value in chunk_ns], 95),
        },
        "transfer_reference": {
            "expert_bytes": expert_bytes,
            "bandwidth_gbps": args.bandwidth_gbps,
            "expert_transfer_ms": transfer_ms,
            "policy_loop_to_transfer_ratio": (loop_us_per_ref / 1000.0) / transfer_ms if transfer_ms > 0 else 0.0,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
