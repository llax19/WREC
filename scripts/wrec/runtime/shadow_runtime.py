#!/usr/bin/env python3
"""Run WREC as an online shadow expert-cache policy over routed expert events.

The first version consumes a router-event JSONL trace sequentially. This keeps
the same event contract a runtime adapter would provide, without requiring WREC
to control real expert loading yet.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from moe_affinity.simulate_expert_cache_offload import ExpertRef, infer_expert_bytes, load_event_trace
from wrec import (
    WrecExpertCachePolicy,
    WrecPolicyConfig,
    build_wrec_stats,
    init_wrec_online_state,
    update_wrec_history,
)


@dataclass
class ShadowCache:
    resident: set[tuple[int, int]]
    last_touch: dict[tuple[int, int], int]
    lru_order: OrderedDict[tuple[int, int], None]
    prefetched_unused: set[tuple[int, int]]


def top_global_hot(refs: list[ExpertRef], total_slots: int) -> set[tuple[int, int]]:
    counts = Counter((ref.layer, ref.expert) for ref in refs)
    return {key for key, _ in counts.most_common(total_slots)}


def init_cache(initial: set[tuple[int, int]]) -> ShadowCache:
    return ShadowCache(
        resident=set(initial),
        last_touch={key: 0 for key in initial},
        lru_order=OrderedDict((key, None) for key in initial),
        prefetched_unused=set(),
    )


def write_jsonl(handle: Any | None, payload: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    counts = payload["counts"]
    timing = payload["timing"]
    transfer = payload["transfer"]
    lines = [
        "# WREC Runtime Shadow Report",
        "",
        "## Setup",
        "",
        f"- Event stream: `{payload['inputs']['event_stream']}`",
        f"- Train prior trace: `{payload['inputs']['train_trace']}`",
        f"- Model path: `{payload['inputs']['model_path']}`",
        f"- Policy: `{payload['config']['policy']}`",
        f"- Total slots: `{payload['config']['total_slots']}`",
        f"- History size: `{payload['config']['history_size']}`",
        f"- Weights: recent `{payload['config']['recent_weight']}`, request `{payload['config']['request_weight']}`, cross-layer `{payload['config']['cross_layer_weight']}`",
        f"- Bandwidth: `{payload['config']['bandwidth_gbps']}` GB/s",
        "",
        "## Online Shadow Counts",
        "",
        f"- Expert refs processed: `{counts['expert_refs']}`",
        f"- Router events observed: `{counts['router_events']}`",
        f"- Shadow hits: `{counts['shadow_hits']}`",
        f"- Shadow misses: `{counts['shadow_misses']}`",
        f"- Shadow hit rate: `{counts['shadow_hit_rate']:.9f}`",
        f"- Shadow miss rate: `{counts['shadow_miss_rate']:.9f}`",
        f"- Would-admit count: `{counts['would_admit']}`",
        f"- Would-bypass count: `{counts['would_bypass']}`",
        f"- Would-evict count: `{counts['would_evict']}`",
        f"- Final resident experts: `{counts['final_resident']}`",
        "",
        "## Transfer Proxy",
        "",
        f"- Expert bytes: `{transfer['expert_bytes']}`",
        f"- Demand transfer bytes: `{transfer['demand_transfer_bytes']}`",
        f"- Estimated stall: `{transfer['stall_ms']:.6f}` ms",
        f"- Estimated stall per input token: `{transfer['stall_ms_per_input_token']:.6f}` ms",
        "",
        "## Runtime Overhead",
        "",
        f"- Total shadow loop: `{timing['shadow_loop_seconds']:.6f}` s",
        f"- Mean overhead per expert ref: `{timing['shadow_loop_us_per_expert_ref']:.3f}` us",
        f"- Mean overhead per router event: `{timing['shadow_loop_us_per_router_event']:.3f}` us",
        f"- History update cost per expert ref: `{timing['history_update_us_per_expert_ref']:.3f}` us",
        f"- Decision cost per miss: `{timing['decision_us_per_miss']:.3f}` us",
        "",
        "## Conclusion",
        "",
        "- WREC consumed routed expert events sequentially and made each shadow cache decision without future accesses.",
        "- This validates the policy/state interface as runtime-compatible, but it does not prove end-to-end serving latency improvement.",
        "- The current shadow mode does not control real expert loading; it records what WREC would keep, admit, bypass, and evict.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-stream", type=Path, required=True)
    parser.add_argument("--train-trace", type=Path, required=True)
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
    parser.add_argument("--max-refs", type=int, default=0, help="0 means process the full stream")
    parser.add_argument("--decision-log", type=Path, default=None)
    parser.add_argument("--max-decision-records", type=int, default=256)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    load_start = time.perf_counter()
    refs, metadata = load_event_trace(args.event_stream)
    train_refs, _ = load_event_trace(args.train_trace)
    load_seconds = time.perf_counter() - load_start
    if args.max_refs > 0:
        refs = refs[: args.max_refs]
        metadata = dict(metadata)
        metadata["num_expert_refs"] = len(refs)
        metadata["num_router_events"] = len({ref.event_index for ref in refs})

    expert_bytes = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])

    prior_start = time.perf_counter()
    wrec_stats = build_wrec_stats(
        train_refs,
        num_layers=num_layers,
        num_experts=num_experts,
        window_size=args.window_size,
        expert_bytes=expert_bytes,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    prior_seconds = time.perf_counter() - prior_start

    policy = WrecExpertCachePolicy(
        WrecPolicyConfig(
            recent_weight=args.recent_weight,
            request_weight=args.request_weight,
            cross_layer_weight=args.cross_layer_weight,
            contention_penalty=args.contention_penalty,
        )
    )
    total_possible_slots = num_layers * num_experts
    total_slots = max(0, min(args.total_slots, total_possible_slots))
    cache = init_cache(top_global_hot(train_refs, total_slots))
    online = init_wrec_online_state(num_layers, args.history_size, decisions=None)

    hits = 0
    misses = 0
    would_admit = 0
    would_bypass = 0
    would_evict = 0
    history_update_ns = 0
    decision_ns = 0
    decision_records = 0
    seen_router_events: set[int] = set()

    decision_handle = None
    if args.decision_log is not None:
        args.decision_log.parent.mkdir(parents=True, exist_ok=True)
        decision_handle = args.decision_log.open("w", encoding="utf-8")

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        loop_start = time.perf_counter_ns()
        for ref in refs:
            seen_router_events.add(ref.event_index)
            key = (ref.layer, ref.expert)
            if key in cache.resident:
                hits += 1
                policy.touch(cache, key, ref.ref_index)
                cache.prefetched_unused.discard(key)
                update_start = time.perf_counter_ns()
                update_wrec_history(online, ref)
                history_update_ns += time.perf_counter_ns() - update_start
                if decision_records < args.max_decision_records:
                    write_jsonl(
                        decision_handle,
                        {
                            "ref_index": ref.ref_index,
                            "event_index": ref.event_index,
                            "request_id": ref.request_id,
                            "layer": ref.layer,
                            "token_pos": ref.token_pos,
                            "expert": ref.expert,
                            "shadow_hit": True,
                            "would_admit": False,
                            "would_bypass": False,
                            "would_evict": None,
                        },
                    )
                    decision_records += 1
                continue

            misses += 1
            update_start = time.perf_counter_ns()
            update_wrec_history(online, ref)
            history_update_ns += time.perf_counter_ns() - update_start

            before = set(cache.resident)
            decision_start = time.perf_counter_ns()
            policy.admit_or_bypass(
                cache,
                ref=ref,
                total_slots=total_slots,
                online=online,
                stats=wrec_stats,
            )
            decision_ns += time.perf_counter_ns() - decision_start
            after = set(cache.resident)
            admitted = key in after and key not in before
            evicted = sorted(before - after)
            if admitted:
                would_admit += 1
            else:
                would_bypass += 1
            would_evict += len(evicted)
            if decision_records < args.max_decision_records:
                incoming_score = policy.score(
                    layer=ref.layer,
                    token_pos=ref.token_pos,
                    expert=ref.expert,
                    timestamp=ref.ref_index,
                    cache=cache,
                    online=online,
                    stats=wrec_stats,
                )
                write_jsonl(
                    decision_handle,
                    {
                        "ref_index": ref.ref_index,
                        "event_index": ref.event_index,
                        "request_id": ref.request_id,
                        "layer": ref.layer,
                        "token_pos": ref.token_pos,
                        "expert": ref.expert,
                        "shadow_hit": False,
                        "would_admit": admitted,
                        "would_bypass": not admitted,
                        "would_evict": evicted[0] if evicted else None,
                        "incoming_score_after_update": incoming_score,
                    },
                )
                decision_records += 1
        loop_ns = time.perf_counter_ns() - loop_start
    finally:
        if decision_handle is not None:
            decision_handle.close()
        if gc_was_enabled:
            gc.enable()

    expert_refs = max(1, len(refs))
    router_events = max(1, len(seen_router_events))
    input_tokens = max(1, int(metadata.get("num_input_tokens", 1)))
    demand_transfer_bytes = misses * expert_bytes
    stall_ms = demand_transfer_bytes / (args.bandwidth_gbps * 1e9) * 1000.0
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "event_stream": str(args.event_stream),
            "train_trace": str(args.train_trace),
            "model_path": str(args.model_path) if args.model_path else None,
            "decision_log": str(args.decision_log) if args.decision_log else None,
        },
        "config": {
            "policy": "wrec_h2_shadow",
            "streaming_contract": "sequential routed expert refs; no future accesses",
            "total_slots": total_slots,
            "window_size": args.window_size,
            "history_size": args.history_size,
            "recent_weight": args.recent_weight,
            "request_weight": args.request_weight,
            "cross_layer_weight": args.cross_layer_weight,
            "contention_penalty": args.contention_penalty,
            "bandwidth_gbps": args.bandwidth_gbps,
            "dtype": args.dtype,
            "max_refs": args.max_refs,
        },
        "metadata": metadata,
        "counts": {
            "expert_refs": len(refs),
            "router_events": len(seen_router_events),
            "shadow_hits": hits,
            "shadow_misses": misses,
            "shadow_hit_rate": hits / len(refs) if refs else 0.0,
            "shadow_miss_rate": misses / len(refs) if refs else 0.0,
            "would_admit": would_admit,
            "would_bypass": would_bypass,
            "would_evict": would_evict,
            "final_resident": len(cache.resident),
            "decision_records_written": decision_records,
        },
        "transfer": {
            "expert_bytes": expert_bytes,
            "demand_transfer_bytes": demand_transfer_bytes,
            "stall_ms": stall_ms,
            "stall_ms_per_input_token": stall_ms / input_tokens,
        },
        "timing": {
            "trace_load_seconds": load_seconds,
            "prior_build_seconds": prior_seconds,
            "shadow_loop_seconds": loop_ns / 1e9,
            "shadow_loop_us_per_expert_ref": loop_ns / expert_refs / 1000.0,
            "shadow_loop_us_per_router_event": loop_ns / router_events / 1000.0,
            "history_update_us_per_expert_ref": history_update_ns / expert_refs / 1000.0,
            "decision_us_per_miss": decision_ns / max(1, misses) / 1000.0,
        },
        "claim_boundary": {
            "does_control_real_expert_loading": False,
            "does_measure_end_to_end_latency": False,
            "purpose": "validate online WREC policy/state interface against runtime-style routed expert events",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
