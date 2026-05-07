#!/usr/bin/env python3
"""Validate that WREC runtime shadow metrics align with replay metrics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def find_replay_row(path: Path, *, policy: str, total_slots: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("policy") == policy and as_int(row, "total_cache_slots") == total_slots:
                return row
    raise ValueError(f"no replay row found for policy={policy!r}, total_slots={total_slots}")


def metric_delta(name: str, shadow: float, replay: float, tolerance: float) -> dict[str, Any]:
    delta = shadow - replay
    abs_delta = abs(delta)
    return {
        "name": name,
        "shadow": shadow,
        "replay": replay,
        "delta": delta,
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "pass": abs_delta <= tolerance,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# WREC Runtime Shadow Alignment Validation",
        "",
        "## Inputs",
        "",
        f"- Shadow result: `{payload['inputs']['shadow_result']}`",
        f"- Replay CSV: `{payload['inputs']['replay_csv']}`",
        f"- Replay policy: `{payload['config']['replay_policy']}`",
        f"- Total slots: `{payload['config']['total_slots']}`",
        "",
        "## Verdict",
        "",
        f"- Overall pass: `{payload['overall_pass']}`",
        "",
        "## Metrics",
        "",
        "| metric | shadow | replay | delta | abs delta | tolerance | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["metrics"]:
        lines.append(
            f"| {item['name']} | {item['shadow']:.12g} | {item['replay']:.12g} | "
            f"{item['delta']:.12g} | {item['abs_delta']:.12g} | {item['tolerance']:.12g} | {item['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The validation compares runtime shadow output against the existing fixed total-budget replay row.",
            "- Passing this validation means the shadow runtime preserves replay policy behavior under the same trace, prior, budget, and weights.",
            "- This does not imply real expert loading control or end-to-end serving latency improvement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-json", type=Path, required=True)
    parser.add_argument("--replay-csv", type=Path, required=True)
    parser.add_argument("--replay-policy", default="wrec_h2")
    parser.add_argument("--total-slots", type=int, required=True)
    parser.add_argument("--rate-tolerance", type=float, default=1e-12)
    parser.add_argument("--count-tolerance", type=float, default=0.0)
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    shadow = json.loads(args.shadow_json.read_text(encoding="utf-8"))
    replay = find_replay_row(args.replay_csv, policy=args.replay_policy, total_slots=args.total_slots)

    metrics = [
        metric_delta(
            "expert_refs",
            float(shadow["counts"]["expert_refs"]),
            float(as_int(replay, "total_expert_refs")),
            args.count_tolerance,
        ),
        metric_delta(
            "demand_hits",
            float(shadow["counts"]["shadow_hits"]),
            float(as_int(replay, "demand_hits")),
            args.count_tolerance,
        ),
        metric_delta(
            "demand_misses",
            float(shadow["counts"]["shadow_misses"]),
            float(as_int(replay, "demand_misses")),
            args.count_tolerance,
        ),
        metric_delta(
            "hit_rate",
            float(shadow["counts"]["shadow_hit_rate"]),
            as_float(replay, "cache_hit_rate"),
            args.rate_tolerance,
        ),
        metric_delta(
            "miss_rate",
            float(shadow["counts"]["shadow_miss_rate"]),
            as_float(replay, "workload_weighted_miss_rate"),
            args.rate_tolerance,
        ),
        metric_delta(
            "demand_transfer_bytes",
            float(shadow["transfer"]["demand_transfer_bytes"]),
            as_float(replay, "demand_transfer_bytes"),
            args.float_tolerance,
        ),
        metric_delta(
            "stall_ms",
            float(shadow["transfer"]["stall_ms"]),
            as_float(replay, "stall_ms"),
            args.float_tolerance,
        ),
        metric_delta(
            "stall_ms_per_input_token",
            float(shadow["transfer"]["stall_ms_per_input_token"]),
            as_float(replay, "stall_ms_per_input_token"),
            args.float_tolerance,
        ),
    ]
    overall_pass = all(item["pass"] for item in metrics)
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "shadow_result": str(args.shadow_json),
            "replay_csv": str(args.replay_csv),
        },
        "config": {
            "replay_policy": args.replay_policy,
            "total_slots": args.total_slots,
            "rate_tolerance": args.rate_tolerance,
            "count_tolerance": args.count_tolerance,
            "float_tolerance": args.float_tolerance,
        },
        "overall_pass": overall_pass,
        "metrics": metrics,
        "claim_boundary": {
            "validates_shadow_replay_alignment": overall_pass,
            "does_control_real_expert_loading": False,
            "does_measure_end_to_end_latency": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
