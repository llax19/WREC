#!/usr/bin/env python3
"""Validate HTTP sidecar smoke metrics against local shadow runtime metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def metric_delta(name: str, sidecar: float, shadow: float, tolerance: float) -> dict[str, Any]:
    delta = sidecar - shadow
    abs_delta = abs(delta)
    return {
        "name": name,
        "sidecar": sidecar,
        "shadow": shadow,
        "delta": delta,
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "pass": abs_delta <= tolerance,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# WREC Runtime Sidecar Alignment Validation",
        "",
        "## Inputs",
        "",
        f"- Sidecar smoke: `{payload['inputs']['sidecar_smoke']}`",
        f"- Local shadow: `{payload['inputs']['local_shadow']}`",
        "",
        "## Verdict",
        "",
        f"- Overall pass: `{payload['overall_pass']}`",
        "",
        "## Metrics",
        "",
        "| metric | sidecar | local shadow | delta | abs delta | tolerance | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["metrics"]:
        lines.append(
            f"| {item['name']} | {item['sidecar']:.12g} | {item['shadow']:.12g} | "
            f"{item['delta']:.12g} | {item['abs_delta']:.12g} | {item['tolerance']:.12g} | {item['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Passing this validation means the HTTP sidecar preserves local shadow behavior for the same event prefix.",
            "- This validates the external runtime integration boundary, not vLLM internal expert loading control.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-json", type=Path, required=True)
    parser.add_argument("--shadow-json", type=Path, required=True)
    parser.add_argument("--rate-tolerance", type=float, default=1e-12)
    parser.add_argument("--count-tolerance", type=float, default=0.0)
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    sidecar = json.loads(args.sidecar_json.read_text(encoding="utf-8"))["sidecar_metrics"]
    shadow = json.loads(args.shadow_json.read_text(encoding="utf-8"))
    shadow_counts = shadow["counts"]
    shadow_transfer = shadow["transfer"]

    metrics = [
        metric_delta("expert_refs", float(sidecar["expert_refs"]), float(shadow_counts["expert_refs"]), args.count_tolerance),
        metric_delta("router_events", float(sidecar["router_events"]), float(shadow_counts["router_events"]), args.count_tolerance),
        metric_delta("shadow_hits", float(sidecar["shadow_hits"]), float(shadow_counts["shadow_hits"]), args.count_tolerance),
        metric_delta("shadow_misses", float(sidecar["shadow_misses"]), float(shadow_counts["shadow_misses"]), args.count_tolerance),
        metric_delta("shadow_hit_rate", float(sidecar["shadow_hit_rate"]), float(shadow_counts["shadow_hit_rate"]), args.rate_tolerance),
        metric_delta("shadow_miss_rate", float(sidecar["shadow_miss_rate"]), float(shadow_counts["shadow_miss_rate"]), args.rate_tolerance),
        metric_delta("would_admit", float(sidecar["would_admit"]), float(shadow_counts["would_admit"]), args.count_tolerance),
        metric_delta("would_bypass", float(sidecar["would_bypass"]), float(shadow_counts["would_bypass"]), args.count_tolerance),
        metric_delta("would_evict", float(sidecar["would_evict"]), float(shadow_counts["would_evict"]), args.count_tolerance),
        metric_delta("demand_transfer_bytes", float(sidecar["demand_transfer_bytes"]), float(shadow_transfer["demand_transfer_bytes"]), args.float_tolerance),
        metric_delta("stall_ms", float(sidecar["stall_ms"]), float(shadow_transfer["stall_ms"]), args.float_tolerance),
    ]
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "sidecar_smoke": str(args.sidecar_json),
            "local_shadow": str(args.shadow_json),
        },
        "overall_pass": all(item["pass"] for item in metrics),
        "metrics": metrics,
        "claim_boundary": {
            "validates_http_sidecar_alignment": all(item["pass"] for item in metrics),
            "does_control_real_expert_loading": False,
            "does_measure_end_to_end_latency": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)
    if not payload["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
