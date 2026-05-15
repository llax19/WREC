#!/usr/bin/env python3
"""Summarize real vLLM serving logs for WREC runtime experiments.

The input is a result directory containing files named like:

  <case_id>_request_log.jsonl
  <case_id>_sidecar_metrics.json

The output keeps the main-paper surface aligned with the current WREC runtime:
request/input-token throughput, p95 TTFT, p95 E2E latency, and WREC miss rate
when sidecar metrics are present. TPOT is still reported as a decode sanity
metric, but it is not the primary surface for a prefill-dominant mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def case_id_from_request_log(path: Path) -> str:
    suffix = "_request_log.jsonl"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected request log name: {path}")
    return name[: -len(suffix)]


def summarize_case(request_log: Path) -> dict[str, Any]:
    case_id = case_id_from_request_log(request_log)
    rows = read_jsonl(request_log)
    sidecar_metrics = read_json(request_log.with_name(f"{case_id}_sidecar_metrics.json"))
    gpu_summary_path = request_log.with_name(f"{case_id}_gpu_summary.json")
    gpu_summary = read_json(gpu_summary_path)
    gpu0 = {}
    per_gpu = gpu_summary.get("per_gpu")
    if isinstance(per_gpu, list) and per_gpu:
        gpu0 = per_gpu[0]

    if not rows:
        return {
            "case_id": case_id,
            "status": "empty",
            "num_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests_per_s": None,
            "input_tokens_per_s": None,
            "total_tokens_per_s": None,
            "p95_ttft_ms": None,
            "p95_e2e_ms": None,
            "shadow_miss_rate": sidecar_metrics.get("shadow_miss_rate"),
            "p95_tpot_ms": None,
            "gpu0_peak_memory_used_mib": gpu0.get("peak_memory_used_mib"),
            "gpu0_avg_memory_used_mib": gpu0.get("avg_memory_used_mib"),
            "gpu0_peak_memory_utilization": gpu0.get("peak_memory_utilization"),
        }

    start = min(float(row["submit_time"]) for row in rows)
    end = max(float(row["finish_time"]) for row in rows)
    wall_time = max(end - start, 1e-9)
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    input_tokens = sum(int(row.get("input_tokens", 0)) for row in rows)
    total_tokens = output_tokens + input_tokens

    ttft_s = [float(row["first_token_time"]) - float(row["submit_time"]) for row in rows]
    e2e_s = [float(row["finish_time"]) - float(row["submit_time"]) for row in rows]
    tpot_s = []
    for row in rows:
        out_tokens = int(row.get("output_tokens", 0))
        if out_tokens <= 1:
            continue
        decode_s = float(row["finish_time"]) - float(row["first_token_time"])
        tpot_s.append(decode_s / (out_tokens - 1))

    return {
        "case_id": case_id,
        "status": "success",
        "num_requests": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time,
        "requests_per_s": len(rows) / wall_time,
        "input_tokens_per_s": input_tokens / wall_time,
        "total_tokens_per_s": total_tokens / wall_time,
        "output_tokens_per_s": output_tokens / wall_time,
        "mean_ttft_ms": mean(ttft_s) * 1000.0,
        "p95_ttft_ms": percentile(ttft_s, 0.95) * 1000.0,
        "mean_e2e_ms": mean(e2e_s) * 1000.0,
        "p95_e2e_ms": percentile(e2e_s, 0.95) * 1000.0,
        "mean_tpot_ms": mean(tpot_s) * 1000.0 if tpot_s else None,
        "p95_tpot_ms": (
            percentile(tpot_s, 0.95) * 1000.0 if percentile(tpot_s, 0.95) is not None else None
        ),
        "shadow_miss_rate": sidecar_metrics.get("shadow_miss_rate"),
        "shadow_hits": sidecar_metrics.get("shadow_hits"),
        "shadow_misses": sidecar_metrics.get("shadow_misses"),
        "router_events": sidecar_metrics.get("router_events"),
        "expert_refs": sidecar_metrics.get("expert_refs"),
        "online_loop_us_per_expert_ref": sidecar_metrics.get("timing", {}).get(
            "online_loop_us_per_expert_ref"
        ),
        "gpu0_peak_memory_used_mib": gpu0.get("peak_memory_used_mib"),
        "gpu0_avg_memory_used_mib": gpu0.get("avg_memory_used_mib"),
        "gpu0_peak_memory_utilization": gpu0.get("peak_memory_utilization"),
        "gpu_metrics_summary_json": str(gpu_summary_path),
        "request_log": str(request_log),
        "sidecar_metrics_json": str(request_log.with_name(f"{case_id}_sidecar_metrics.json")),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "case_id",
        "status",
        "num_requests",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "wall_time_s",
        "requests_per_s",
        "input_tokens_per_s",
        "total_tokens_per_s",
        "p95_ttft_ms",
        "p95_e2e_ms",
        "gpu0_peak_memory_used_mib",
        "gpu0_avg_memory_used_mib",
        "gpu0_peak_memory_utilization",
        "shadow_miss_rate",
        "p95_tpot_ms",
        "output_tokens_per_s",
        "mean_tpot_ms",
        "mean_ttft_ms",
        "mean_e2e_ms",
        "router_events",
        "expert_refs",
        "online_loop_us_per_expert_ref",
        "gpu_metrics_summary_json",
        "request_log",
        "sidecar_metrics_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    request_logs = sorted(args.result_root.glob("*_request_log.jsonl"))
    if not request_logs:
        raise ValueError(f"no *_request_log.jsonl files found under {args.result_root}")

    rows = [summarize_case(path) for path in request_logs]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)

    json_output = args.json_output or args.result_root / "serving_metrics_summary.json"
    csv_output = args.csv_output or args.result_root / "serving_metrics_summary.csv"
    json_output.write_text(text + "\n", encoding="utf-8")
    write_csv(rows, csv_output)


if __name__ == "__main__":
    main()
