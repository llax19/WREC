#!/usr/bin/env python3
"""Benchmark host-to-GPU copy time for one MoE expert worth of bytes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from moe_affinity.simulate_expert_cache_offload import infer_expert_bytes


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "mean": statistics.fmean(values) if values else 0.0,
        "p90": percentile(values, 0.90),
        "max": max(values) if values else 0.0,
    }


def cuda_event_elapsed_ms(torch: Any, start: Any, end: Any) -> float:
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark_copy(
    *,
    torch: Any,
    nbytes: int,
    device: str,
    pinned: bool,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    dev = torch.device(device)
    try:
        src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
    except RuntimeError as exc:
        return {
            "available": False,
            "pinned": pinned,
            "reason": repr(exc),
        }
    dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    src.fill_(1)
    torch.cuda.synchronize()

    for _ in range(warmups):
        dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=pinned)
        end.record()
        times_ms.append(cuda_event_elapsed_ms(torch, start, end))

    gbps = [nbytes / (ms / 1000.0) / 1e9 for ms in times_ms if ms > 0]
    return {
        "available": True,
        "pinned": pinned,
        "nbytes": nbytes,
        "repeats": repeats,
        "warmups": warmups,
        "copy_ms": stats(times_ms),
        "h2d_gbps": stats(gbps),
        "raw_copy_ms": times_ms,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    expert = payload["expert"]
    assumed = payload["assumed_reference"]
    lines = [
        "# WREC Expert Transfer Microbenchmark",
        "",
        "## Setup",
        "",
        f"- Device: `{payload['device']['name']}`",
        f"- CUDA device: `{payload['config']['device']}`",
        f"- Model path: `{payload['inputs']['model_path']}`",
        f"- Dtype for expert-size inference: `{payload['config']['dtype']}`",
        f"- Expert bytes: `{expert['bytes']}`",
        f"- Expert size: `{expert['mib']:.2f} MiB`",
        f"- Repeats / warmups: `{payload['config']['repeats']}` / `{payload['config']['warmups']}`",
        "",
        "## Simulator Reference",
        "",
        f"- Assumed bandwidth: `{assumed['bandwidth_gbps']:.6f} GB/s`",
        f"- Assumed transfer time: `{assumed['transfer_ms']:.6f} ms/expert`",
        "",
        "## Measurements",
        "",
        "| host memory | available | median ms | mean ms | p90 ms | median GB/s | mean GB/s | ratio vs assumed bandwidth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["pageable", "pinned"]:
        result = payload["measurements"][key]
        if not result.get("available"):
            lines.append(f"| {key} | no | - | - | - | - | - | - |")
            continue
        copy_ms = result["copy_ms"]
        h2d = result["h2d_gbps"]
        ratio = h2d["median"] / assumed["bandwidth_gbps"] if assumed["bandwidth_gbps"] > 0 else 0.0
        lines.append(
            f"| {key} | yes | {copy_ms['median']:.6f} | {copy_ms['mean']:.6f} | "
            f"{copy_ms['p90']:.6f} | {h2d['median']:.6f} | {h2d['mean']:.6f} | {ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- This benchmark measures host-to-GPU copy latency for exactly one inferred Mixtral expert worth of bytes.",
            "- The result calibrates the replay simulator's `expert_bytes / bandwidth` transfer-stall proxy.",
            "- It is still a transfer microbenchmark, not an end-to-end serving latency measurement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assumed-bandwidth-gbps", type=float, default=41.37)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("torch is required for CUDA transfer benchmarking")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available")

    expert_bytes_float = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    expert_bytes = int(round(expert_bytes_float))
    if expert_bytes <= 0:
        raise ValueError("expert bytes must be positive")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()

    measurements = {
        "pageable": benchmark_copy(
            torch=torch,
            nbytes=expert_bytes,
            device=args.device,
            pinned=False,
            repeats=args.repeats,
            warmups=args.warmups,
        ),
        "pinned": benchmark_copy(
            torch=torch,
            nbytes=expert_bytes,
            device=args.device,
            pinned=True,
            repeats=args.repeats,
            warmups=args.warmups,
        ),
    }
    assumed_transfer_ms = expert_bytes / (args.assumed_bandwidth_gbps * 1e9) * 1000.0
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_path": str(args.model_path) if args.model_path else None,
            "expert_bytes_override": args.expert_bytes,
        },
        "config": {
            "dtype": args.dtype,
            "device": args.device,
            "assumed_bandwidth_gbps": args.assumed_bandwidth_gbps,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
        "device": {
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
        },
        "expert": {
            "bytes": expert_bytes,
            "mib": expert_bytes / (1024.0 * 1024.0),
        },
        "assumed_reference": {
            "bandwidth_gbps": args.assumed_bandwidth_gbps,
            "transfer_ms": assumed_transfer_ms,
        },
        "measurements": measurements,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
