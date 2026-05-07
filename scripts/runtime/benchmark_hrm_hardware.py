#!/usr/bin/env python3
"""Measure hardware constants used by the HRM policy evaluator.

Outputs a JSON profile that can replace default HRM constants:

  b_g   -> gpu_bandwidth_gbps
  b_c   -> cpu_bandwidth_gbps
  b_cg  -> cpu_gpu_bandwidth_gbps
  p_g   -> gpu_tflops
  p_c   -> cpu_tflops

GPU measurements require PyTorch with CUDA. CPU measurements require NumPy.
When a dependency is missing, the corresponding section is marked skipped
instead of failing the whole benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def benchmark_cpu_bandwidth(size_mib: int, repeats: int) -> dict[str, Any]:
    if importlib.util.find_spec("numpy") is None:
        return {"available": False, "reason": "numpy is not installed"}
    import numpy as np

    n = (size_mib * 1024 * 1024) // np.dtype(np.float32).itemsize
    src = np.ones(n, dtype=np.float32)
    dst = np.empty_like(src)
    triad_a = np.ones(n, dtype=np.float32)
    triad_b = np.ones(n, dtype=np.float32)
    triad_c = np.empty_like(src)

    # Warmup.
    dst[:] = src
    triad_c[:] = triad_a + 1.5 * triad_b

    copy_gbps: list[float] = []
    triad_gbps: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst[:] = src
        elapsed = time.perf_counter() - t0
        # STREAM copy accounts for one read and one write.
        copy_gbps.append((2.0 * src.nbytes) / elapsed / 1e9)

        t0 = time.perf_counter()
        triad_c[:] = triad_a + 1.5 * triad_b
        elapsed = time.perf_counter() - t0
        # STREAM triad accounts for two reads and one write.
        triad_gbps.append((3.0 * src.nbytes) / elapsed / 1e9)

    return {
        "available": True,
        "size_mib": size_mib,
        "repeats": repeats,
        "copy_gbps": stats(copy_gbps),
        "triad_gbps": stats(triad_gbps),
        "recommended_cpu_bandwidth_gbps": max(statistics.median(copy_gbps), statistics.median(triad_gbps)),
    }


def benchmark_cpu_flops(matrix_size: int, repeats: int) -> dict[str, Any]:
    if importlib.util.find_spec("numpy") is None:
        return {"available": False, "reason": "numpy is not installed"}
    import numpy as np

    rng = np.random.default_rng(20260428)
    a = rng.standard_normal((matrix_size, matrix_size), dtype=np.float32)
    b = rng.standard_normal((matrix_size, matrix_size), dtype=np.float32)
    _ = a @ b

    tflops: list[float] = []
    flops = 2.0 * matrix_size**3
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = a @ b
        elapsed = time.perf_counter() - t0
        tflops.append(flops / elapsed / 1e12)

    return {
        "available": True,
        "matrix_size": matrix_size,
        "repeats": repeats,
        "tflops": stats(tflops),
        "recommended_cpu_tflops": statistics.median(tflops),
    }


def cuda_event_elapsed_ms(torch: Any, start: Any, end: Any) -> float:
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark_gpu_bandwidth(size_mib: int, repeats: int, device: str) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda is not available"}

    dev = torch.device(device)
    n = (size_mib * 1024 * 1024) // torch.empty((), dtype=torch.float32).element_size()
    src = torch.ones(n, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        elapsed_ms = cuda_event_elapsed_ms(torch, start, end)
        times_ms.append(elapsed_ms)

    # Device copy reads and writes HBM.
    gbps = [(2.0 * src.nbytes) / (ms / 1000.0) / 1e9 for ms in times_ms]
    return {
        "available": True,
        "device": device,
        "size_mib": size_mib,
        "repeats": repeats,
        "copy_ms": stats(times_ms),
        "hbm_copy_gbps": stats(gbps),
        "recommended_gpu_bandwidth_gbps": statistics.median(gbps),
    }


def benchmark_cpu_gpu_bandwidth(size_mib: int, repeats: int, device: str) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda is not available"}

    dev = torch.device(device)
    n = (size_mib * 1024 * 1024) // torch.empty((), dtype=torch.float32).element_size()
    dst = torch.empty(n, dtype=torch.float32, device=dev)
    results: dict[str, Any] = {"available": True, "device": device, "size_mib": size_mib, "repeats": repeats}

    for pinned in [False, True]:
        try:
            src = torch.ones(n, dtype=torch.float32, pin_memory=pinned)
        except RuntimeError as exc:
            results["pinned" if pinned else "pageable"] = {
                "available": False,
                "reason": repr(exc),
            }
            continue

        torch.cuda.synchronize()
        times_ms: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dst.copy_(src, non_blocking=pinned)
            end.record()
            elapsed_ms = cuda_event_elapsed_ms(torch, start, end)
            times_ms.append(elapsed_ms)
        gbps = [src.nbytes / (ms / 1000.0) / 1e9 for ms in times_ms]
        results["pinned" if pinned else "pageable"] = {
            "available": True,
            "copy_ms": stats(times_ms),
            "h2d_gbps": stats(gbps),
            "recommended_cpu_gpu_bandwidth_gbps": statistics.median(gbps),
        }

    pinned_result = results.get("pinned", {})
    pageable_result = results.get("pageable", {})
    if pinned_result.get("available"):
        results["recommended_cpu_gpu_bandwidth_gbps"] = pinned_result[
            "recommended_cpu_gpu_bandwidth_gbps"
        ]
    elif pageable_result.get("available"):
        results["recommended_cpu_gpu_bandwidth_gbps"] = pageable_result[
            "recommended_cpu_gpu_bandwidth_gbps"
        ]
    return results


def benchmark_gpu_flops(matrix_size: int, repeats: int, device: str, dtype_name: str) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda is not available"}

    dtype_map = {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported GPU FLOPS dtype: {dtype_name}")

    torch.backends.cuda.matmul.allow_tf32 = dtype_name == "tf32"
    dev = torch.device(device)
    dtype = dtype_map[dtype_name]
    a = torch.randn((matrix_size, matrix_size), device=dev, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=dev, dtype=dtype)
    _ = a @ b
    torch.cuda.synchronize()

    times_ms: list[float] = []
    flops = 2.0 * matrix_size**3
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a @ b
        end.record()
        elapsed_ms = cuda_event_elapsed_ms(torch, start, end)
        times_ms.append(elapsed_ms)

    tflops = [flops / (ms / 1000.0) / 1e12 for ms in times_ms]
    return {
        "available": True,
        "device": device,
        "dtype": dtype_name,
        "matrix_size": matrix_size,
        "repeats": repeats,
        "matmul_ms": stats(times_ms),
        "tflops": stats(tflops),
        "recommended_gpu_tflops": statistics.median(tflops),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/wrec/hrm_hardware_profile_20260428.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu-bytes-mib", type=int, default=512)
    parser.add_argument("--gpu-bytes-mib", type=int, default=512)
    parser.add_argument("--cpu-matmul-size", type=int, default=1536)
    parser.add_argument("--gpu-matmul-size", type=int, default=4096)
    parser.add_argument("--gpu-flops-dtype", default="fp16", choices=["fp32", "tf32", "fp16", "bf16"])
    args = parser.parse_args()

    profile: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "units": {
            "bandwidth": "GB/s decimal",
            "compute": "TFLOP/s",
        },
        "benchmarks": {
            "cpu_bandwidth": benchmark_cpu_bandwidth(args.cpu_bytes_mib, args.repeats),
            "cpu_flops": benchmark_cpu_flops(args.cpu_matmul_size, args.repeats),
            "gpu_bandwidth": benchmark_gpu_bandwidth(args.gpu_bytes_mib, args.repeats, args.device),
            "cpu_gpu_bandwidth": benchmark_cpu_gpu_bandwidth(
                args.gpu_bytes_mib, args.repeats, args.device
            ),
            "gpu_flops": benchmark_gpu_flops(
                args.gpu_matmul_size, args.repeats, args.device, args.gpu_flops_dtype
            ),
        },
    }

    recommended: dict[str, float] = {}
    cpu_bw = profile["benchmarks"]["cpu_bandwidth"]
    if cpu_bw.get("available"):
        recommended["cpu_bandwidth_gbps"] = cpu_bw["recommended_cpu_bandwidth_gbps"]
    cpu_flops = profile["benchmarks"]["cpu_flops"]
    if cpu_flops.get("available"):
        recommended["cpu_tflops"] = cpu_flops["recommended_cpu_tflops"]
    gpu_bw = profile["benchmarks"]["gpu_bandwidth"]
    if gpu_bw.get("available"):
        recommended["gpu_bandwidth_gbps"] = gpu_bw["recommended_gpu_bandwidth_gbps"]
    cpu_gpu_bw = profile["benchmarks"]["cpu_gpu_bandwidth"]
    if cpu_gpu_bw.get("available") and "recommended_cpu_gpu_bandwidth_gbps" in cpu_gpu_bw:
        recommended["cpu_gpu_bandwidth_gbps"] = cpu_gpu_bw[
            "recommended_cpu_gpu_bandwidth_gbps"
        ]
    gpu_flops = profile["benchmarks"]["gpu_flops"]
    if gpu_flops.get("available"):
        recommended["gpu_tflops"] = gpu_flops["recommended_gpu_tflops"]
    profile["recommended_hrm_constants"] = recommended

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(recommended, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
