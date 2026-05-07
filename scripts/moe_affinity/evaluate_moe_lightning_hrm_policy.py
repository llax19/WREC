#!/usr/bin/env python3
"""Evaluate MoE-Lightning-style HRM policies for one MoE model.

This script implements the policy-evaluation structure described in
MoE-Lightning Section 4.2:

    P = (N, mu, A_g, F_g, r_w, r_c)
    T(M, H, W, P) = max(comm_cpu_to_gpu, T_cpu, T_gpu)

It is intended as a formula-level HRM reproducer for planning. The paper does
not provide exact implementation code for every FLOP/byte term, so the script
uses explicit, documented formulas for attention and MoE FFN bytes/FLOPs. The
outputs should be used for bottleneck and policy comparisons, then calibrated
with profiling before making final performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from estimate_moe_hrm_bottleneck import DTYPE_BYTES, infer_model_profile, read_json


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_bools(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"0", "false", "False"}:
            out.append(0)
        elif item in {"1", "true", "True"}:
            out.append(1)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return out


def read_meminfo_total_gb() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kb = float(line.split()[1])
            return kb / 1024.0 / 1024.0
    return None


def read_hardware_profile(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    profile = read_json(path)
    values = profile.get("recommended_hrm_constants", profile)
    out: dict[str, float] = {}
    for key in [
        "gpu_memory_gb",
        "cpu_memory_gb",
        "gpu_bandwidth_gbps",
        "cpu_bandwidth_gbps",
        "cpu_gpu_bandwidth_gbps",
        "gpu_tflops",
        "cpu_tflops",
    ]:
        if key in values and values[key] is not None:
            out[key] = float(values[key])
    return out


def parse_env_gpu_memory_total_gb(env_inventory: Path | None) -> float | None:
    if env_inventory is None or not env_inventory.exists():
        return None
    inventory = read_json(env_inventory)
    command = inventory.get("commands", {}).get("nvidia_smi_query", {})
    stdout = command.get("stdout", "")
    total = 0.0
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                total += float(parts[2]) / 1024.0
            except ValueError:
                continue
    return total or None


def expected_unique_experts(num_experts: int, topk: int, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    miss_probability = (1.0 - topk / num_experts) ** tokens
    return num_experts * (1.0 - miss_probability)


def canonical_policy(
    attention_on_gpu: int,
    ffn_on_gpu: int,
    rw: float,
    rc: float,
) -> tuple[int, int, float, float]:
    # r_c only matters when attention is on GPU; otherwise KV cache stays on CPU.
    if not attention_on_gpu:
        rc = 0.0
    # r_w only matters for GPU FFN. If FFN runs on CPU, weights stay on CPU.
    if not ffn_on_gpu:
        rw = 0.0
    return attention_on_gpu, ffn_on_gpu, rw, rc


def model_shapes(profile: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hidden = int(profile["hidden_size"])
    n_q = int(config.get("num_attention_heads", 0))
    n_kv = int(config.get("num_key_value_heads", n_q))
    if not n_q:
        raise ValueError("num_attention_heads is required for HRM policy evaluation")
    head_dim = hidden // n_q
    return {
        "hidden_size": hidden,
        "num_query_heads": n_q,
        "num_kv_heads": n_kv,
        "head_dim": head_dim,
    }


def attention_terms(
    *,
    batch_size: int,
    avg_context_length: float,
    shapes: dict[str, Any],
    bytes_per_param: float,
    attention_on_gpu: int,
    rc: float,
    gpu_bandwidth_gbps: float,
    cpu_bandwidth_gbps: float,
    gpu_tflops: float,
    cpu_tflops: float,
) -> dict[str, float]:
    hidden = int(shapes["hidden_size"])
    n_q = int(shapes["num_query_heads"])
    n_kv = int(shapes["num_kv_heads"])
    head_dim = int(shapes["head_dim"])

    kv_bytes = batch_size * avg_context_length * 2 * n_kv * head_dim * bytes_per_param
    hidden_bytes = batch_size * hidden * bytes_per_param
    softmax_flops = 5.0 * batch_size * n_q * avg_context_length
    qk_flops = 2.0 * batch_size * n_q * avg_context_length * head_dim
    av_flops = 2.0 * batch_size * n_q * avg_context_length * head_dim
    flops = qk_flops + av_flops + softmax_flops

    if attention_on_gpu:
        cross_bytes = (1.0 - rc) * kv_bytes
        gpu_local_bytes = kv_bytes + 2.0 * hidden_bytes
        t_gpu_ms = max(
            gpu_local_bytes / (gpu_bandwidth_gbps * 1e9),
            flops / (gpu_tflops * 1e12),
        ) * 1000.0
        t_cpu_ms = 0.0
    else:
        # CPU attention in CGOPipe-style execution needs query/state transfer to
        # CPU and attention output transfer back to GPU.
        cross_bytes = 2.0 * hidden_bytes
        cpu_local_bytes = kv_bytes + 2.0 * hidden_bytes
        t_cpu_ms = max(
            cpu_local_bytes / (cpu_bandwidth_gbps * 1e9),
            flops / (cpu_tflops * 1e12),
        ) * 1000.0
        t_gpu_ms = 0.0

    return {
        "attn_flops": flops,
        "attn_kv_bytes": kv_bytes,
        "attn_cross_bytes": cross_bytes,
        "attn_t_gpu_ms": t_gpu_ms,
        "attn_t_cpu_ms": t_cpu_ms,
    }


def ffn_terms(
    *,
    batch_size: int,
    micro_batch_size: int,
    profile: dict[str, Any],
    bytes_per_param: float,
    ffn_on_gpu: int,
    rw: float,
    gpu_bandwidth_gbps: float,
    cpu_bandwidth_gbps: float,
    gpu_tflops: float,
    cpu_tflops: float,
) -> dict[str, float]:
    hidden = int(profile["hidden_size"])
    num_experts = int(profile["num_experts"])
    topk = int(profile["num_experts_per_tok"])
    expert_params = int(profile["expert_params_per_expert"])
    shared_params = int(profile["shared_expert_params_per_layer"])

    expert_bytes = expert_params * bytes_per_param
    shared_bytes = shared_params * bytes_per_param
    ffn_total_layer_bytes = num_experts * expert_bytes + shared_bytes

    active_experts_for_batch = expected_unique_experts(num_experts, topk, batch_size)
    active_experts_for_micro = expected_unique_experts(num_experts, topk, micro_batch_size)
    active_weight_bytes_for_batch = active_experts_for_batch * expert_bytes + shared_bytes
    active_weight_bytes_for_micro = active_experts_for_micro * expert_bytes + shared_bytes
    num_micro_batches = math.ceil(batch_size / micro_batch_size)

    flops = 2.0 * batch_size * (topk * expert_params + shared_params)
    activation_bytes = 2.0 * batch_size * hidden * bytes_per_param

    if ffn_on_gpu:
        resident_budget_bytes = rw * ffn_total_layer_bytes
        cross_bytes = max(0.0, active_weight_bytes_for_batch - resident_budget_bytes)
        gpu_local_bytes = num_micro_batches * active_weight_bytes_for_micro + activation_bytes
        t_gpu_ms = max(
            gpu_local_bytes / (gpu_bandwidth_gbps * 1e9),
            flops / (gpu_tflops * 1e12),
        ) * 1000.0
        t_cpu_ms = 0.0
    else:
        # If FFN runs on CPU, hidden states have to come back for the next layer.
        cross_bytes = activation_bytes
        cpu_local_bytes = num_micro_batches * active_weight_bytes_for_micro + activation_bytes
        t_cpu_ms = max(
            cpu_local_bytes / (cpu_bandwidth_gbps * 1e9),
            flops / (cpu_tflops * 1e12),
        ) * 1000.0
        t_gpu_ms = 0.0

    return {
        "ffn_flops": flops,
        "ffn_total_layer_bytes": ffn_total_layer_bytes,
        "ffn_active_weight_bytes_for_batch": active_weight_bytes_for_batch,
        "ffn_active_weight_bytes_for_micro": active_weight_bytes_for_micro,
        "ffn_cross_bytes": cross_bytes,
        "ffn_t_gpu_ms": t_gpu_ms,
        "ffn_t_cpu_ms": t_cpu_ms,
        "expected_active_experts_batch": active_experts_for_batch,
        "expected_active_experts_micro": active_experts_for_micro,
        "num_micro_batches": num_micro_batches,
    }


def memory_terms(
    *,
    profile: dict[str, Any],
    shapes: dict[str, Any],
    dtype_bytes: float,
    gpu_memory_gb: float,
    cpu_memory_gb: float,
    batch_size: int,
    avg_context_length: float,
    attention_on_gpu: int,
    ffn_on_gpu: int,
    rw: float,
    rc: float,
    weight_buffer_layers: float,
    activation_buffer_factor: float,
) -> dict[str, Any]:
    layers = int(profile["num_hidden_layers"])
    hidden = int(profile["hidden_size"])
    n_kv = int(shapes["num_kv_heads"])
    head_dim = int(shapes["head_dim"])
    total_weight_bytes = float(profile["model_total_params"]) * dtype_bytes
    expert_total_bytes = float(profile["expert_total_params"]) * dtype_bytes
    dense_bytes = float(profile["dense_non_expert_params"]) * dtype_bytes
    shared_total_bytes = float(profile["shared_expert_params_per_layer"]) * layers * dtype_bytes
    ffn_total_bytes = expert_total_bytes + shared_total_bytes

    if ffn_on_gpu:
        gpu_weight_bytes = dense_bytes + rw * ffn_total_bytes
        cpu_weight_bytes = (1.0 - rw) * ffn_total_bytes
        offloaded_ffn_per_layer = (1.0 - rw) * ffn_total_bytes / layers
        weight_buffer_bytes = weight_buffer_layers * offloaded_ffn_per_layer
    else:
        gpu_weight_bytes = dense_bytes
        cpu_weight_bytes = ffn_total_bytes
        weight_buffer_bytes = 0.0

    total_kv_bytes = layers * batch_size * avg_context_length * 2 * n_kv * head_dim * dtype_bytes
    if attention_on_gpu:
        gpu_kv_bytes = rc * total_kv_bytes
        cpu_kv_bytes = (1.0 - rc) * total_kv_bytes
    else:
        gpu_kv_bytes = 0.0
        cpu_kv_bytes = total_kv_bytes

    activation_buffer_bytes = activation_buffer_factor * batch_size * hidden * dtype_bytes
    gpu_total = gpu_weight_bytes + gpu_kv_bytes + weight_buffer_bytes + activation_buffer_bytes
    cpu_total = cpu_weight_bytes + cpu_kv_bytes
    gpu_capacity = gpu_memory_gb * (1024.0**3)
    cpu_capacity = cpu_memory_gb * (1024.0**3)

    return {
        "total_weight_gb": total_weight_bytes / (1024.0**3),
        "dense_weight_gb": dense_bytes / (1024.0**3),
        "ffn_weight_gb": ffn_total_bytes / (1024.0**3),
        "gpu_weight_gb": gpu_weight_bytes / (1024.0**3),
        "cpu_weight_gb": cpu_weight_bytes / (1024.0**3),
        "gpu_kv_gb": gpu_kv_bytes / (1024.0**3),
        "cpu_kv_gb": cpu_kv_bytes / (1024.0**3),
        "weight_buffer_gb": weight_buffer_bytes / (1024.0**3),
        "activation_buffer_gb": activation_buffer_bytes / (1024.0**3),
        "gpu_memory_used_gb": gpu_total / (1024.0**3),
        "cpu_memory_used_gb": cpu_total / (1024.0**3),
        "gpu_memory_feasible": gpu_total <= gpu_capacity,
        "cpu_memory_feasible": cpu_total <= cpu_capacity,
    }


def evaluate_policy(
    *,
    model_name: str,
    profile: dict[str, Any],
    shapes: dict[str, Any],
    dtype_name: str,
    dtype_bytes: float,
    hardware: dict[str, float],
    workload: dict[str, float],
    batch_size: int,
    micro_batch_size: int,
    attention_on_gpu: int,
    ffn_on_gpu: int,
    rw: float,
    rc: float,
    weight_buffer_layers: float,
    activation_buffer_factor: float,
) -> dict[str, Any]:
    avg_context_length = workload["avg_prompt_length"] + workload["generation_length"] / 2.0

    attn = attention_terms(
        batch_size=batch_size,
        avg_context_length=avg_context_length,
        shapes=shapes,
        bytes_per_param=dtype_bytes,
        attention_on_gpu=attention_on_gpu,
        rc=rc,
        gpu_bandwidth_gbps=hardware["gpu_bandwidth_gbps"],
        cpu_bandwidth_gbps=hardware["cpu_bandwidth_gbps"],
        gpu_tflops=hardware["gpu_tflops"],
        cpu_tflops=hardware["cpu_tflops"],
    )
    ffn = ffn_terms(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        profile=profile,
        bytes_per_param=dtype_bytes,
        ffn_on_gpu=ffn_on_gpu,
        rw=rw,
        gpu_bandwidth_gbps=hardware["gpu_bandwidth_gbps"],
        cpu_bandwidth_gbps=hardware["cpu_bandwidth_gbps"],
        gpu_tflops=hardware["gpu_tflops"],
        cpu_tflops=hardware["cpu_tflops"],
    )
    mem = memory_terms(
        profile=profile,
        shapes=shapes,
        dtype_bytes=dtype_bytes,
        gpu_memory_gb=hardware["gpu_memory_gb"],
        cpu_memory_gb=hardware["cpu_memory_gb"],
        batch_size=batch_size,
        avg_context_length=avg_context_length,
        attention_on_gpu=attention_on_gpu,
        ffn_on_gpu=ffn_on_gpu,
        rw=rw,
        rc=rc,
        weight_buffer_layers=weight_buffer_layers,
        activation_buffer_factor=activation_buffer_factor,
    )

    cross_bytes = attn["attn_cross_bytes"] + ffn["ffn_cross_bytes"]
    comm_cpu_to_gpu_ms = cross_bytes / (hardware["cpu_gpu_bandwidth_gbps"] * 1e9) * 1000.0
    t_gpu_ms = attn["attn_t_gpu_ms"] + ffn["ffn_t_gpu_ms"]
    t_cpu_ms = attn["attn_t_cpu_ms"] + ffn["ffn_t_cpu_ms"]
    layer_latency_ms = max(comm_cpu_to_gpu_ms, t_cpu_ms, t_gpu_ms)

    feasible = bool(mem["gpu_memory_feasible"] and mem["cpu_memory_feasible"])
    if not mem["gpu_memory_feasible"]:
        bottleneck = "infeasible-gpu-memory"
    elif not mem["cpu_memory_feasible"]:
        bottleneck = "infeasible-cpu-memory"
    elif layer_latency_ms == comm_cpu_to_gpu_ms:
        bottleneck = "cpu-gpu-transfer"
    elif layer_latency_ms == t_cpu_ms:
        bottleneck = "cpu-compute-or-memory"
    else:
        bottleneck = "gpu-compute-or-memory"

    return {
        "model": model_name,
        "dtype": dtype_name,
        "batch_size_N": batch_size,
        "micro_batch_size_mu": micro_batch_size,
        "A_g_attention_on_gpu": attention_on_gpu,
        "F_g_ffn_on_gpu": ffn_on_gpu,
        "r_w_gpu_weight_ratio": rw,
        "r_c_gpu_kv_ratio": rc,
        "avg_prompt_length": workload["avg_prompt_length"],
        "generation_length": workload["generation_length"],
        "avg_decode_context_length": avg_context_length,
        "gpu_memory_gb": hardware["gpu_memory_gb"],
        "cpu_memory_gb": hardware["cpu_memory_gb"],
        "gpu_bandwidth_gbps": hardware["gpu_bandwidth_gbps"],
        "cpu_bandwidth_gbps": hardware["cpu_bandwidth_gbps"],
        "cpu_gpu_bandwidth_gbps": hardware["cpu_gpu_bandwidth_gbps"],
        "gpu_tflops": hardware["gpu_tflops"],
        "cpu_tflops": hardware["cpu_tflops"],
        "memory_feasible": feasible,
        "bottleneck": bottleneck,
        "layer_latency_ms": layer_latency_ms,
        "throughput_tokens_per_s_proxy": (
            batch_size / (layer_latency_ms / 1000.0) if layer_latency_ms > 0 else 0.0
        ),
        "comm_cpu_to_gpu_ms": comm_cpu_to_gpu_ms,
        "T_cpu_ms": t_cpu_ms,
        "T_gpu_ms": t_gpu_ms,
        "cross_transfer_gb_per_layer": cross_bytes / (1024.0**3),
        **mem,
        "attn_flops": attn["attn_flops"],
        "ffn_flops": ffn["ffn_flops"],
        "attn_t_cpu_ms": attn["attn_t_cpu_ms"],
        "attn_t_gpu_ms": attn["attn_t_gpu_ms"],
        "ffn_t_cpu_ms": ffn["ffn_t_cpu_ms"],
        "ffn_t_gpu_ms": ffn["ffn_t_gpu_ms"],
        "attn_cross_gb": attn["attn_cross_bytes"] / (1024.0**3),
        "ffn_cross_gb": ffn["ffn_cross_bytes"] / (1024.0**3),
        "expected_active_experts_batch": ffn["expected_active_experts_batch"],
        "expected_active_experts_micro": ffn["expected_active_experts_micro"],
        "num_micro_batches": ffn["num_micro_batches"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["memory_feasible"]]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["bottleneck"])
        counts[key] = counts.get(key, 0) + 1

    best_latency = min(feasible, key=lambda row: row["layer_latency_ms"]) if feasible else None
    best_throughput = (
        max(feasible, key=lambda row: row["throughput_tokens_per_s_proxy"]) if feasible else None
    )
    best_by_batch: dict[str, dict[str, Any]] = {}
    for row in feasible:
        key = str(row["batch_size_N"])
        current = best_by_batch.get(key)
        if current is None or row["layer_latency_ms"] < current["layer_latency_ms"]:
            best_by_batch[key] = row

    return {
        "num_rows": len(rows),
        "num_feasible": len(feasible),
        "bottleneck_counts": dict(sorted(counts.items())),
        "best_latency_policy": best_latency,
        "best_throughput_policy": best_throughput,
        "best_policy_by_batch": best_by_batch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen1.5-moe-a2.7b")
    parser.add_argument("--env-inventory", type=Path, default=Path("results/wrec/env_inventory_20260427.json"))
    parser.add_argument("--output-json", type=Path, default=Path("results/wrec/hrm_policy_eval_qwen_20260428.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/wrec/hrm_policy_eval_qwen_20260428.csv"))
    parser.add_argument("--hardware-profile", type=Path, default=None, help="JSON produced by scripts/runtime/benchmark_hrm_hardware.py")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--gpu-memory-gb", type=float, default=None)
    parser.add_argument("--cpu-memory-gb", type=float, default=None)
    parser.add_argument("--gpu-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--cpu-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--cpu-gpu-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--gpu-tflops", type=float, default=None)
    parser.add_argument("--cpu-tflops", type=float, default=None)
    parser.add_argument("--avg-prompt-length", type=float, default=512.0)
    parser.add_argument("--generation-length", type=float, default=128.0)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64")
    parser.add_argument("--micro-batch-sizes", default="1,4,8,16,32,64")
    parser.add_argument("--attention-gpu", default="0,1")
    parser.add_argument("--ffn-gpu", default="0,1")
    parser.add_argument("--rw-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--rc-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--weight-buffer-layers", type=float, default=2.0)
    parser.add_argument("--activation-buffer-factor", type=float, default=8.0)
    args = parser.parse_args()

    config_path = args.model_path / "config.json" if args.model_path.is_dir() else args.model_path
    config = read_json(config_path)
    profile = infer_model_profile(config)
    shapes = model_shapes(profile, config)
    dtype_key = args.dtype.lower()
    if dtype_key not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype_bytes = DTYPE_BYTES[dtype_key]

    hardware_profile = read_hardware_profile(args.hardware_profile)

    gpu_memory = args.gpu_memory_gb or hardware_profile.get("gpu_memory_gb")
    if gpu_memory is None:
        gpu_memory = parse_env_gpu_memory_total_gb(args.env_inventory) or 24.0
    cpu_memory = args.cpu_memory_gb or hardware_profile.get("cpu_memory_gb")
    if cpu_memory is None:
        cpu_memory = read_meminfo_total_gb() or 128.0

    hardware = {
        "gpu_memory_gb": gpu_memory,
        "cpu_memory_gb": cpu_memory,
        "gpu_bandwidth_gbps": args.gpu_bandwidth_gbps
        or hardware_profile.get("gpu_bandwidth_gbps", 900.0),
        "cpu_bandwidth_gbps": args.cpu_bandwidth_gbps
        or hardware_profile.get("cpu_bandwidth_gbps", 100.0),
        "cpu_gpu_bandwidth_gbps": args.cpu_gpu_bandwidth_gbps
        or hardware_profile.get("cpu_gpu_bandwidth_gbps", 16.0),
        "gpu_tflops": args.gpu_tflops or hardware_profile.get("gpu_tflops", 40.0),
        "cpu_tflops": args.cpu_tflops or hardware_profile.get("cpu_tflops", 1.6),
    }
    workload = {
        "avg_prompt_length": args.avg_prompt_length,
        "generation_length": args.generation_length,
    }

    seen: set[tuple[int, int, int, int, float, float]] = set()
    rows: list[dict[str, Any]] = []
    for n in parse_csv_ints(args.batch_sizes):
        for mu in parse_csv_ints(args.micro_batch_sizes):
            if mu > n:
                continue
            for ag in parse_csv_bools(args.attention_gpu):
                for fg in parse_csv_bools(args.ffn_gpu):
                    for rw_input in parse_csv_floats(args.rw_values):
                        for rc_input in parse_csv_floats(args.rc_values):
                            ag_eff, fg_eff, rw, rc = canonical_policy(ag, fg, rw_input, rc_input)
                            key = (n, mu, ag_eff, fg_eff, rw, rc)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append(
                                evaluate_policy(
                                    model_name=args.model_name,
                                    profile=profile,
                                    shapes=shapes,
                                    dtype_name=args.dtype,
                                    dtype_bytes=dtype_bytes,
                                    hardware=hardware,
                                    workload=workload,
                                    batch_size=n,
                                    micro_batch_size=mu,
                                    attention_on_gpu=ag_eff,
                                    ffn_on_gpu=fg_eff,
                                    rw=rw,
                                    rc=rc,
                                    weight_buffer_layers=args.weight_buffer_layers,
                                    activation_buffer_factor=args.activation_buffer_factor,
                                )
                            )

    payload = {
        "model_name": args.model_name,
        "config_path": str(config_path),
        "paper_alignment": {
            "policy_tuple": "(N, mu, A_g, F_g, r_w, r_c)",
            "latency_equation": "T(M,H,W,P)=max(comm_cpu_to_gpu,T_cpu,T_gpu)",
            "source": "MoE-Lightning Section 4.2",
        },
        "assumptions": {
            "dtype": args.dtype,
            "hardware": hardware,
            "hardware_profile": str(args.hardware_profile) if args.hardware_profile else None,
            "workload": workload,
            "weight_buffer_layers": args.weight_buffer_layers,
            "activation_buffer_factor": args.activation_buffer_factor,
            "attention": "decode attention core; QK, AV, softmax; QKVO projections omitted as in Section 3.3 case-study note",
            "ffn": "SwiGLU FFN flops from active routed experts plus shared expert when present",
            "routing": "expected unique active experts under uniform top-k routing before route traces are available",
            "calibration": "kernel/profile calibration is still required before final performance claims",
        },
        "profile": profile,
        "shapes": shapes,
        "summary": summarize(rows),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(args.output_json)
    print(args.output_csv)
    print(json.dumps(payload["summary"]["bottleneck_counts"], ensure_ascii=False, sort_keys=True))
    best = payload["summary"]["best_throughput_policy"]
    if best:
        print(
            "best_throughput",
            {
                "N": best["batch_size_N"],
                "mu": best["micro_batch_size_mu"],
                "A_g": best["A_g_attention_on_gpu"],
                "F_g": best["F_g_ffn_on_gpu"],
                "r_w": best["r_w_gpu_weight_ratio"],
                "r_c": best["r_c_gpu_kv_ratio"],
                "latency_ms": best["layer_latency_ms"],
                "tokens_per_s_proxy": best["throughput_tokens_per_s_proxy"],
                "bottleneck": best["bottleneck"],
            },
        )


if __name__ == "__main__":
    main()
