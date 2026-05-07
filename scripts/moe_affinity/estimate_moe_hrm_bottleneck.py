#!/usr/bin/env python3
"""Pre-HRM capacity/transfer screen for MoE expert offload.

Important: this is not a full implementation of MoE-Lightning's HRM.
It is a Phase 0 screening tool that uses transparent roofline-style
capacity and transfer assumptions to decide whether a model/configuration is
worth testing with expert cache/offload experiments.

MoE-Lightning's HRM searches policies over batch size, micro-batch size,
CPU/GPU placement for attention and FFN, GPU-resident weight/KV ratios, and
memory constraints. This script intentionally does not do that policy search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_profile(config: dict[str, Any]) -> dict[str, Any]:
    model_type = config.get("model_type", "unknown")
    hidden_size = int(config.get("hidden_size", config.get("d_model")))
    num_layers = int(config.get("num_hidden_layers", config.get("n_layers")))
    vocab_size = int(config.get("vocab_size", 0))
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    ffn_config = config.get("ffn_config", {})
    attn_config = config.get("attn_config", {})

    if "num_experts" in config:
        num_experts = int(config["num_experts"])
    elif "n_routed_experts" in config:
        num_experts = int(config["n_routed_experts"])
    elif "moe_num_experts" in ffn_config:
        num_experts = int(ffn_config["moe_num_experts"])
    else:
        num_experts = int(config.get("num_local_experts", 0))
    if not num_experts:
        raise ValueError("Could not infer num_experts / num_local_experts from config")

    experts_per_token = int(config.get("num_experts_per_tok", ffn_config.get("moe_top_k")))
    if not experts_per_token:
        raise ValueError("Could not infer num_experts_per_tok / moe_top_k from config")

    expert_intermediate_size = int(
        config.get(
            "moe_intermediate_size",
            ffn_config.get("ffn_hidden_size", config.get("intermediate_size")),
        )
    )
    shared_expert_intermediate_size = int(config.get("shared_expert_intermediate_size", 0))
    if not shared_expert_intermediate_size and "n_shared_experts" in config:
        shared_expert_intermediate_size = int(config["n_shared_experts"]) * expert_intermediate_size

    moe_layer_freq = int(config.get("moe_layer_freq", 1))
    first_k_dense_replace = int(config.get("first_k_dense_replace", 0))
    routed_layers = max(0, num_layers - first_k_dense_replace)
    num_moe_layers = first_k_dense_replace + math.ceil(routed_layers / moe_layer_freq) if routed_layers > 0 else first_k_dense_replace
    num_moe_layers = min(num_layers, num_moe_layers)
    num_dense_ffn_layers = max(0, num_layers - num_moe_layers)
    dense_intermediate_size = int(config.get("intermediate_size", expert_intermediate_size))

    num_attention_heads = int(config.get("num_attention_heads", config.get("n_heads", 0)))
    num_key_value_heads = int(config.get("num_key_value_heads", attn_config.get("kv_n_heads", num_attention_heads)))
    if num_attention_heads:
        head_dim = hidden_size // num_attention_heads
        q_params = hidden_size * hidden_size
        k_params = hidden_size * num_key_value_heads * head_dim
        v_params = hidden_size * num_key_value_heads * head_dim
        o_params = hidden_size * hidden_size
        attention_params_per_layer = q_params + k_params + v_params + o_params
    else:
        attention_params_per_layer = 4 * hidden_size * hidden_size

    # SwiGLU-style FFN: gate_proj + up_proj + down_proj, no bias in these models.
    expert_params = 3 * hidden_size * expert_intermediate_size
    dense_ffn_params_per_layer = 3 * hidden_size * dense_intermediate_size
    shared_expert_params_per_layer = 0
    if shared_expert_intermediate_size:
        shared_expert_params_per_layer = 3 * hidden_size * shared_expert_intermediate_size

    router_params_per_layer = hidden_size * num_experts
    norm_params_per_layer = 2 * hidden_size
    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size

    expert_total_params = num_moe_layers * num_experts * expert_params
    dense_non_expert_params = (
        num_layers
        * (
            attention_params_per_layer
            + norm_params_per_layer
        )
        + num_moe_layers * (shared_expert_params_per_layer + router_params_per_layer)
        + num_dense_ffn_layers * dense_ffn_params_per_layer
        + embedding_params
        + lm_head_params
    )

    active_params_per_token = (
        num_layers * (attention_params_per_layer + norm_params_per_layer)
        + num_moe_layers * (shared_expert_params_per_layer + experts_per_token * expert_params + router_params_per_layer)
        + num_dense_ffn_layers * dense_ffn_params_per_layer
        + embedding_params
        + lm_head_params
    )

    return {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_experts": num_experts,
        "num_experts_per_tok": experts_per_token,
        "num_moe_layers": num_moe_layers,
        "num_dense_ffn_layers": num_dense_ffn_layers,
        "expert_intermediate_size": expert_intermediate_size,
        "dense_intermediate_size": dense_intermediate_size,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
        "attention_params_per_layer": attention_params_per_layer,
        "expert_params_per_expert": expert_params,
        "dense_ffn_params_per_layer": dense_ffn_params_per_layer,
        "shared_expert_params_per_layer": shared_expert_params_per_layer,
        "router_params_per_layer": router_params_per_layer,
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "expert_total_params": expert_total_params,
        "dense_non_expert_params": dense_non_expert_params,
        "model_total_params": dense_non_expert_params + expert_total_params,
        "active_params_per_token": active_params_per_token,
    }


def parse_env_gpu_memory_gb(env_inventory: Path | None) -> list[float]:
    if env_inventory is None or not env_inventory.exists():
        return []
    inventory = read_json(env_inventory)
    command = inventory.get("commands", {}).get("nvidia_smi_query", {})
    stdout = command.get("stdout", "")
    per_gpu: list[float] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                per_gpu.append(float(parts[2]) / 1024.0)
            except ValueError:
                continue
    if not per_gpu:
        return []
    memories = sorted({round(value, 3) for value in per_gpu})
    total = round(sum(per_gpu), 3)
    if total not in memories:
        memories.append(total)
    return memories


def estimate_unique_active_experts(num_experts: int, experts_per_token: int, active_tokens: int) -> float:
    # Uniform-access approximation: expected unique experts touched by active
    # decode tokens in one layer.
    miss_probability = (1.0 - experts_per_token / num_experts) ** active_tokens
    return num_experts * (1.0 - miss_probability)


def build_rows(
    *,
    model_name: str,
    profile: dict[str, Any],
    dtype_names: list[str],
    gpu_memory_gb: list[float],
    kv_cache_budget_gb: list[float],
    bandwidth_gbps: list[float],
    resident_fractions: list[float],
    active_tokens: list[int],
    gpu_tflops: float,
    transfer_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layers = int(profile["num_hidden_layers"])
    num_experts = int(profile["num_experts"])
    experts_per_token = int(profile["num_experts_per_tok"])
    expert_params = int(profile["expert_params_per_expert"])
    dense_params = int(profile["dense_non_expert_params"])
    total_params = int(profile["model_total_params"])
    active_params_per_token = int(profile["active_params_per_token"])

    for dtype_name in dtype_names:
        dtype_key = dtype_name.lower()
        if dtype_key not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        bytes_per_param = DTYPE_BYTES[dtype_key]
        expert_bytes = expert_params * bytes_per_param
        dense_bytes = dense_params * bytes_per_param
        total_bytes = total_params * bytes_per_param
        active_flops_per_token = 2.0 * active_params_per_token

        for memory_gb in gpu_memory_gb:
            for kv_gb in kv_cache_budget_gb:
                available_bytes = max(0.0, (memory_gb - kv_gb) * (1024.0**3))
                all_resident = total_bytes <= available_bytes
                cache_bytes_after_dense = max(0.0, available_bytes - dense_bytes)
                max_resident_experts_total = math.floor(cache_bytes_after_dense / expert_bytes)
                max_resident_experts_per_layer = max_resident_experts_total // layers

                for resident_fraction in resident_fractions:
                    requested_resident_per_layer = min(
                        num_experts, max(0, math.floor(num_experts * resident_fraction))
                    )
                    feasible_resident_per_layer = min(
                        requested_resident_per_layer, max_resident_experts_per_layer
                    )
                    effective_resident_fraction = feasible_resident_per_layer / num_experts

                    for tokens in active_tokens:
                        unique_active = estimate_unique_active_experts(
                            num_experts, experts_per_token, tokens
                        )
                        missing_unique = unique_active * (1.0 - effective_resident_fraction)
                        transfer_bytes_per_step = layers * missing_unique * expert_bytes
                        compute_flops_per_step = active_flops_per_token * tokens
                        compute_ms = compute_flops_per_step / (gpu_tflops * 1e12) * 1000.0

                        for bandwidth in bandwidth_gbps:
                            transfer_ms = transfer_bytes_per_step / (bandwidth * 1e9) * 1000.0
                            decode_ms = compute_ms + transfer_ms
                            transfer_ratio = transfer_ms / decode_ms if decode_ms > 0 else 0.0
                            if all_resident:
                                bottleneck = "all-resident-feasible"
                            elif transfer_ratio >= transfer_threshold:
                                bottleneck = "expert-transfer-bound"
                            else:
                                bottleneck = "memory-capacity-bound"

                            rows.append(
                                {
                                    "model": model_name,
                                    "model_type": profile["model_type"],
                                    "dtype": dtype_name,
                                    "gpu_memory_gb": memory_gb,
                                    "kv_cache_budget_gb": kv_gb,
                                    "bandwidth_gbps": bandwidth,
                                    "decode_active_tokens": tokens,
                                    "resident_fraction_requested": resident_fraction,
                                    "resident_expert_budget_per_layer": feasible_resident_per_layer,
                                    "effective_resident_fraction": effective_resident_fraction,
                                    "all_experts_resident_feasible": all_resident,
                                    "dense_non_expert_gb": dense_bytes / (1024.0**3),
                                    "expert_total_gb": (total_bytes - dense_bytes) / (1024.0**3),
                                    "model_total_gb": total_bytes / (1024.0**3),
                                    "expert_bytes_per_expert_mb": expert_bytes / (1024.0**2),
                                    "expected_unique_active_experts_per_layer": unique_active,
                                    "expected_missing_experts_per_layer": missing_unique,
                                    "estimated_transfer_bytes_per_step": transfer_bytes_per_step,
                                    "estimated_transfer_ms_per_step": transfer_ms,
                                    "estimated_compute_ms_per_step": compute_ms,
                                    "expert_transfer_ratio": transfer_ratio,
                                    "bottleneck_type": bottleneck,
                                }
                            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for row in rows:
        key = str(row["bottleneck_type"])
        by_type[key] = by_type.get(key, 0) + 1

    expert_transfer_rows = [
        row for row in rows if row["bottleneck_type"] == "expert-transfer-bound"
    ]
    max_ratio_row = max(rows, key=lambda row: row["expert_transfer_ratio"])
    resident_rows = [row for row in rows if row["all_experts_resident_feasible"]]

    return {
        "profile": profile,
        "num_rows": len(rows),
        "bottleneck_counts": dict(sorted(by_type.items())),
        "expert_transfer_bound_rows": len(expert_transfer_rows),
        "all_resident_feasible_rows": len(resident_rows),
        "max_transfer_ratio_row": max_ratio_row,
        "notes": [
            "This is a config-based pre-HRM capacity/transfer screen, not a full MoE-Lightning HRM implementation.",
            "It does not search policies over N, micro-batch size, CPU/GPU attention, CPU/GPU FFN, rw, or rc.",
            "Compute time uses the configured GPU TFLOPS value.",
            "Transfer uses uniform expert access approximation before route traces exist.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Model directory or config path")
    parser.add_argument("--model-name", default="qwen1.5-moe-a2.7b")
    parser.add_argument("--env-inventory", type=Path, default=Path("results/wrec/env_inventory_20260427.json"))
    parser.add_argument("--output-json", type=Path, default=Path("results/wrec/hrm_bottleneck_matrix_qwen_20260427.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/wrec/hrm_bottleneck_matrix_qwen_20260427.csv"))
    parser.add_argument("--dtype", default="bf16,int8,int4")
    parser.add_argument("--gpu-memory-gb", default=None, help="Comma-separated memory budgets. Defaults to env per-GPU and total GPU memory.")
    parser.add_argument("--kv-cache-budget-gb", default="4,8,16")
    parser.add_argument("--bandwidth-gbps", default="8,16,32")
    parser.add_argument("--resident-fractions", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--active-tokens", default="1,8,32")
    parser.add_argument("--gpu-tflops", type=float, default=40.0, help="Effective compute throughput used for rough compute-time estimate.")
    parser.add_argument("--transfer-threshold", type=float, default=0.30)
    args = parser.parse_args()

    config_path = args.model_path / "config.json" if args.model_path.is_dir() else args.model_path
    config = read_json(config_path)
    profile = infer_model_profile(config)

    gpu_memory = (
        parse_csv_floats(args.gpu_memory_gb)
        if args.gpu_memory_gb
        else parse_env_gpu_memory_gb(args.env_inventory)
    )
    if not gpu_memory:
        gpu_memory = [24.0, 48.0]

    rows = build_rows(
        model_name=args.model_name,
        profile=profile,
        dtype_names=[item.strip() for item in args.dtype.split(",") if item.strip()],
        gpu_memory_gb=gpu_memory,
        kv_cache_budget_gb=parse_csv_floats(args.kv_cache_budget_gb),
        bandwidth_gbps=parse_csv_floats(args.bandwidth_gbps),
        resident_fractions=parse_csv_floats(args.resident_fractions),
        active_tokens=parse_csv_ints(args.active_tokens),
        gpu_tflops=args.gpu_tflops,
        transfer_threshold=args.transfer_threshold,
    )
    payload = {
        "model_name": args.model_name,
        "config_path": str(config_path),
        "assumptions": {
            "dtype": args.dtype,
            "gpu_memory_gb": gpu_memory,
            "kv_cache_budget_gb": parse_csv_floats(args.kv_cache_budget_gb),
            "bandwidth_gbps": parse_csv_floats(args.bandwidth_gbps),
            "resident_fractions": parse_csv_floats(args.resident_fractions),
            "active_tokens": parse_csv_ints(args.active_tokens),
            "gpu_tflops": args.gpu_tflops,
            "transfer_threshold": args.transfer_threshold,
            "access_model": "uniform expected unique experts before route traces are available",
        },
        "summary": summarize(rows, profile),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(args.output_json)
    print(args.output_csv)
    print(json.dumps(payload["summary"]["bottleneck_counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
