#!/usr/bin/env python3
"""Build sparse expert signatures from Qwen2-MoE router traces.

This script runs an offline prefill forward pass for each prompt, captures the
Qwen2-MoE router probabilities, and exports the same sidecar schema consumed by
the local ``moe_affinity`` scheduler and replay simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "prompt"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing required key: {key}")
            rows.append(row)
    return rows


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    if not text:
        return None
    values: dict[int | str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--max-memory entries must look like 0=22GiB,cpu=48GiB")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        values["cpu" if key == "cpu" else int(key)] = value
    return values


def prune_top_per_layer(
    signature: dict[tuple[int, int], float], top_per_layer: int
) -> dict[tuple[int, int], float]:
    if top_per_layer <= 0:
        return signature

    by_layer: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for key, weight in signature.items():
        by_layer.setdefault(key[0], []).append((key, weight))

    pruned: dict[tuple[int, int], float] = {}
    for items in by_layer.values():
        items.sort(key=lambda item: (-item[1], item[0][1]))
        for key, weight in items[:top_per_layer]:
            pruned[key] = weight
    return pruned


def build_signature_from_router_logits(
    router_logits: tuple[torch.Tensor, ...],
    *,
    num_experts: int,
    top_k: int,
    normalize_by_tokens: bool,
) -> dict[tuple[int, int], float]:
    signature: dict[tuple[int, int], float] = {}
    for layer_idx, layer_logits in enumerate(router_logits):
        probs = layer_logits.detach().float().cpu()
        if probs.ndim == 3:
            probs = probs.reshape(-1, probs.shape[-1])
        if probs.ndim != 2:
            raise ValueError(
                f"unexpected router logits shape at layer {layer_idx}: {tuple(probs.shape)}"
            )
        if probs.shape[-1] != num_experts:
            raise ValueError(
                f"layer {layer_idx} has {probs.shape[-1]} experts, expected {num_experts}"
            )

        layer_top_k = min(top_k, probs.shape[-1])
        values, indices = torch.topk(probs, k=layer_top_k, dim=-1)
        normalizer = float(max(probs.shape[0], 1)) if normalize_by_tokens else 1.0
        for token_values, token_indices in zip(values, indices):
            for value, expert in zip(token_values.tolist(), token_indices.tolist()):
                key = (layer_idx, int(expert))
                signature[key] = signature.get(key, 0.0) + float(value) / normalizer
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("qwen1.5-MoE-A2.7B"))
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--router-top-k", type=int, default=None)
    parser.add_argument("--prune-top-per-layer", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Optional accelerate max_memory, e.g. '0=22GiB,1=22GiB,cpu=48GiB'.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--no-normalize-by-tokens",
        action="store_true",
        help="Store raw summed router probabilities instead of per-token averages.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    requests = load_requests(args.request_file)
    if args.offset:
        requests = requests[args.offset :]
    if args.limit is not None:
        requests = requests[: args.limit]
    if not requests:
        raise ValueError("no requests selected")

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map=args.device_map,
        max_memory=parse_max_memory(args.max_memory),
    )
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_experts = int(model.config.num_experts)
    router_top_k = int(args.router_top_k or model.config.num_experts_per_tok)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(requests, start=1):
            encoded = tokenizer(
                str(row["prompt"]),
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            input_tokens = int(encoded["input_ids"].shape[-1])
            encoded = {key: value.to(model.device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    output_router_logits=True,
                    logits_to_keep=1,
                )
            if outputs.router_logits is None:
                raise RuntimeError("model did not return router_logits")

            signature = build_signature_from_router_logits(
                outputs.router_logits,
                num_experts=num_experts,
                top_k=router_top_k,
                normalize_by_tokens=not args.no_normalize_by_tokens,
            )
            signature = prune_top_per_layer(signature, args.prune_top_per_layer)
            experts = [
                [layer, expert, round(weight, 8)]
                for (layer, expert), weight in sorted(signature.items())
                if weight > 0
            ]
            output_row = {
                "request_id": row["id"],
                "signature_source": "qwen2_moe_router_prefill",
                "num_layers": num_layers,
                "num_experts": num_experts,
                "num_experts_per_tok": int(model.config.num_experts_per_tok),
                "router_top_k": router_top_k,
                "input_tokens": input_tokens,
                "target_tokens": int(row.get("target_max_new_tokens", 0)),
                "target_max_new_tokens": int(row.get("target_max_new_tokens", 0)),
                "estimated_total_tokens": input_tokens + int(row.get("target_max_new_tokens", 0)),
                "experts": experts,
                "group": row.get("group"),
                "domain": row.get("domain"),
                "task_family": row.get("task_family"),
                "topic": row.get("topic"),
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{index}/{len(requests)}] {row['id']} tokens={input_tokens} "
                f"experts={len(experts)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
