#!/usr/bin/env python3
"""Build a request-to-expert-signature file for MoE affinity scheduling.

This helper intentionally supports a lightweight deterministic proxy mode. It
does not claim to be a real router trace; it creates stable sparse signatures
so the server-side scheduler path can be validated before wiring a real MoE
router tracer or pre-attention expert predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


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


def maybe_build_tokenizer(tokenizer_path: str | None):
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_expert(text: str, num_experts: int) -> int:
    return stable_int(text) % num_experts


def iter_metadata_terms(row: dict[str, Any]) -> list[tuple[str, str, float]]:
    terms: list[tuple[str, str, float]] = []
    scalar_weights = {
        "topic": 3.0,
        "domain": 2.0,
        "task_family": 2.0,
        "lang": 0.75,
        "group": 0.75,
        "source": 0.5,
    }
    for key, weight in scalar_weights.items():
        value = row.get(key)
        if value:
            terms.append((key, str(value), weight))

    for tag in row.get("tags", []) or []:
        terms.append(("tag", str(tag), 1.0))
    return terms


def add_metadata_signature(
    signature: dict[tuple[int, int], float],
    row: dict[str, Any],
    *,
    num_layers: int,
    num_experts: int,
    experts_per_term: int,
    salt: str,
) -> None:
    for layer in range(num_layers):
        for namespace, value, weight in iter_metadata_terms(row):
            for rank in range(experts_per_term):
                expert = stable_expert(
                    f"{salt}|metadata|{layer}|{namespace}|{value}|{rank}",
                    num_experts,
                )
                key = (layer, expert)
                signature[key] = signature.get(key, 0.0) + weight / (rank + 1)


def add_token_hash_signature(
    signature: dict[tuple[int, int], float],
    row: dict[str, Any],
    *,
    tokenizer,
    num_layers: int,
    num_experts: int,
    max_token_features: int,
    salt: str,
) -> None:
    if tokenizer is None:
        token_ids = [ord(ch) for ch in str(row["prompt"])]
    else:
        token_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]

    if not token_ids:
        return

    stride = max(1, math.ceil(len(token_ids) / max_token_features))
    sampled_token_ids = token_ids[::stride][:max_token_features]
    for token_index, token_id in enumerate(sampled_token_ids):
        token_weight = 0.2
        for layer in range(num_layers):
            expert = stable_expert(
                f"{salt}|token|{layer}|{token_index}|{int(token_id)}",
                num_experts,
            )
            key = (layer, expert)
            signature[key] = signature.get(key, 0.0) + token_weight


def build_signature(
    row: dict[str, Any],
    *,
    mode: str,
    tokenizer,
    num_layers: int,
    num_experts: int,
    experts_per_term: int,
    max_token_features: int,
    salt: str,
) -> dict[tuple[int, int], float]:
    signature: dict[tuple[int, int], float] = {}
    if mode in ("metadata_hash", "hybrid_hash"):
        add_metadata_signature(
            signature,
            row,
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_term=experts_per_term,
            salt=salt,
        )
    if mode in ("token_hash", "hybrid_hash"):
        add_token_hash_signature(
            signature,
            row,
            tokenizer=tokenizer,
            num_layers=num_layers,
            num_experts=num_experts,
            max_token_features=max_token_features,
            salt=salt,
        )
    return signature


def prune_top_per_layer(
    signature: dict[tuple[int, int], float], top_per_layer: int
) -> dict[tuple[int, int], float]:
    if top_per_layer <= 0:
        return signature

    by_layer: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for key, weight in signature.items():
        layer, _ = key
        by_layer.setdefault(layer, []).append((key, weight))

    pruned: dict[tuple[int, int], float] = {}
    for items in by_layer.values():
        items.sort(key=lambda item: (-item[1], item[0][1]))
        for key, weight in items[:top_per_layer]:
            pruned[key] = weight
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("metadata_hash", "token_hash", "hybrid_hash"),
        default="hybrid_hash",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-experts", type=int, default=60)
    parser.add_argument("--experts-per-term", type=int, default=2)
    parser.add_argument("--max-token-features", type=int, default=32)
    parser.add_argument(
        "--prune-top-per-layer",
        type=int,
        default=8,
        help="Keep only the highest-weight experts per layer. Use 0 to disable.",
    )
    parser.add_argument("--salt", default="qwen1.5-moe-a2.7b-proxy-v1")
    args = parser.parse_args()

    rows = load_requests(args.request_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = maybe_build_tokenizer(args.tokenizer_path)
    output_rows = []
    for row in rows:
        signature = build_signature(
            row,
            mode=args.mode,
            tokenizer=tokenizer,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            experts_per_term=args.experts_per_term,
            max_token_features=args.max_token_features,
            salt=args.salt,
        )
        signature = prune_top_per_layer(signature, args.prune_top_per_layer)
        experts = [
            [layer, expert, round(weight, 6)]
            for (layer, expert), weight in sorted(signature.items())
            if weight > 0
        ]
        output_rows.append(
            {
                "request_id": row["id"],
                "signature_source": args.mode,
                "num_layers": args.num_layers,
                "num_experts": args.num_experts,
                "experts": experts,
                "group": row.get("group"),
                "domain": row.get("domain"),
                "task_family": row.get("task_family"),
                "topic": row.get("topic"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(output_rows)} MoE affinity signatures to {args.output}")


if __name__ == "__main__":
    main()
