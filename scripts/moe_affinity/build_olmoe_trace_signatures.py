#!/usr/bin/env python3
"""Build sparse request expert signatures from the public OLMoE trace.

The source dataset ``allenai/analysis_olmoe`` exposes token-level expert ids in
the ``exp_ids`` field. This script uses the Hugging Face datasets-server rows
API so it can sample a modest number of rows without downloading the full
dataset.

The output schema matches the sidecar signature format used by the local
``moe_affinity`` scheduler prototype:

    {"request_id": "...", "experts": [[layer, expert, weight], ...], ...}

These signatures are oracle route signatures from another model family. They
are useful for offline locality replay, not for final Qwen/vLLM latency claims.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"


def fetch_rows_page(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout_s: float,
    retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DATASETS_SERVER_ROWS}?{query}"
    headers = {"User-Agent": "codex-moe-affinity-trace-builder/1.0"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout_s) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - network-dependent path.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(retry_sleep_s * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def choose_token_indices(num_tokens: int, max_tokens: int, sample: str) -> list[int]:
    if num_tokens <= 0:
        return []
    if max_tokens <= 0 or max_tokens >= num_tokens:
        return list(range(num_tokens))
    if sample == "first":
        return list(range(max_tokens))
    if max_tokens == 1:
        return [0]
    step = (num_tokens - 1) / (max_tokens - 1)
    return sorted({int(round(i * step)) for i in range(max_tokens)})


def rank_weight(rank: int, max_ranks: int, mode: str) -> float:
    if mode == "flat":
        return 1.0
    if mode == "linear":
        return max(0.0, float(max_ranks - rank) / float(max_ranks))
    if mode == "reciprocal":
        return 1.0 / float(rank + 1)
    raise ValueError(f"unknown rank weight mode: {mode}")


def infer_route_layout(token_route: Any, expected_layers: int) -> str:
    """Infer whether a token route is rank->layer or layer->rank."""
    if not isinstance(token_route, list) or not token_route:
        raise ValueError("exp_ids token entry must be a non-empty list")
    if not isinstance(token_route[0], list) or not token_route[0]:
        raise ValueError("exp_ids token entry must be a nested list")

    outer = len(token_route)
    inner = len(token_route[0])
    if inner == expected_layers:
        return "rank_layer"
    if outer == expected_layers:
        return "layer_rank"
    if inner > outer:
        return "rank_layer"
    return "layer_rank"


def add_token_route_to_counts(
    counts: dict[tuple[int, int], float],
    token_route: list[list[int]],
    *,
    layout: str,
    num_layers: int,
    max_ranks: int,
    rank_weight_mode: str,
) -> None:
    if layout == "rank_layer":
        for rank, by_layer in enumerate(token_route[:max_ranks]):
            weight = rank_weight(rank, max_ranks, rank_weight_mode)
            for layer, expert in enumerate(by_layer[:num_layers]):
                counts[(layer, int(expert))] = counts.get((layer, int(expert)), 0.0) + weight
        return

    if layout == "layer_rank":
        for layer, by_rank in enumerate(token_route[:num_layers]):
            for rank, expert in enumerate(by_rank[:max_ranks]):
                weight = rank_weight(rank, max_ranks, rank_weight_mode)
                counts[(layer, int(expert))] = counts.get((layer, int(expert)), 0.0) + weight
        return

    raise ValueError(f"unknown exp_ids layout: {layout}")


def prune_top_per_layer(
    counts: dict[tuple[int, int], float],
    top_per_layer: int,
) -> dict[tuple[int, int], float]:
    if top_per_layer <= 0:
        return counts

    by_layer: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for key, value in counts.items():
        by_layer.setdefault(key[0], []).append((key, value))

    pruned: dict[tuple[int, int], float] = {}
    for items in by_layer.values():
        items.sort(key=lambda item: (-item[1], item[0][1]))
        for key, value in items[:top_per_layer]:
            pruned[key] = value
    return pruned


def build_signature_from_exp_ids(
    exp_ids: list[Any],
    *,
    num_layers: int,
    max_ranks: int,
    max_tokens: int,
    token_sample: str,
    rank_weight_mode: str,
    top_per_layer: int,
    normalize: bool,
) -> tuple[list[list[float]], str, int]:
    token_indices = choose_token_indices(len(exp_ids), max_tokens, token_sample)
    if not token_indices:
        return [], "unknown", 0

    layout = infer_route_layout(exp_ids[token_indices[0]], num_layers)
    counts: dict[tuple[int, int], float] = {}
    for token_index in token_indices:
        add_token_route_to_counts(
            counts,
            exp_ids[token_index],
            layout=layout,
            num_layers=num_layers,
            max_ranks=max_ranks,
            rank_weight_mode=rank_weight_mode,
        )

    if normalize:
        scale = float(len(token_indices))
        counts = {key: value / scale for key, value in counts.items()}

    counts = prune_top_per_layer(counts, top_per_layer)
    experts = [
        [layer, expert, round(weight, 8)]
        for (layer, expert), weight in sorted(counts.items())
        if weight > 0 and math.isfinite(weight)
    ]
    return experts, layout, len(token_indices)


def build_output_row(
    row: dict[str, Any],
    *,
    source_offset: int,
    dataset: str,
    split: str,
    num_layers: int,
    num_experts: int,
    max_ranks: int,
    max_tokens: int,
    token_sample: str,
    rank_weight_mode: str,
    top_per_layer: int,
    normalize: bool,
    request_id_prefix: str,
) -> dict[str, Any]:
    exp_ids = row.get("exp_ids")
    if not isinstance(exp_ids, list):
        raise ValueError(f"row offset {source_offset} missing list exp_ids")

    experts, layout, signature_tokens = build_signature_from_exp_ids(
        exp_ids,
        num_layers=num_layers,
        max_ranks=max_ranks,
        max_tokens=max_tokens,
        token_sample=token_sample,
        rank_weight_mode=rank_weight_mode,
        top_per_layer=top_per_layer,
        normalize=normalize,
    )
    input_ids = row.get("input_ids") or []
    predicted_token_ids = row.get("predicted_token_ids") or []
    return {
        "request_id": f"{request_id_prefix}-{source_offset:06d}",
        "source_dataset": dataset,
        "source_split": split,
        "source_offset": source_offset,
        "signature_source": "olmoe_exp_ids_oracle",
        "route_layout": layout,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_ranks_used": max_ranks,
        "route_tokens": len(exp_ids),
        "signature_tokens": signature_tokens,
        "input_tokens": len(input_ids),
        "predicted_tokens": len(predicted_token_ids),
        "estimated_total_tokens": len(input_ids) + len(predicted_token_ids),
        "experts": experts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="allenai/analysis_olmoe")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=3.0)
    parser.add_argument("--request-id-prefix", default="olmoe-train")
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--max-ranks", type=int, default=8)
    parser.add_argument(
        "--max-tokens-per-request",
        type=int,
        default=512,
        help="Use 0 to consume all route tokens in each row.",
    )
    parser.add_argument("--token-sample", choices=("first", "uniform"), default="uniform")
    parser.add_argument(
        "--rank-weight",
        choices=("flat", "linear", "reciprocal"),
        default="reciprocal",
    )
    parser.add_argument(
        "--top-per-layer",
        type=int,
        default=16,
        help="Keep the highest-weight experts per layer. Use 0 to keep all experts.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_total = None
    with args.output.open("w", encoding="utf-8") as output:
        while rows_written < args.limit:
            page_offset = args.offset + rows_written
            page_length = min(args.page_size, args.limit - rows_written)
            page = fetch_rows_page(
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=page_offset,
                length=page_length,
                timeout_s=args.timeout_s,
                retries=args.retries,
                retry_sleep_s=args.retry_sleep_s,
            )
            rows_total = page.get("num_rows_total", rows_total)
            rows = page.get("rows") or []
            if not rows:
                break

            for item in rows:
                source_offset = int(item.get("row_idx", page_offset))
                out = build_output_row(
                    item["row"],
                    source_offset=source_offset,
                    dataset=args.dataset,
                    split=args.split,
                    num_layers=args.num_layers,
                    num_experts=args.num_experts,
                    max_ranks=args.max_ranks,
                    max_tokens=args.max_tokens_per_request,
                    token_sample=args.token_sample,
                    rank_weight_mode=args.rank_weight,
                    top_per_layer=args.top_per_layer,
                    normalize=not args.no_normalize,
                    request_id_prefix=args.request_id_prefix,
                )
                output.write(json.dumps(out, ensure_ascii=False) + "\n")
                rows_written += 1
                if rows_written >= args.limit:
                    break

            print(
                f"fetched {rows_written}/{args.limit} rows"
                + (f" (dataset rows: {rows_total})" if rows_total is not None else ""),
                flush=True,
            )
            if len(rows) < page_length:
                break

    print(f"wrote {rows_written} OLMoE trace signatures to {args.output}")


if __name__ == "__main__":
    main()
