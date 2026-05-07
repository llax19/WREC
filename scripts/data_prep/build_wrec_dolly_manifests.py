#!/usr/bin/env python3
"""Download Dolly 15k and build deterministic WREC request manifests."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


def download_if_needed(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def approx_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def format_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("context", "")).strip()
    if context:
        return f"Instruction:\n{instruction}\n\nContext:\n{context}\n\nAnswer:"
    return f"Instruction:\n{instruction}\n\nAnswer:"


def target_tokens_for(row: dict[str, Any], prompt_tokens: int) -> tuple[str, int]:
    category = str(row.get("category", "unknown"))
    if category in {"closed_qa", "classification"} or prompt_tokens < 80:
        return "short", 64
    if category in {"summarization", "creative_writing"} or prompt_tokens > 220:
        return "long", 256
    return "medium", 128


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stats_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["approx_input_tokens"]) for row in rows]
    categories = Counter(str(row.get("category", "unknown")) for row in rows)
    groups = Counter(str(row.get("group", "unknown")) for row in rows)
    return {
        "num_requests": len(rows),
        "approx_input_tokens": {
            "min": min(lengths) if lengths else 0,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "mean": statistics.fmean(lengths) if lengths else 0.0,
        },
        "category_counts": dict(sorted(categories.items())),
        "group_counts": dict(sorted(groups.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("data/external/databricks-dolly-15k.jsonl"),
        help="Local path for the downloaded raw Dolly JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--stats-output", type=Path, default=Path("results/wrec/prompt_stats_dolly_20260427.json"))
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--debug-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--max-approx-input-tokens", type=int, default=1024)
    args = parser.parse_args()

    download_if_needed(args.raw_output, DOLLY_URL)
    raw_rows = read_jsonl(args.raw_output)

    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        prompt = format_prompt(raw)
        approx_tokens = approx_token_count(prompt)
        if approx_tokens > args.max_approx_input_tokens:
            continue
        group, target_max_new_tokens = target_tokens_for(raw, approx_tokens)
        category = str(raw.get("category", "unknown"))
        candidates.append(
            {
                "source_index": index,
                "prompt": prompt,
                "source": "databricks-dolly-15k",
                "category": category,
                "group": group,
                "target_max_new_tokens": target_max_new_tokens,
                "approx_input_tokens": approx_tokens,
            }
        )

    required = args.debug_size + args.train_size + args.eval_size
    if len(candidates) < required:
        raise ValueError(f"Need {required} usable Dolly rows, found {len(candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    splits = {
        "debug": candidates[: args.debug_size],
        "train": candidates[args.debug_size : args.debug_size + args.train_size],
        "eval": candidates[
            args.debug_size + args.train_size : args.debug_size + args.train_size + args.eval_size
        ],
    }

    manifest_paths: dict[str, str] = {}
    for split, rows in splits.items():
        out_rows: list[dict[str, Any]] = []
        for split_index, row in enumerate(rows):
            request_id = f"dolly-{split}-{split_index:06d}"
            out = dict(row)
            out["id"] = request_id
            out["request_id"] = request_id
            out["split"] = split
            out_rows.append(out)
        path = args.output_dir / f"wrec_dolly_{split}_n{len(out_rows)}.jsonl"
        write_jsonl(path, out_rows)
        manifest_paths[split] = str(path)
        splits[split] = out_rows

    stats = {
        "source": "databricks-dolly-15k",
        "source_url": DOLLY_URL,
        "seed": args.seed,
        "raw_rows": len(raw_rows),
        "usable_rows": len(candidates),
        "max_approx_input_tokens": args.max_approx_input_tokens,
        "tokenizer": "regex_approximation",
        "manifests": manifest_paths,
        "splits": {split: stats_for(rows) for split, rows in splits.items()},
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.stats_output)


if __name__ == "__main__":
    main()
