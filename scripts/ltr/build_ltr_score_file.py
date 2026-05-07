#!/usr/bin/env python3
"""Build an LTR score file for the server-side scheduler.

The vLLM `ltr` policy consumes a sidecar JSON/JSONL score file through the
`VLLM_LTR_SCORE_FILE` environment variable. This helper creates such a file
from an experiment request manifest so that scheduler plumbing can be tested
before a trained predictor is available.

Modes:
- `target_max_new_tokens`: use the manifest decode budget as a proxy ranking
  score. This is useful for validating the LTR scheduler path, but it is not a
  trained LTR predictor.
- `estimated_total_tokens`: use prompt tokens plus decode budget, matching the
  current Length-only proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_requests(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "prompt", "target_max_new_tokens"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing required key: {key}")
            rows.append(row)
    return rows


def maybe_build_tokenizer(tokenizer_path: str | None):
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_input_tokens(prompt: str, tokenizer) -> int:
    if tokenizer is None:
        return len(prompt)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("target_max_new_tokens", "estimated_total_tokens"),
        default="target_max_new_tokens",
    )
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_requests(args.request_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = maybe_build_tokenizer(args.tokenizer_path)
    output_rows = []
    for row in rows:
        target_max_new_tokens = int(row["target_max_new_tokens"])
        input_tokens = count_input_tokens(row["prompt"], tokenizer)
        if args.mode == "estimated_total_tokens":
            score = input_tokens + target_max_new_tokens
        else:
            score = target_max_new_tokens
        output_rows.append(
            {
                "request_id": row["id"],
                "score": float(score),
                "score_source": args.mode,
                "input_tokens": input_tokens,
                "target_max_new_tokens": target_max_new_tokens,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(output_rows)} LTR scores to {args.output}")


if __name__ == "__main__":
    main()
