#!/usr/bin/env python3
"""Build a short Dolly eval subset for Qwen finite-slot runtime sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = {
                "request_id",
                "prompt",
                "target_max_new_tokens",
                "approx_input_tokens",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/WREC/data/prompts/wrec_dolly_eval_n256.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=120)
    parser.add_argument("--target-max-new-tokens", type=int, default=1)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")
    if args.target_max_new_tokens <= 0:
        raise ValueError("--target-max-new-tokens must be positive")

    source_rows = load_rows(args.source)
    selected: list[dict] = []
    for row in source_rows:
        if int(row["approx_input_tokens"]) > args.max_input_tokens:
            continue
        updated = dict(row)
        updated["target_max_new_tokens"] = int(args.target_max_new_tokens)
        selected.append(updated)
        if len(selected) >= args.limit:
            break

    if len(selected) < args.limit:
        raise RuntimeError(
            f"Only found {len(selected)} rows with approx_input_tokens <= "
            f"{args.max_input_tokens}, need {args.limit}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "rows": len(selected),
                "max_input_tokens": args.max_input_tokens,
                "target_max_new_tokens": args.target_max_new_tokens,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
