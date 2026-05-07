#!/usr/bin/env python3
"""聚合多轮实验摘要，输出均值、标准差和范围。

这个脚本面向 `results/*_summary.json` 这类单轮实验摘要文件。它会读取多份
摘要，按 `strategy + load_level` 分组，对常用指标计算均值、标准差、最小值
和最大值，便于把多次重复实验整理成论文可用结果表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


NUMERIC_FIELDS = [
    "num_requests",
    "throughput_tokens_per_s",
    "avg_ttft",
    "p95_ttft",
    "p99_ttft",
    "avg_tpot",
    "p95_tpot",
    "p99_tpot",
    "total_output_tokens",
    "wall_time",
]


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if any(char in item for char in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        else:
            paths.append(Path(item))
    deduped = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def read_summary_file(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    for row in rows:
        if "strategy" not in row or "load_level" not in row:
            raise ValueError(f"{path} is missing strategy/load_level")
    return rows


def summarize_numeric(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(paths: list[Path]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    source_files: dict[tuple[str, str], list[str]] = {}

    for path in paths:
        for row in read_summary_file(path):
            key = (str(row["strategy"]), str(row["load_level"]))
            buckets.setdefault(key, []).append(row)
            source_files.setdefault(key, []).append(str(path))

    aggregated = []
    for (strategy, load_level), rows in sorted(buckets.items()):
        metrics = {}
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in rows if field in row]
            if values:
                metrics[field] = summarize_numeric(values)
        aggregated.append(
            {
                "strategy": strategy,
                "load_level": load_level,
                "num_runs": len(rows),
                "source_files": source_files[(strategy, load_level)],
                "metrics": metrics,
            }
        )
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files or glob patterns such as 'results/tp2_fcfs_*_summary.json'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path used to save the aggregated JSON result",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise ValueError("no input summary files matched")

    result = aggregate(paths)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
