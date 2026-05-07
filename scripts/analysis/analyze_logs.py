#!/usr/bin/env python3
"""汇总请求级实验日志并输出核心时延指标。

这个脚本面向 `logs/raw/*.jsonl` 这类原始请求日志文件，读取每条请求的
提交时间、首 token 时间、完成时间、输入输出 token 数等字段，然后按
策略和负载等级分组，计算吞吐、TTFT、TPOT 以及 P95/P99 等摘要指标。

典型用途：
1. 在一次实验完成后，把原始请求日志转成可直接写入论文表格的摘要 JSON。
2. 对不同调度策略或负载档位进行统一口径的指标比较。
3. 为后续画图或人工整理结果提供结构稳定的中间结果。

输入：
- 一个符合当前实验日志 schema 的 JSONL 文件。

输出：
- 标准输出打印 JSON 摘要。
- 如果传入 `--output`，则额外写入目标摘要文件。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {
    "request_id",
    "strategy",
    "load_level",
    "submit_time",
    "first_token_time",
    "finish_time",
    "input_tokens",
    "output_tokens",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def group_key(row: dict, group_by: Iterable[str]) -> tuple:
    return tuple(row[field] for field in group_by)


def summarize(rows: list[dict], group_by: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(group_key(row, group_by), []).append(row)

    summaries: list[dict] = []
    for key, bucket in sorted(buckets.items()):
        ttfts = []
        tpots = []
        total_output_tokens = 0
        submit_min = min(row["submit_time"] for row in bucket)
        finish_max = max(row["finish_time"] for row in bucket)

        for row in bucket:
            ttft = row["first_token_time"] - row["submit_time"]
            ttfts.append(ttft)

            decode_time = row["finish_time"] - row["first_token_time"]
            output_tokens = max(int(row["output_tokens"]), 1)
            total_output_tokens += output_tokens
            tpot = decode_time / max(output_tokens, 1)
            tpots.append(tpot)

        wall_time = max(finish_max - submit_min, 1e-9)
        summary = {
            field: value for field, value in zip(group_by, key)
        }
        summary.update(
            {
                "num_requests": len(bucket),
                "throughput_tokens_per_s": total_output_tokens / wall_time,
                "avg_ttft": sum(ttfts) / len(ttfts),
                "p95_ttft": percentile(ttfts, 0.95),
                "p99_ttft": percentile(ttfts, 0.99),
                "avg_tpot": sum(tpots) / len(tpots),
                "p95_tpot": percentile(tpots, 0.95),
                "p99_tpot": percentile(tpots, 0.99),
                "total_output_tokens": total_output_tokens,
                "wall_time": wall_time,
            }
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to a JSONL request log file")
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["strategy", "load_level"],
        help="Fields used to aggregate metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the aggregated JSON summary",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.log_path)
    summaries = summarize(rows, args.group_by)
    text = json.dumps(summaries, ensure_ascii=False, indent=2)
    print(text)

    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
