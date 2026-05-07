#!/usr/bin/env python3
"""对请求清单 JSONL 做快速结构检查和基础统计。

这个脚本用于在正式发压前检查 `data/prompts/*.jsonl` 是否满足实验脚本
约定的字段格式。它不会修改原文件，只负责验证每行是否包含请求 id、
prompt 和目标输出长度，并顺手打印长度范围和分组分布，帮助判断样本集
是否明显异常。

典型用途：
1. 新生成或手工修改 prompt 清单后先做一次 sanity check。
2. 确认各负载组别是否存在数量失衡或空组。
3. 避免实验跑到一半才发现输入文件字段缺失。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path, help="JSONL file with request prompts")
    args = parser.parse_args()

    rows = []
    with args.request_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row or "target_max_new_tokens" not in row:
                raise ValueError(
                    f"{args.request_path}:{line_no} must contain id, prompt, target_max_new_tokens"
                )
            rows.append(row)

    char_lengths = [len(row["prompt"]) for row in rows]
    print(f"num_requests: {len(rows)}")
    print(f"min_prompt_chars: {min(char_lengths) if char_lengths else 0}")
    print(f"max_prompt_chars: {max(char_lengths) if char_lengths else 0}")
    print(f"avg_prompt_chars: {sum(char_lengths) / len(char_lengths) if char_lengths else 0:.2f}")

    groups: dict[str, int] = {}
    for row in rows:
        group = row.get("group", "unknown")
        groups[group] = groups.get(group, 0) + 1

    print("group_counts:")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")


if __name__ == "__main__":
    main()
