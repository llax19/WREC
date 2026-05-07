#!/usr/bin/env python3
"""周期采样 GPU 显存和利用率，并保存为 JSONL。

这个脚本通常作为后台辅助进程运行，配合单卡或双卡实验脚本一起启动。
它通过 `nvidia-smi` 定期抓取每块 GPU 的显存占用、总显存、计算利用率和
显存带宽利用率，并把每次采样追加到 JSONL 文件中，供实验结束后做峰值
显存统计和资源曲线分析。

典型用途：
1. 给 `run_stage1_single_gpu.sh` / `run_stage1_dual_gpu.sh` 提供原始 GPU 监控数据。
2. 对不同调度参数下的显存压力做横向比较。
3. 在服务异常退出时保留一段可追溯的资源使用记录。
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path


RUNNING = True


def handle_signal(signum, frame) -> None:  # pragma: no cover - signal path
    del signum, frame
    global RUNNING
    RUNNING = False


def read_gpu_metrics() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    timestamp = time.time()
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
                "utilization_memory_pct": int(parts[5]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Where to save JSONL samples")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds",
    )
    parser.add_argument(
        "--gpu-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional physical GPU indices to keep",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    keep = set(args.gpu_indices) if args.gpu_indices is not None else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fout:
        while RUNNING:
            for row in read_gpu_metrics():
                if keep is not None and row["gpu_index"] not in keep:
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    main()
