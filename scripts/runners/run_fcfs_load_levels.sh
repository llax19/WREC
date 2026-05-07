#!/usr/bin/env bash
# 用途：
#   以单卡模式批量运行 FCFS 基线实验，并自动覆盖 low / medium / high
#   三档负载。这个脚本本身不直接启动服务，而是为每一轮实验设置时间戳、
#   到达间隔、最大并发等环境变量，然后调用 `run_stage1_single_gpu.sh`。
#
# 适用场景：
#   1. 先在单卡上做基线校准，确认实验链路可跑通。
#   2. 对三档负载做快速摸底，观察 FCFS 的吞吐和时延。
#
# 关键输入：
#   - REQUEST_FILE：请求清单 JSONL。
#   - REPEATS：每个负载档位重复次数。
#   - PHYSICAL_GPU_INDEX：单卡实验使用的物理 GPU 编号。
#
# 主要输出：
#   实际输出由 `run_stage1_single_gpu.sh` 统一写入 `logs/`、`results/`
#   和 `logs/runs/`、`logs/server/`。
set -euo pipefail

LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"
REPEATS="${REPEATS:-1}"
PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-1}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/requests_template.jsonl}"
MAX_REQUESTS="${MAX_REQUESTS:-}"

for level in $LOAD_LEVELS; do
  case "$level" in
    low)
      ARRIVAL_INTERVAL="${LOW_ARRIVAL_INTERVAL:-8.0}"
      MAX_CONCURRENCY="${LOW_MAX_CONCURRENCY:-1}"
      ;;
    medium)
      ARRIVAL_INTERVAL="${MEDIUM_ARRIVAL_INTERVAL:-1.5}"
      MAX_CONCURRENCY="${MEDIUM_MAX_CONCURRENCY:-2}"
      ;;
    high)
      ARRIVAL_INTERVAL="${HIGH_ARRIVAL_INTERVAL:-0.0}"
      MAX_CONCURRENCY="${HIGH_MAX_CONCURRENCY:-6}"
      ;;
    *)
      echo "unknown load level: $level" >&2
      exit 1
      ;;
  esac

  for repeat in $(seq 1 "$REPEATS"); do
    TIMESTAMP="fcfs_${level}_r${repeat}_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "running level=$level repeat=$repeat arrival_interval=$ARRIVAL_INTERVAL max_concurrency=$MAX_CONCURRENCY"
    env \
      TIMESTAMP="$TIMESTAMP" \
      STRATEGY=fcfs \
      LOAD_LEVEL="$level" \
      PHYSICAL_GPU_INDEX="$PHYSICAL_GPU_INDEX" \
      REQUEST_FILE="$REQUEST_FILE" \
      MAX_REQUESTS="$MAX_REQUESTS" \
      ARRIVAL_INTERVAL="$ARRIVAL_INTERVAL" \
      MAX_CONCURRENCY="$MAX_CONCURRENCY" \
      bash /root/workspace/scripts/runners/run_stage1_single_gpu.sh
  done
done
