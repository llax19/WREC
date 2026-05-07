#!/usr/bin/env bash
# 用途：
#   作为 TP=2 正式 FCFS 实验的“发令脚本”。它不自己跑单轮实验，而是先把
#   本次正式实验的参数写入一份 launch 记录，再调用 `run_tp2_load_levels.sh`
#   批量执行 low / medium / high 三档负载与多次重复实验。
#
# 适用场景：
#   1. 已完成试跑和参数校准，准备收集论文可用结果。
#   2. 希望把正式实验的请求集、重复次数和推荐配置单独记录下来。
#
# 关键环境变量：
#   - REQUEST_FILE：正式实验请求集。
#   - REPEATS：每档负载重复次数。
#   - MAX_REQUESTS：每轮请求上限。
#   - GPU_MEMORY_UTILIZATION / CPU_OFFLOAD_GB / MAX_MODEL_LEN：推荐装载参数。
#
# 主要输出：
#   - `logs/runs/*_tp2_fcfs_formal_launch.md`：本次正式实验的启动记录。
#   - 其余实验输出由 `run_tp2_load_levels.sh` 和 `run_stage1_dual_gpu.sh` 产生。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
REPEATS="${REPEATS:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-240}"
LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"

RUN_LOG_DIR="/root/workspace/logs/runs"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_tp2_fcfs_formal_launch.md"
mkdir -p "$RUN_LOG_DIR"

cat >"$RUN_NOTE" <<EOF
# TP2 FCFS Formal Launch

- Timestamp: $TIMESTAMP
- Strategy: fcfs
- Server scheduling policy: fcfs
- Client dispatch strategy: fcfs
- Request file: $REQUEST_FILE
- Max requests: $MAX_REQUESTS
- Repeats: $REPEATS
- Load levels: $LOAD_LEVELS
- Runner: /root/workspace/scripts/runners/run_tp2_load_levels.sh
- Recommended config:
  - CUDA_VISIBLE_DEVICES=0,1
  - tensor_parallel_size=2
  - gpu_memory_utilization=0.70
  - cpu_offload_gb=4
  - max_model_len=1024
  - disable_custom_all_reduce=True
  - enforce_eager=False
EOF

env \
  STRATEGY="fcfs" \
  SCHEDULING_POLICY="fcfs" \
  CLIENT_DISPATCH_STRATEGY="fcfs" \
  REQUEST_FILE="$REQUEST_FILE" \
  REPEATS="$REPEATS" \
  MAX_REQUESTS="$MAX_REQUESTS" \
  LOAD_LEVELS="$LOAD_LEVELS" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}" \
  CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-4}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  bash /root/workspace/scripts/runners/run_tp2_load_levels.sh
