#!/usr/bin/env bash
# 用途：
#   作为 TP=2 Length-only 客户端代理实验的发令脚本。它会先记录本次
#   实验参数，再调用 `run_tp2_load_levels.sh` 批量执行三档负载与多次重复实验。
#
# 说明：
#   Length-only 在当前实现里属于轻量级近似长度排序：客户端在发请求前按
#   `input_tokens + target_max_new_tokens` 从小到大重排，从而形成可复现实验。
#   这不是最终期望的 vLLM 服务端内嵌调度实现，结果只能作为 proxy /
#   preliminary 对照使用。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
REPEATS="${REPEATS:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-240}"
LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"

RUN_LOG_DIR="/root/workspace/logs/runs"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_tp2_length_only_proxy_launch.md"
mkdir -p "$RUN_LOG_DIR"

cat >"$RUN_NOTE" <<EOF
# TP2 Length-only Proxy Launch

- Timestamp: $TIMESTAMP
- Strategy: length_only
- Implementation: client-side request reordering proxy
- Server scheduling policy: fcfs
- Client dispatch strategy: length_only
- Request file: $REQUEST_FILE
- Max requests: $MAX_REQUESTS
- Repeats: $REPEATS
- Load levels: $LOAD_LEVELS
- Runner: /root/workspace/scripts/runners/run_tp2_load_levels.sh
- Scheduling note: sort requests by estimated total tokens before submission
- Validity note: this is not the final vLLM-integrated scheduler result
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
  STRATEGY="length_only" \
  SCHEDULING_POLICY="fcfs" \
  CLIENT_DISPATCH_STRATEGY="length_only" \
  TIMESTAMP_PREFIX="tp2_length_only_proxy" \
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
