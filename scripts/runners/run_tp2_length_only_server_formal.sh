#!/usr/bin/env bash
# 用途：
#   作为 TP=2 服务端内嵌 Length-only 实验的发令脚本。客户端保持 FCFS
#   原始提交顺序，vLLM 服务端通过 `--scheduling-policy length_only`
#   在内部 waiting queue 中按估计长度选择请求。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
REPEATS="${REPEATS:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-240}"
LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"

RUN_LOG_DIR="/root/workspace/logs/runs"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_tp2_length_only_server_formal_launch.md"
mkdir -p "$RUN_LOG_DIR"

cat >"$RUN_NOTE" <<EOF
# TP2 Length-only Server-side Formal Launch

- Timestamp: $TIMESTAMP
- Strategy: length_only
- Implementation: vLLM server-side scheduling policy
- Server scheduling policy: length_only
- Client dispatch strategy: fcfs
- Request file: $REQUEST_FILE
- Max requests: $MAX_REQUESTS
- Repeats: $REPEATS
- Load levels: $LOAD_LEVELS
- Runner: /root/workspace/scripts/runners/run_tp2_load_levels.sh
- Scheduling note: vLLM waiting queue sorts by estimated total tokens
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
  SCHEDULING_POLICY="length_only" \
  CLIENT_DISPATCH_STRATEGY="fcfs" \
  TIMESTAMP_PREFIX="tp2_length_only_server" \
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
