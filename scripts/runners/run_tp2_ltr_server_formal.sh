#!/usr/bin/env bash
# 用途：
#   作为 TP=2 服务端内嵌 LTR 实验的发令脚本。客户端保持 FCFS 原始提交
#   顺序，vLLM 服务端通过 `--scheduling-policy ltr` 按外部 predictor 分数
#   调度 waiting queue。
#
# 关键环境变量：
#   - VLLM_LTR_SCORE_FILE：LTR 分数 JSON/JSONL。每行至少包含 request_id/id
#     和 score/ltr_score/predicted_score/rank/predicted_rank 之一。
#   - VLLM_LTR_SCORE_DIRECTION：ascending 或 descending，默认 ascending。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
REPEATS="${REPEATS:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-240}"
LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"
LTR_SCORE_FILE="${VLLM_LTR_SCORE_FILE:-}"
LTR_SCORE_DIRECTION="${VLLM_LTR_SCORE_DIRECTION:-ascending}"

RUN_LOG_DIR="/root/workspace/logs/runs"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_tp2_ltr_server_formal_launch.md"
mkdir -p "$RUN_LOG_DIR"

cat >"$RUN_NOTE" <<EOF
# TP2 LTR Server-side Formal Launch

- Timestamp: $TIMESTAMP
- Strategy: ltr
- Implementation: vLLM server-side scheduling policy
- Server scheduling policy: ltr
- Client dispatch strategy: fcfs
- Request file: $REQUEST_FILE
- Max requests: $MAX_REQUESTS
- Repeats: $REPEATS
- Load levels: $LOAD_LEVELS
- LTR score file: ${LTR_SCORE_FILE:-<unset>}
- LTR score direction: $LTR_SCORE_DIRECTION
- Runner: /root/workspace/scripts/runners/run_tp2_load_levels.sh
- Scheduling note: vLLM waiting queue sorts by external LTR score, then length and arrival tie breakers
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
  STRATEGY="ltr" \
  SCHEDULING_POLICY="ltr" \
  CLIENT_DISPATCH_STRATEGY="fcfs" \
  TIMESTAMP_PREFIX="tp2_ltr_server" \
  REQUEST_FILE="$REQUEST_FILE" \
  REPEATS="$REPEATS" \
  MAX_REQUESTS="$MAX_REQUESTS" \
  LOAD_LEVELS="$LOAD_LEVELS" \
  VLLM_LTR_SCORE_FILE="$LTR_SCORE_FILE" \
  VLLM_LTR_SCORE_DIRECTION="$LTR_SCORE_DIRECTION" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}" \
  CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-4}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  bash /root/workspace/scripts/runners/run_tp2_load_levels.sh
