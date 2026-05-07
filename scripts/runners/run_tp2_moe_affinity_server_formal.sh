#!/usr/bin/env bash
# 用途：
#   作为 TP=2 服务端内嵌 MoE-affinity 实验的发令脚本。客户端保持 FCFS
#   原始提交顺序，vLLM 服务端通过 `--scheduling-policy moe_affinity`
#   在 waiting queue 的 top-K 候选中按专家 signature affinity 动态重排。
#
# 关键环境变量：
#   - VLLM_MOE_AFFINITY_SIGNATURE_FILE：request_id -> expert signature JSONL。
#   - VLLM_MOE_AFFINITY_SCORE_FILE：可选基础分数 JSON/JSONL；未设置时退化为
#     prompt_tokens + max_tokens。
#   - VLLM_MOE_AFFINITY_TOPK：每次动态重排的候选窗口，默认 8。
#   - VLLM_MOE_AFFINITY_BUCKET_SIZE：基础分数桶宽，默认 32。
#   - VLLM_MOE_AFFINITY_WEIGHT：affinity 加权，默认 1.0。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
REPEATS="${REPEATS:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-240}"
LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"
SIGNATURE_FILE="${VLLM_MOE_AFFINITY_SIGNATURE_FILE:-}"
SCORE_FILE="${VLLM_MOE_AFFINITY_SCORE_FILE:-}"

if [[ -z "$SIGNATURE_FILE" ]]; then
  echo "VLLM_MOE_AFFINITY_SIGNATURE_FILE must be set" >&2
  exit 1
fi

RUN_LOG_DIR="/root/workspace/logs/runs"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_tp2_moe_affinity_server_formal_launch.md"
mkdir -p "$RUN_LOG_DIR"

cat >"$RUN_NOTE" <<EOF
# TP2 MoE-Affinity Server-side Formal Launch

- Timestamp: $TIMESTAMP
- Strategy: moe_affinity
- Implementation: vLLM server-side scheduling policy
- Server scheduling policy: moe_affinity
- Client dispatch strategy: fcfs
- Request file: $REQUEST_FILE
- Max requests: $MAX_REQUESTS
- Repeats: $REPEATS
- Load levels: $LOAD_LEVELS
- Signature file: $SIGNATURE_FILE
- Base score file: ${SCORE_FILE:-<unset>}
- Affinity top-K: ${VLLM_MOE_AFFINITY_TOPK:-8}
- Affinity bucket size: ${VLLM_MOE_AFFINITY_BUCKET_SIZE:-32}
- Affinity weight: ${VLLM_MOE_AFFINITY_WEIGHT:-1.0}
- Runner: /root/workspace/scripts/runners/run_tp2_load_levels.sh
- Scheduling note: vLLM uses a length/LTR-like base order, then reranks a
  small candidate window by expert-signature affinity to the current running
  batch.
EOF

env \
  STRATEGY="moe_affinity" \
  SCHEDULING_POLICY="moe_affinity" \
  CLIENT_DISPATCH_STRATEGY="fcfs" \
  TIMESTAMP_PREFIX="tp2_moe_affinity_server" \
  REQUEST_FILE="$REQUEST_FILE" \
  REPEATS="$REPEATS" \
  MAX_REQUESTS="$MAX_REQUESTS" \
  LOAD_LEVELS="$LOAD_LEVELS" \
  VLLM_MOE_AFFINITY_SIGNATURE_FILE="$SIGNATURE_FILE" \
  VLLM_MOE_AFFINITY_SCORE_FILE="$SCORE_FILE" \
  VLLM_MOE_AFFINITY_SCORE_DIRECTION="${VLLM_MOE_AFFINITY_SCORE_DIRECTION:-ascending}" \
  VLLM_MOE_AFFINITY_TOPK="${VLLM_MOE_AFFINITY_TOPK:-8}" \
  VLLM_MOE_AFFINITY_BUCKET_SIZE="${VLLM_MOE_AFFINITY_BUCKET_SIZE:-32}" \
  VLLM_MOE_AFFINITY_WEIGHT="${VLLM_MOE_AFFINITY_WEIGHT:-1.0}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}" \
  CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-4}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  bash /root/workspace/scripts/runners/run_tp2_load_levels.sh
