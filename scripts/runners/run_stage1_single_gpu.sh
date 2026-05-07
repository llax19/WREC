#!/usr/bin/env bash
# 用途：
#   这是单 GPU 实验的总入口脚本，负责从环境激活到结果落盘的整条链路。
#   它适合做最初的环境验证、单卡基线实验和请求日志采集。
#
# 执行流程：
#   1. 激活 `vllm_moe` 环境。
#   2. 启动单卡 GPU 监控。
#   3. 以单卡模式启动 vLLM 服务。
#   4. 等待服务就绪。
#   5. 调用请求发送脚本记录逐请求日志。
#   6. 生成请求摘要和运行记录。
#
# 关键环境变量：
#   - PHYSICAL_GPU_INDEX：使用哪张物理 GPU。
#   - MODEL_PATH：模型目录。
#   - GPU_MEMORY_UTILIZATION / CPU_OFFLOAD_GB / MAX_MODEL_LEN：装载相关参数。
#   - REQUEST_FILE / ARRIVAL_INTERVAL / MAX_CONCURRENCY：实验负载参数。
#
# 输出位置：
#   - `logs/raw/`：原始请求日志、GPU 监控日志。
#   - `logs/processed/`：GPU 摘要。
#   - `results/`：实验指标摘要。
#   - `logs/server/`：vLLM 服务端日志。
#   - `logs/runs/`：本轮运行记录。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$PHYSICAL_GPU_INDEX}"
MODEL_PATH="${MODEL_PATH:-/root/workspace/qwen1.5-MoE-A2.7B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/requests_template.jsonl}"
LOAD_LEVEL="${LOAD_LEVEL:-debug}"
STRATEGY="${STRATEGY:-fcfs}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
CLIENT_DISPATCH_STRATEGY="${CLIENT_DISPATCH_STRATEGY:-$STRATEGY}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1.0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

RAW_LOG="/root/workspace/logs/raw/${TIMESTAMP}_${STRATEGY}_${LOAD_LEVEL}.jsonl"
GPU_LOG="/root/workspace/logs/raw/${TIMESTAMP}_gpu_metrics.jsonl"
GPU_SUMMARY="/root/workspace/logs/processed/${TIMESTAMP}_gpu_summary.json"
REQUEST_SUMMARY="/root/workspace/results/${TIMESTAMP}_${STRATEGY}_${LOAD_LEVEL}_summary.json"
SERVER_LOG_DIR="/root/workspace/logs/server"
RUN_LOG_DIR="/root/workspace/logs/runs"
SERVER_LOG="${SERVER_LOG_DIR}/${TIMESTAMP}_vllm_server.log"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_stage1_run.md"

mkdir -p /root/workspace/logs/raw /root/workspace/logs/processed /root/workspace/results "$SERVER_LOG_DIR" "$RUN_LOG_DIR"

MONITOR_PID=""
SERVER_PID=""

cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

python /root/workspace/scripts/runtime/monitor_gpu_memory.py \
  --output "$GPU_LOG" \
  --interval "$MONITOR_INTERVAL" \
  --gpu-indices "$PHYSICAL_GPU_INDEX" &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --max-model-len "$MAX_MODEL_LEN" \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --host "$HOST" \
  --port "$PORT" \
  $(if [[ "$ENFORCE_EAGER" == "1" ]]; then printf '%s' "--enforce-eager"; fi) \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

python /root/workspace/scripts/runtime/wait_vllm_server.py --base-url "$BASE_URL" --timeout 900 --pid "$SERVER_PID"

LOGGER_ARGS=(
  --request-file "$REQUEST_FILE"
  --output-log "$RAW_LOG"
  --base-url "$BASE_URL"
  --model-name "$MODEL_PATH"
  --strategy "$STRATEGY"
  --dispatch-strategy "$CLIENT_DISPATCH_STRATEGY"
  --server-scheduling-policy "$SCHEDULING_POLICY"
  --load-level "$LOAD_LEVEL"
  --tokenizer-path "$MODEL_PATH"
  --gpu-metrics-log "$GPU_LOG"
  --gpu-metrics-summary-output "$GPU_SUMMARY"
  --arrival-interval "$ARRIVAL_INTERVAL"
  --max-concurrency "$MAX_CONCURRENCY"
)

if [[ -n "$MAX_REQUESTS" ]]; then
  LOGGER_ARGS+=(--limit "$MAX_REQUESTS")
fi

python /root/workspace/scripts/runtime/log_vllm_requests.py "${LOGGER_ARGS[@]}"
python /root/workspace/scripts/analysis/analyze_logs.py "$RAW_LOG" --output "$REQUEST_SUMMARY"

cat >"$RUN_NOTE" <<EOF
# Stage 1 Run Record

- Timestamp: $TIMESTAMP
- Model: $MODEL_PATH
- Base URL: $BASE_URL
- Physical GPU index: $PHYSICAL_GPU_INDEX
- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
- Strategy: $STRATEGY
- Server scheduling policy: $SCHEDULING_POLICY
- Client dispatch strategy: $CLIENT_DISPATCH_STRATEGY
- Load level: $LOAD_LEVEL
- Arrival interval: $ARRIVAL_INTERVAL
- Max concurrency: $MAX_CONCURRENCY
- Request file: $REQUEST_FILE
- Raw request log: $RAW_LOG
- GPU metrics log: $GPU_LOG
- GPU summary: $GPU_SUMMARY
- Request summary: $REQUEST_SUMMARY
- Server log: $SERVER_LOG
EOF

echo "stage1 run completed"
echo "raw request log: $RAW_LOG"
echo "gpu metrics log: $GPU_LOG"
echo "request summary: $REQUEST_SUMMARY"
