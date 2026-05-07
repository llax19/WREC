#!/usr/bin/env bash
# 用途：
#   这是双 GPU Tensor Parallel 实验的总入口脚本。它负责激活实验环境、
#   启动 GPU 监控、拉起 vLLM 服务、等待服务就绪、发送请求并在结束后汇总
#   原始日志和摘要结果。
#
# 执行流程：
#   1. 激活 `vllm_moe` 环境。
#   2. 启动 `monitor_gpu_memory.py` 记录显存使用。
#   3. 按给定 TP 参数启动 `vllm serve`。
#   4. 等待 `/health` 或 `/v1/models` 可访问。
#   5. 调用 `log_vllm_requests.py` 发送请求并记录逐请求时延。
#   6. 调用 `analyze_logs.py` 生成摘要指标。
#
# 关键环境变量：
#   - MODEL_PATH：本地模型目录。
#   - PHYSICAL_GPU_INDICES / CUDA_VISIBLE_DEVICES：双卡映射方式。
#   - GPU_MEMORY_UTILIZATION / CPU_OFFLOAD_GB / MAX_MODEL_LEN：装载参数。
#   - ARRIVAL_INTERVAL / MAX_CONCURRENCY：负载控制参数。
#
# 输出位置：
#   - `logs/raw/`：请求日志与 GPU 监控日志。
#   - `logs/processed/`：GPU 摘要。
#   - `results/`：请求指标摘要。
#   - `logs/server/`：vLLM 服务端日志。
#   - `logs/runs/`：本轮运行记录。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-0 1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-/root/workspace/qwen1.5-MoE-A2.7B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/requests_template.jsonl}"
LOAD_LEVEL="${LOAD_LEVEL:-debug}"
STRATEGY="${STRATEGY:-fcfs}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
CLIENT_DISPATCH_STRATEGY="${CLIENT_DISPATCH_STRATEGY:-$STRATEGY}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1.0}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

RAW_LOG="/root/workspace/logs/raw/${TIMESTAMP}_${STRATEGY}_${LOAD_LEVEL}.jsonl"
GPU_LOG="/root/workspace/logs/raw/${TIMESTAMP}_gpu_metrics.jsonl"
GPU_SUMMARY="/root/workspace/logs/processed/${TIMESTAMP}_gpu_summary.json"
REQUEST_SUMMARY="/root/workspace/results/${TIMESTAMP}_${STRATEGY}_${LOAD_LEVEL}_summary.json"
SERVER_LOG_DIR="/root/workspace/logs/server"
RUN_LOG_DIR="/root/workspace/logs/runs"
SERVER_LOG="${SERVER_LOG_DIR}/${TIMESTAMP}_vllm_server.log"
RUN_NOTE="${RUN_LOG_DIR}/${TIMESTAMP}_dual_gpu_run.md"

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

read -r -a GPU_INDEX_ARGS <<< "$PHYSICAL_GPU_INDICES"

python /root/workspace/scripts/runtime/monitor_gpu_memory.py \
  --output "$GPU_LOG" \
  --interval "$MONITOR_INTERVAL" \
  --gpu-indices "${GPU_INDEX_ARGS[@]}" &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --max-model-len "$MAX_MODEL_LEN" \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --host "$HOST" \
  --port "$PORT" \
  $(if [[ "$ENFORCE_EAGER" == "1" ]]; then printf '%s' "--enforce-eager"; fi) \
  $(if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then printf '%s' "--disable-custom-all-reduce"; fi) \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

python /root/workspace/scripts/runtime/wait_vllm_server.py \
  --base-url "$BASE_URL" \
  --timeout "$SERVER_READY_TIMEOUT" \
  --pid "$SERVER_PID"

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
# Dual GPU Run Record

- Timestamp: $TIMESTAMP
- Model: $MODEL_PATH
- Base URL: $BASE_URL
- Physical GPU indices: $PHYSICAL_GPU_INDICES
- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
- Tensor parallel size: $TENSOR_PARALLEL_SIZE
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

echo "dual gpu run completed"
echo "raw request log: $RAW_LOG"
echo "gpu metrics log: $GPU_LOG"
echo "request summary: $REQUEST_SUMMARY"
