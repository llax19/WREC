#!/usr/bin/env bash
# 用途：
#   运行 Qwen1.5-MoE + vLLM 的 WREC 真实推理主实验。
#
# 设计：
#   - 主指标由 request log 汇总：request/s、input/total tok/s、p95 TTFT、p95 E2E。
#   - WREC miss rate 只作为机制解释指标，由 sidecar /metrics 提供。
#   - p95 TPOT 仍会汇总，但只作为 decode sanity 指标。
#   - 默认请求集使用较长 decode 的 Dolly n64，而不是 finite-slot 校准用
#     target_max_new_tokens=1 的短请求集。
#
# 默认主实验矩阵：
#   - no_wrec: vLLM CPU offload baseline。
#   - state_only: 启用 WREC sidecar 与 routed expert export，但不限制真实 slot。
#   - finite_slot: slot=16,24,32 且 mbt=4；另加校准中唯一稳定的 slot=32,mbt=8。
#
# 输出：
#   - /root/WREC/results/wrec/runtime_qwen_real_inference_main_<timestamp>/
#   - /root/WREC/logs/server/qwen_real_inference_main_<timestamp>/
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

MODEL_PATH="${MODEL_PATH:-/root/WREC/models/qwen1.5-MoE-A2.7B}"
TRAIN_TRACE="${TRAIN_TRACE:-/root/WREC/logs/processed/wrec/qwen_single_gpu_20260511/train_n256.jsonl}"
REQUEST_FILE="${REQUEST_FILE:-/root/WREC/data/prompts/wrec_dolly_debug_n64.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-28}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1280}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
SIDECAR_TOTAL_SLOTS="${SIDECAR_TOTAL_SLOTS:-192}"
SIDECAR_RANKING_SCORE_THRESHOLD="${SIDECAR_RANKING_SCORE_THRESHOLD:-}"
SIDECAR_BASE_PORT="${SIDECAR_BASE_PORT:-18765}"
SERVER_BASE_PORT="${SERVER_BASE_PORT:-18000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REPEATS="${REPEATS:-3}"
METHODS="${METHODS:-no_wrec state_only finite_slot}"
NO_WREC_MBTS="${NO_WREC_MBTS:-4}"
STATE_ONLY_MBTS="${STATE_ONLY_MBTS:-4}"
FINITE_SLOT_CASES="${FINITE_SLOT_CASES:-16:4 24:4 32:4 32:8}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1.0}"
GPU_MONITOR_INDICES="${GPU_MONITOR_INDICES:-0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-qwen_real_inference_main_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-/root/WREC/results/wrec/runtime_qwen_real_inference_main_${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-/root/WREC/logs/server/${EXPERIMENT_TAG}}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

SUMMARY_JSONL="${RESULT_ROOT}/run_manifest.jsonl"
SUMMARY_CSV="${RESULT_ROOT}/run_manifest.csv"
printf '%s\n' \
  'case_id,status,method,repeat,slot_capacity,max_num_batched_tokens,requests_completed,request_log,sidecar_metrics_json,gpu_metrics_log,gpu_metrics_summary_json,server_log,sidecar_log,error_kind,error_message' \
  >"$SUMMARY_CSV"

cleanup() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    unset GPU_MONITOR_PID
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    unset SERVER_PID
  fi
  if [[ -n "${SIDECAR_PID:-}" ]]; then
    kill "$SIDECAR_PID" >/dev/null 2>&1 || true
    wait "$SIDECAR_PID" >/dev/null 2>&1 || true
    unset SIDECAR_PID
  fi
}

trap cleanup EXIT

write_manifest_row() {
  local case_json="$1"
  python - "$case_json" "$SUMMARY_JSONL" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_json = Path(sys.argv[1])
summary_jsonl = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
row = json.loads(case_json.read_text(encoding="utf-8"))
fields = [
    "case_id", "status", "method", "repeat", "slot_capacity",
    "max_num_batched_tokens", "requests_completed", "request_log",
    "sidecar_metrics_json", "gpu_metrics_log", "gpu_metrics_summary_json",
    "server_log", "sidecar_log",
    "error_kind", "error_message",
]
with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
with summary_csv.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writerow({field: row.get(field, "") for field in fields})
PY
}

start_sidecar() {
  local port="$1"
  local side_log="$2"
  local sidecar_args=(
    /root/WREC/scripts/wrec/runtime/runtime_sidecar.py
    --host 127.0.0.1
    --port "$port"
    --train-trace "$TRAIN_TRACE"
    --model-path "$MODEL_PATH"
    --total-slots "$SIDECAR_TOTAL_SLOTS"
  )
  if [[ -n "$SIDECAR_RANKING_SCORE_THRESHOLD" ]]; then
    sidecar_args+=(--ranking-score-threshold "$SIDECAR_RANKING_SCORE_THRESHOLD")
  fi
  PYTHONPATH=/root/WREC/scripts \
  python "${sidecar_args[@]}" \
    >"$side_log" 2>&1 &
  SIDECAR_PID=$!
}

fetch_sidecar_metrics() {
  local port="$1"
  local output_json="$2"
  python - "$port" "$output_json" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

port = sys.argv[1]
output_json = Path(sys.argv[2])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    payload = {}
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_case() {
  local method="$1"
  local repeat="$2"
  local slot="$3"
  local mbt="$4"
  local case_id="$5"
  local server_port="$6"
  local sidecar_port="$7"

  local side_log="${LOG_ROOT}/${case_id}_sidecar.log"
  local server_log="${LOG_ROOT}/${case_id}_server.log"
  local request_log="${RESULT_ROOT}/${case_id}_request_log.jsonl"
  local request_stdout="${RESULT_ROOT}/${case_id}_request_stdout.txt"
  local metrics_json="${RESULT_ROOT}/${case_id}_sidecar_metrics.json"
  local gpu_metrics_log="${RESULT_ROOT}/${case_id}_gpu_metrics.jsonl"
  local gpu_summary_json="${RESULT_ROOT}/${case_id}_gpu_summary.json"
  local case_json="${RESULT_ROOT}/${case_id}_manifest.json"

  echo "=== running ${case_id} ==="
  cleanup
  : >"$server_log"
  : >"$side_log"

  unset WREC_SIDECAR_URL || true
  unset WREC_SIDECAR_MAX_EVENTS_PER_REQUEST || true
  unset WREC_EXPERT_RESIDENCY || true
  unset WREC_EXPERT_RESIDENCY_SLOT_CAPACITY || true
  unset WREC_EXPERT_RESIDENCY_ROW_COPY || true

  local enable_routed_experts=0
  if [[ "$method" == "state_only" || "$method" == "finite_slot" ]]; then
    start_sidecar "$sidecar_port" "$side_log"
    export WREC_SIDECAR_URL="http://127.0.0.1:${sidecar_port}/event"
    export WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0
    export WREC_EXPERT_RESIDENCY=1
    export WREC_EXPERT_RESIDENCY_ROW_COPY=0
    enable_routed_experts=1
  fi
  if [[ "$method" == "finite_slot" ]]; then
    export WREC_EXPERT_RESIDENCY_SLOT_CAPACITY="$slot"
  fi

  export CUDA_VISIBLE_DEVICES
  local vllm_args=(
    serve "$MODEL_PATH"
    --tensor-parallel-size 1
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --cpu-offload-gb "$CPU_OFFLOAD_GB"
    --offload-backend uva
    --cpu-offload-params experts.w13_weight experts.w2_weight
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$mbt"
    --host 127.0.0.1
    --port "$server_port"
    --enforce-eager
    --trust-remote-code
  )
  if [[ "$enable_routed_experts" -eq 1 ]]; then
    vllm_args+=(--enable-return-routed-experts)
  fi

  vllm "${vllm_args[@]}" >"$server_log" 2>&1 &
  SERVER_PID=$!

  python /root/WREC/scripts/runtime/wait_vllm_server.py \
    --base-url "http://127.0.0.1:${server_port}" \
    --timeout 900 \
    --interval 2 \
    --pid "$SERVER_PID"

  python /root/WREC/scripts/runtime/monitor_gpu_memory.py \
    --output "$gpu_metrics_log" \
    --interval "$GPU_MONITOR_INTERVAL" \
    --gpu-indices $GPU_MONITOR_INDICES \
    >"${LOG_ROOT}/${case_id}_gpu_monitor.log" 2>&1 &
  GPU_MONITOR_PID=$!

  set +e
  python /root/WREC/scripts/runtime/log_vllm_requests.py \
    --request-file "$REQUEST_FILE" \
    --output-log "$request_log" \
    --base-url "http://127.0.0.1:${server_port}" \
    --model-name "$MODEL_PATH" \
    --strategy fcfs \
    --dispatch-strategy fcfs \
    --server-scheduling-policy fcfs \
    --load-level "${method}_slot${slot}_mbt${mbt}_r${repeat}" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --arrival-interval "$ARRIVAL_INTERVAL" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --gpu-metrics-log "$gpu_metrics_log" \
    --gpu-metrics-summary-output "$gpu_summary_json" \
    >"$request_stdout" 2>&1
  local request_exit_code=$?
  set -e

  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    unset GPU_MONITOR_PID
  fi

  if [[ "$method" == "state_only" || "$method" == "finite_slot" ]]; then
    fetch_sidecar_metrics "$sidecar_port" "$metrics_json"
  else
    printf '{}\n' >"$metrics_json"
  fi

  local status="success"
  local error_kind=""
  local error_message=""
  if [[ "$request_exit_code" -ne 0 ]]; then
    if rg -n "slot overflow|WREC expert residency slot overflow" "$server_log" >/dev/null 2>&1; then
      status="overflow"
      error_kind="slot_overflow"
      error_message="$(rg -n "slot overflow|WREC expert residency slot overflow" "$server_log" | tail -n 1 | sed "s/\"/'/g")"
    else
      status="runtime_error"
      error_kind="request_driver_failed"
      error_message="$(tail -n 20 "$request_stdout" "$server_log" 2>/dev/null | tr '\n' ' ' | sed "s/\"/'/g" | cut -c1-1000)"
    fi
  fi

  local requests_completed=0
  if [[ -f "$request_log" ]]; then
    requests_completed="$(wc -l < "$request_log" | tr -d ' ')"
  fi

  python - "$case_json" <<PY
import json
import sys
from pathlib import Path

payload = {
    "case_id": "${case_id}",
    "status": "${status}",
    "method": "${method}",
    "repeat": ${repeat},
    "slot_capacity": ${slot},
    "max_num_batched_tokens": ${mbt},
    "requests_completed": ${requests_completed},
    "request_file": "${REQUEST_FILE}",
    "request_log": "${request_log}",
    "sidecar_metrics_json": "${metrics_json}",
    "gpu_metrics_log": "${gpu_metrics_log}",
    "gpu_metrics_summary_json": "${gpu_summary_json}",
    "server_log": "${server_log}",
    "sidecar_log": "${side_log}",
    "error_kind": "${error_kind}",
    "error_message": "${error_message}",
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
  write_manifest_row "$case_json"
  cleanup
}

echo "result_root=$RESULT_ROOT"
echo "log_root=$LOG_ROOT"
echo "request_file=$REQUEST_FILE"
echo "methods=$METHODS"
echo "no_wrec_mbts=$NO_WREC_MBTS"
echo "state_only_mbts=$STATE_ONLY_MBTS"
echo "finite_slot_cases=$FINITE_SLOT_CASES"
echo "sidecar_ranking_score_threshold=${SIDECAR_RANKING_SCORE_THRESHOLD:-none}"
echo "repeats=$REPEATS"

case_index=0
for repeat in $(seq 1 "$REPEATS"); do
  for method in $METHODS; do
    if [[ "$method" == "no_wrec" ]]; then
      for mbt in $NO_WREC_MBTS; do
        case_index=$((case_index + 1))
        run_case "$method" "$repeat" 0 "$mbt" "${method}_mbt${mbt}_r${repeat}" \
          "$((SERVER_BASE_PORT + case_index))" "$((SIDECAR_BASE_PORT + case_index))"
      done
    elif [[ "$method" == "state_only" ]]; then
      for mbt in $STATE_ONLY_MBTS; do
        case_index=$((case_index + 1))
        run_case "$method" "$repeat" 0 "$mbt" "${method}_mbt${mbt}_r${repeat}" \
          "$((SERVER_BASE_PORT + case_index))" "$((SIDECAR_BASE_PORT + case_index))"
      done
    elif [[ "$method" == "finite_slot" ]]; then
      for item in $FINITE_SLOT_CASES; do
        slot="${item%%:*}"
        mbt="${item##*:}"
        case_index=$((case_index + 1))
        run_case "$method" "$repeat" "$slot" "$mbt" \
          "finite_slot_slot${slot}_mbt${mbt}_r${repeat}" \
          "$((SERVER_BASE_PORT + case_index))" "$((SIDECAR_BASE_PORT + case_index))"
      done
    else
      echo "unknown method: $method" >&2
      exit 1
    fi
  done
done

python /root/WREC/scripts/analysis/summarize_runtime_serving_metrics.py "$RESULT_ROOT" \
  --csv-output "${RESULT_ROOT}/serving_metrics_summary.csv" \
  --json-output "${RESULT_ROOT}/serving_metrics_summary.json"
