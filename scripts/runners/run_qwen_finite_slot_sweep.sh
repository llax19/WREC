#!/usr/bin/env bash
# 用途：
#   对 Qwen 单卡 finite-slot runtime 做 slot_capacity x max_num_batched_tokens
#   的受控 sweep。每次运行使用一个请求清单，遍历多个 case，并为每个 case
#   单独启动 sidecar 与 vLLM server，输出请求日志、sidecar metrics 和服务日志。
#
# 默认：
#   - 请求清单：n32 短 eval 子集
#   - slot：16 24 32
#   - max_num_batched_tokens：4 8
#
# 输出：
#   - 结果 JSONL：results/wrec/runtime_finite_slot_qwen_single_gpu_sweep_*/
#   - sidecar metrics：同目录 *_sidecar_metrics.json
#   - case 汇总：同目录 summary.jsonl / summary.csv
#   - 服务日志：workspace/logs/server/qwen_finite_slot_sweep_*/
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

MODEL_PATH="${MODEL_PATH:-/root/WREC/models/qwen1.5-MoE-A2.7B}"
TRAIN_TRACE="${TRAIN_TRACE:-/root/WREC/logs/processed/wrec/qwen_single_gpu_20260511/train_n256.jsonl}"
REQUEST_FILE="${REQUEST_FILE:-/root/WREC/data/prompts/qwen_finite_slot_eval_short_n32_20260511.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-28}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
SLOT_CAPACITIES="${SLOT_CAPACITIES:-16 24 32}"
MAX_BATCHED_TOKENS_SET="${MAX_BATCHED_TOKENS_SET:-4 8}"
SIDECAR_TOTAL_SLOTS="${SIDECAR_TOTAL_SLOTS:-192}"
SIDECAR_PORT="${SIDECAR_PORT:-18765}"
SERVER_PORT="${SERVER_PORT:-18000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_TAG="${SWEEP_TAG:-qwen_finite_slot_sweep_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-/root/WREC/results/wrec/runtime_finite_slot_qwen_single_gpu_sweep_${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-/root/WREC/logs/server/${SWEEP_TAG}}"
LOAD_LEVEL="${LOAD_LEVEL:-finite_slot_sweep}"
ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
SUMMARY_JSONL="${RESULT_ROOT}/summary.jsonl"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
printf '%s\n' \
  'case_id,status,slot_capacity,max_num_batched_tokens,requests_completed,mean_latency_s,max_latency_s,input_tokens,output_tokens,router_events,expert_refs,shadow_hits,shadow_misses,would_admit,would_bypass,would_evict,error_kind,error_message,request_log,sidecar_metrics_json,server_log,sidecar_log' \
  >"$SUMMARY_CSV"

cleanup() {
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

echo "result_root=$RESULT_ROOT"
echo "log_root=$LOG_ROOT"
echo "request_file=$REQUEST_FILE"

for SLOT in $SLOT_CAPACITIES; do
  for MBT in $MAX_BATCHED_TOKENS_SET; do
    CASE_ID="slot${SLOT}_mbt${MBT}"
    SIDE_LOG="${LOG_ROOT}/${CASE_ID}_sidecar.log"
    SERVER_LOG="${LOG_ROOT}/${CASE_ID}_server.log"
    REQUEST_LOG="${RESULT_ROOT}/${CASE_ID}_request_log.jsonl"
    METRICS_JSON="${RESULT_ROOT}/${CASE_ID}_sidecar_metrics.json"
    REQUEST_STDOUT_JSON="${RESULT_ROOT}/${CASE_ID}_request_stdout.json"
    SUMMARY_CASE_JSON="${RESULT_ROOT}/${CASE_ID}_summary.json"

    echo "=== running case $CASE_ID ==="
    cleanup
    : >"$SIDE_LOG"
    : >"$SERVER_LOG"

    CASE_STATUS="unknown"
    ERROR_KIND=""
    ERROR_MESSAGE=""
    REQUESTS_COMPLETED=0
    MEAN_LATENCY_S=""
    MAX_LATENCY_S=""
    INPUT_TOKENS=""
    OUTPUT_TOKENS=""
    ROUTER_EVENTS=""
    EXPERT_REFS=""
    SHADOW_HITS=""
    SHADOW_MISSES=""
    WOULD_ADMIT=""
    WOULD_BYPASS=""
    WOULD_EVICT=""

    PYTHONPATH=/root/WREC/scripts \
    python /root/WREC/scripts/wrec/runtime/runtime_sidecar.py \
      --host 127.0.0.1 \
      --port "$SIDECAR_PORT" \
      --train-trace "$TRAIN_TRACE" \
      --model-path "$MODEL_PATH" \
      --total-slots "$SIDECAR_TOTAL_SLOTS" \
      >"$SIDE_LOG" 2>&1 &
    SIDECAR_PID=$!

    export CUDA_VISIBLE_DEVICES
    export WREC_SIDECAR_URL="http://127.0.0.1:${SIDECAR_PORT}/event"
    export WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0
    export WREC_EXPERT_RESIDENCY=1
    export WREC_EXPERT_RESIDENCY_SLOT_CAPACITY="$SLOT"
    export WREC_EXPERT_RESIDENCY_ROW_COPY=0

    vllm serve "$MODEL_PATH" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb "$CPU_OFFLOAD_GB" \
      --offload-backend uva \
      --cpu-offload-params experts.w13_weight experts.w2_weight \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MBT" \
      --host 127.0.0.1 \
      --port "$SERVER_PORT" \
      --enable-return-routed-experts \
      --enforce-eager \
      --trust-remote-code \
      >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    python /root/WREC/scripts/runtime/wait_vllm_server.py \
      --base-url "http://127.0.0.1:${SERVER_PORT}" \
      --timeout 900 \
      --interval 2 \
      --pid "$SERVER_PID"

    set +e
    python /root/WREC/scripts/runtime/log_vllm_requests.py \
      --request-file "$REQUEST_FILE" \
      --output-log "$REQUEST_LOG" \
      --base-url "http://127.0.0.1:${SERVER_PORT}" \
      --model-name "$MODEL_PATH" \
      --strategy fcfs \
      --dispatch-strategy fcfs \
      --server-scheduling-policy fcfs \
      --load-level "${LOAD_LEVEL}_${CASE_ID}" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --arrival-interval "$ARRIVAL_INTERVAL" \
      --max-concurrency "$MAX_CONCURRENCY" \
      >"$REQUEST_STDOUT_JSON" 2>&1
    REQUEST_EXIT_CODE=$?
    set -e

    python - <<PY
import json
import urllib.request
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open("http://127.0.0.1:${SIDECAR_PORT}/metrics", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    payload = {}
out_path = Path("${METRICS_JSON}")
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({
    "case": "${CASE_ID}",
    "router_events": payload.get("router_events"),
    "expert_refs": payload.get("expert_refs"),
    "shadow_miss_rate": payload.get("shadow_miss_rate"),
    "sidecar_metrics_json": str(out_path),
}, ensure_ascii=False))
PY

    if [[ -f "$REQUEST_LOG" ]]; then
      REQUESTS_COMPLETED="$(wc -l < "$REQUEST_LOG" | tr -d ' ')"
    fi

    if [[ -s "$REQUEST_LOG" ]]; then
      REQUEST_SUMMARY="$(python - <<PY
import json
from pathlib import Path
path = Path("${REQUEST_LOG}")
rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
print(json.dumps({
    "requests_completed": len(rows),
    "mean_latency_s": sum((row["finish_time"] - row["submit_time"]) for row in rows) / len(rows),
    "max_latency_s": max((row["finish_time"] - row["submit_time"]) for row in rows),
    "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
    "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
}))
PY
)"
      REQUESTS_COMPLETED="$(python - <<PY
import json
print(json.loads('''${REQUEST_SUMMARY}''')["requests_completed"])
PY
)"
      MEAN_LATENCY_S="$(python - <<PY
import json
print(json.loads('''${REQUEST_SUMMARY}''')["mean_latency_s"])
PY
)"
      MAX_LATENCY_S="$(python - <<PY
import json
print(json.loads('''${REQUEST_SUMMARY}''')["max_latency_s"])
PY
)"
      INPUT_TOKENS="$(python - <<PY
import json
print(json.loads('''${REQUEST_SUMMARY}''')["input_tokens"])
PY
)"
      OUTPUT_TOKENS="$(python - <<PY
import json
print(json.loads('''${REQUEST_SUMMARY}''')["output_tokens"])
PY
)"
    fi

    if [[ -f "$METRICS_JSON" ]]; then
      METRICS_SUMMARY="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${METRICS_JSON}").read_text(encoding="utf-8"))
print(json.dumps({
    "router_events": payload.get("router_events", ""),
    "expert_refs": payload.get("expert_refs", ""),
    "shadow_hits": payload.get("shadow_hits", ""),
    "shadow_misses": payload.get("shadow_misses", ""),
    "would_admit": payload.get("would_admit", ""),
    "would_bypass": payload.get("would_bypass", ""),
    "would_evict": payload.get("would_evict", ""),
}))
PY
)"
      ROUTER_EVENTS="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["router_events"])
PY
)"
      EXPERT_REFS="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["expert_refs"])
PY
)"
      SHADOW_HITS="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["shadow_hits"])
PY
)"
      SHADOW_MISSES="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["shadow_misses"])
PY
)"
      WOULD_ADMIT="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["would_admit"])
PY
)"
      WOULD_BYPASS="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["would_bypass"])
PY
)"
      WOULD_EVICT="$(python - <<PY
import json
print(json.loads('''${METRICS_SUMMARY}''')["would_evict"])
PY
)"
    fi

    if [[ "$REQUEST_EXIT_CODE" -eq 0 ]]; then
      CASE_STATUS="success"
    elif rg -n "slot overflow|WREC expert residency slot overflow" "$SERVER_LOG" >/dev/null 2>&1; then
      CASE_STATUS="overflow"
      ERROR_KIND="slot_overflow"
      ERROR_MESSAGE="$(rg -n "slot overflow|WREC expert residency slot overflow" "$SERVER_LOG" | tail -n 1 | sed 's/"/'\''/g')"
    else
      CASE_STATUS="runtime_error"
      ERROR_KIND="request_driver_failed"
      ERROR_MESSAGE="$(tail -n 20 "$REQUEST_STDOUT_JSON" "$SERVER_LOG" 2>/dev/null | tr '\n' ' ' | sed 's/"/'\''/g' | cut -c1-1000)"
    fi

    python - <<PY
import json
from pathlib import Path

payload = {
    "case_id": "${CASE_ID}",
    "status": "${CASE_STATUS}",
    "slot_capacity": ${SLOT},
    "max_num_batched_tokens": ${MBT},
    "requests_completed": ${REQUESTS_COMPLETED:-0},
    "mean_latency_s": ${MEAN_LATENCY_S:-0.0},
    "max_latency_s": ${MAX_LATENCY_S:-0.0},
    "input_tokens": ${INPUT_TOKENS:-0},
    "output_tokens": ${OUTPUT_TOKENS:-0},
    "router_events": ${ROUTER_EVENTS:-0},
    "expert_refs": ${EXPERT_REFS:-0},
    "shadow_hits": ${SHADOW_HITS:-0},
    "shadow_misses": ${SHADOW_MISSES:-0},
    "would_admit": ${WOULD_ADMIT:-0},
    "would_bypass": ${WOULD_BYPASS:-0},
    "would_evict": ${WOULD_EVICT:-0},
    "error_kind": "${ERROR_KIND}",
    "error_message": "${ERROR_MESSAGE}",
    "request_log": "${REQUEST_LOG}",
    "sidecar_metrics_json": "${METRICS_JSON}",
    "server_log": "${SERVER_LOG}",
    "sidecar_log": "${SIDE_LOG}",
}
Path("${SUMMARY_CASE_JSON}").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
with Path("${SUMMARY_JSONL}").open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\\n")
print(json.dumps(payload, ensure_ascii=False))
PY

    python - <<PY
import csv
import json
from pathlib import Path
row = json.loads(Path("${SUMMARY_CASE_JSON}").read_text(encoding="utf-8"))
fields = [
    "case_id", "status", "slot_capacity", "max_num_batched_tokens",
    "requests_completed", "mean_latency_s", "max_latency_s",
    "input_tokens", "output_tokens", "router_events", "expert_refs",
    "shadow_hits", "shadow_misses", "would_admit", "would_bypass",
    "would_evict", "error_kind", "error_message", "request_log",
    "sidecar_metrics_json", "server_log", "sidecar_log",
]
with Path("${SUMMARY_CSV}").open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writerow({key: row.get(key, "") for key in fields})
PY

    cleanup
  done
done
