#!/usr/bin/env bash
# 用途：
#   对 TP=2 运行参数做小范围扫描，主要比较不同 `cpu_offload_gb`、
#   `enforce_eager` 组合在指定负载下的可运行性和性能表现。
#
# 适用场景：
#   1. 正式实验前先找一组更稳的双卡加载参数。
#   2. 对 graph / eager、不同 offload 组合做快速对比。
#
# 工作方式：
#   - 根据 `LOAD_LEVEL` 推导默认到达间隔和最大并发。
#   - 遍历 `SWEEP_CASES` 中定义的参数组合。
#   - 每个组合调用一次 `run_stage1_dual_gpu.sh`。
#
# 关键环境变量：
#   - SWEEP_CASES：要扫描的参数组合名。
#   - LOAD_LEVEL：当前扫描在哪个负载档位进行。
#   - REQUEST_FILE / MAX_REQUESTS：请求集与请求数量上限。
set -euo pipefail

REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
MAX_REQUESTS="${MAX_REQUESTS:-24}"
LOAD_LEVEL="${LOAD_LEVEL:-medium}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-0 1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-/root/workspace/qwen1.5-MoE-A2.7B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SWEEP_CASES="${SWEEP_CASES:-offload4_graph offload8_eager}"

case "${LOAD_LEVEL}" in
  low)
    ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-8.0}"
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
    ;;
  medium)
    ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-1.5}"
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
    ;;
  high)
    ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0.0}"
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
    ;;
  *)
    ARRIVAL_INTERVAL="${ARRIVAL_INTERVAL:-0.0}"
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
    ;;
esac

for CASE_NAME in $SWEEP_CASES; do
  CPU_OFFLOAD_GB=8
  ENFORCE_EAGER=0

  case "$CASE_NAME" in
    baseline_graph)
      CPU_OFFLOAD_GB=8
      ENFORCE_EAGER=0
      ;;
    offload4_graph)
      CPU_OFFLOAD_GB=4
      ENFORCE_EAGER=0
      ;;
    offload8_eager)
      CPU_OFFLOAD_GB=8
      ENFORCE_EAGER=1
      ;;
    offload4_eager)
      CPU_OFFLOAD_GB=4
      ENFORCE_EAGER=1
      ;;
    *)
      echo "unknown sweep case: $CASE_NAME" >&2
      exit 1
      ;;
  esac

  TIMESTAMP="tp2_${CASE_NAME}_${LOAD_LEVEL}_$(date -u +%Y%m%dT%H%M%SZ)"

  TIMESTAMP="$TIMESTAMP" \
  STRATEGY="fcfs" \
  LOAD_LEVEL="$LOAD_LEVEL" \
  REQUEST_FILE="$REQUEST_FILE" \
  MAX_REQUESTS="$MAX_REQUESTS" \
  ARRIVAL_INTERVAL="$ARRIVAL_INTERVAL" \
  MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  PHYSICAL_GPU_INDICES="$PHYSICAL_GPU_INDICES" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  MODEL_PATH="$MODEL_PATH" \
  BASE_URL="$BASE_URL" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  CPU_OFFLOAD_GB="$CPU_OFFLOAD_GB" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  ENFORCE_EAGER="$ENFORCE_EAGER" \
  DISABLE_CUSTOM_ALL_REDUCE="$DISABLE_CUSTOM_ALL_REDUCE" \
  bash /root/workspace/scripts/runners/run_stage1_dual_gpu.sh
done
