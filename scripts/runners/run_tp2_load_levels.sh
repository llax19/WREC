#!/usr/bin/env bash
# 用途：
#   以双卡 TP=2 模式批量运行 FCFS 负载分档实验。脚本会遍历低、中、高三档
#   负载，并为每个档位和重复次数生成时间戳，然后调用
#   `run_stage1_dual_gpu.sh` 完成单轮实验。
#
# 适用场景：
#   1. 需要对 TP=2 配置下的负载敏感性做系统测试。
#   2. 作为正式实验或重复实验的批量调度入口。
#
# 关键环境变量：
#   - LOAD_LEVELS：要跑哪些负载档位，默认 `low medium high`。
#   - REPEATS：每档重复次数。
#   - REQUEST_FILE：输入请求清单。
#   - GPU_MEMORY_UTILIZATION / CPU_OFFLOAD_GB / MAX_MODEL_LEN：模型装载参数。
#
# 负载定义：
#   - low：到达间隔较大，并发较低。
#   - medium：中等到达速率与并发。2026-04-24 重新校准后默认使用
#     arrival_interval=0.5, max_concurrency=4，以形成稳定短队列。
#   - high：无到达间隔或极短间隔，高并发压力测试。
set -euo pipefail

LOAD_LEVELS="${LOAD_LEVELS:-low medium high}"
REPEATS="${REPEATS:-1}"
STRATEGY="${STRATEGY:-fcfs}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
CLIENT_DISPATCH_STRATEGY="${CLIENT_DISPATCH_STRATEGY:-$STRATEGY}"
REQUEST_FILE="${REQUEST_FILE:-/root/workspace/data/prompts/eval_requests.jsonl}"
MAX_REQUESTS="${MAX_REQUESTS:-24}"
PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-0 1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-/root/workspace/qwen1.5-MoE-A2.7B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
STRATEGY_TAG="${STRATEGY//-/_}"
TIMESTAMP_PREFIX="${TIMESTAMP_PREFIX:-tp2_${STRATEGY_TAG}}"

for level in $LOAD_LEVELS; do
  case "$level" in
    low)
      ARRIVAL_INTERVAL="${LOW_ARRIVAL_INTERVAL:-8.0}"
      MAX_CONCURRENCY="${LOW_MAX_CONCURRENCY:-1}"
      ;;
    medium)
      ARRIVAL_INTERVAL="${MEDIUM_ARRIVAL_INTERVAL:-0.5}"
      MAX_CONCURRENCY="${MEDIUM_MAX_CONCURRENCY:-4}"
      ;;
    high)
      ARRIVAL_INTERVAL="${HIGH_ARRIVAL_INTERVAL:-0.0}"
      MAX_CONCURRENCY="${HIGH_MAX_CONCURRENCY:-6}"
      ;;
    *)
      echo "unknown load level: $level" >&2
      exit 1
      ;;
  esac

  for repeat in $(seq 1 "$REPEATS"); do
    TIMESTAMP="${TIMESTAMP_PREFIX}_${level}_r${repeat}_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "running tp2 strategy=$STRATEGY scheduling_policy=$SCHEDULING_POLICY dispatch=$CLIENT_DISPATCH_STRATEGY level=$level repeat=$repeat arrival_interval=$ARRIVAL_INTERVAL max_concurrency=$MAX_CONCURRENCY"
    env \
      TIMESTAMP="$TIMESTAMP" \
      STRATEGY="$STRATEGY" \
      SCHEDULING_POLICY="$SCHEDULING_POLICY" \
      CLIENT_DISPATCH_STRATEGY="$CLIENT_DISPATCH_STRATEGY" \
      LOAD_LEVEL="$level" \
      REQUEST_FILE="$REQUEST_FILE" \
      MAX_REQUESTS="$MAX_REQUESTS" \
      ARRIVAL_INTERVAL="$ARRIVAL_INTERVAL" \
      MAX_CONCURRENCY="$MAX_CONCURRENCY" \
      PHYSICAL_GPU_INDICES="$PHYSICAL_GPU_INDICES" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      MODEL_PATH="$MODEL_PATH" \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
      CPU_OFFLOAD_GB="$CPU_OFFLOAD_GB" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      ENFORCE_EAGER="$ENFORCE_EAGER" \
      DISABLE_CUSTOM_ALL_REDUCE="$DISABLE_CUSTOM_ALL_REDUCE" \
      bash /root/workspace/scripts/runners/run_stage1_dual_gpu.sh
  done
done
