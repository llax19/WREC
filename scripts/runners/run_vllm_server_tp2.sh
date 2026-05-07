#!/usr/bin/env bash
# 用途：
#   用最少参数直接启动一个双卡 TP=2 的 vLLM 服务，适合手工联调或作为
#   独立服务进程运行。与 `run_stage1_dual_gpu.sh` 不同，这个脚本不会做
#   GPU 监控、请求发送或结果汇总，只负责把服务拉起来。
#
# 适用场景：
#   1. 想手工用 curl / Python 客户端连一个长期运行的本地服务。
#   2. 想快速验证模型在双卡 TP=2 配置下能否成功启动。
#
# 默认参数：
#   - 模型：`/root/workspace/qwen1.5-MoE-A2.7B`
#   - TP：2
#   - GPU 利用率：0.70
#   - CPU offload：8 GB
#   - 最大上下文长度：1024
#   - 调度策略：fcfs，可通过 `SCHEDULING_POLICY=length_only` 覆盖
#
# 备注：
#   额外参数会通过 `"$@"` 原样转发给 `vllm serve`。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"

vllm serve /root/workspace/qwen1.5-MoE-A2.7B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70 \
  --cpu-offload-gb 8 \
  --max-model-len 1024 \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
