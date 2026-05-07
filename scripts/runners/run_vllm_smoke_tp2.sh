#!/usr/bin/env bash
# 用途：
#   以双卡 TP=2 配置运行一次最小化本地加载测试，确认模型能否成功初始化并
#   生成一段短文本。它是“先确认能跑起来，再进入正式实验”的保守入口。
#
# 与 `run_vllm_server_tp2.sh` 的区别：
#   - 这个脚本不会启动 HTTP 服务。
#   - 它直接调用 `vllm_smoke_load.py`，在本地构造一次推理请求并打印结果。
#
# 适用场景：
#   1. 新环境刚装好，想先验证双卡 TP=2 是否能加载模型。
#   2. 调参后怀疑是服务层而不是模型加载层出了问题时，先做最小复现。
#
# 备注：
#   额外参数会转发给 `vllm_smoke_load.py`，例如可以覆盖 `--max-model-len`
#   或 `--cpu-offload-gb`。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python /root/workspace/scripts/runtime/vllm_smoke_load.py \
  --model /root/workspace/qwen1.5-MoE-A2.7B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70 \
  --cpu-offload-gb 8 \
  --max-model-len 1024 \
  "$@"
