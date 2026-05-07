#!/usr/bin/env python3
"""执行一次最小化的本地 vLLM 模型加载和生成测试。

这个脚本直接使用 vLLM Python API 初始化本地模型，不经过 HTTP 服务层，
然后用一条简单 prompt 触发一次生成，最后把输出结果以 JSON 打印出来。
它的目标是把“模型能不能在当前参数下成功加载”和“服务链路有没有问题”
这两个问题分离开来。

典型用途：
1. 在正式跑服务前先验证模型加载参数是否可行。
2. 调整 `tensor_parallel_size`、`cpu_offload_gb`、`max_model_len` 时做最小测试。
3. 在脚本失败时更快定位是模型初始化还是请求发送阶段的问题。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def shutdown_llm(llm) -> None:
    """Best-effort shutdown for vLLM objects."""
    if llm is None:
        return

    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return

    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/workspace/qwen1.5-MoE-A2.7B"),
        help="Local model directory",
    )
    parser.add_argument("--prompt", default="请用一句话解释什么是混合专家模型。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import torch

    llm = None
    try:
        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=args.cpu_offload_gb,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
        )
        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        outputs = llm.generate([args.prompt], sampling_params)

        result = {
            "model": str(args.model),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "outputs": [],
        }
        for item in outputs:
            result["outputs"].append(
                {
                    "prompt": item.prompt,
                    "generated_text": item.outputs[0].text if item.outputs else "",
                    "token_ids": item.outputs[0].token_ids if item.outputs else [],
                    "finish_reason": item.outputs[0].finish_reason if item.outputs else None,
                }
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        shutdown_llm(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
