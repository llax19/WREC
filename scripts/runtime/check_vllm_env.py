#!/usr/bin/env python3
"""检查当前 Python / CUDA / vLLM 环境是否满足实验运行条件。

这个脚本会收集当前解释器路径、Python 版本、平台信息、`CUDA_VISIBLE_DEVICES`
以及 `torch`、`transformers`、`vllm` 的导入状态和版本信息。它的目标不是
做完整诊断，而是在实验启动前快速回答两个问题：

1. 当前 conda 环境是不是你期望的那个环境。
2. GPU、PyTorch 和 vLLM 是否至少能被正确识别和导入。

输出是一个结构化 JSON，适合直接保存到实验记录里。
"""

from __future__ import annotations

import json
import os
import platform
import sys


def safe_import(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, str(exc)


def main() -> None:
    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    torch, torch_err = safe_import("torch")
    if torch is not None:
        report["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    else:
        report["torch_import_error"] = torch_err

    transformers, transformers_err = safe_import("transformers")
    if transformers is not None:
        report["transformers"] = {"version": getattr(transformers, "__version__", "unknown")}
    else:
        report["transformers_import_error"] = transformers_err

    vllm, vllm_err = safe_import("vllm")
    if vllm is not None:
        report["vllm"] = {"version": getattr(vllm, "__version__", "unknown")}
    else:
        report["vllm_import_error"] = vllm_err

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
