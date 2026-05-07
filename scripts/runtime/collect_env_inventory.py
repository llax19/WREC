#!/usr/bin/env python3
"""Collect a lightweight environment inventory for WREC experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PYTHON_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "pandas",
    "tqdm",
]


def run_command(args: list[str]) -> dict[str, object]:
    if shutil.which(args[0]) is None:
        return {"available": False, "command": args, "stdout": "", "stderr": ""}
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "available": True,
        "command": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def module_status(name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    status: dict[str, object] = {"installed": spec is not None}
    if spec is None:
        return status
    try:
        module = __import__(name)
        status["version"] = getattr(module, "__version__", None)
    except Exception as exc:  # pragma: no cover - inventory should not fail here
        status["import_error"] = repr(exc)
    return status


def path_status(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    status: dict[str, object] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.exists():
        status["is_dir"] = resolved.is_dir()
        status["size_bytes"] = resolved.stat().st_size if resolved.is_file() else None
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wrec/env_inventory_20260427.json"),
        help="Path to write inventory JSON.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace root used for disk and path checks.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    disk = shutil.disk_usage(workspace)

    inventory = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "platform": {
            "python": sys.version,
            "python_executable": sys.executable,
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        },
        "python_modules": {name: module_status(name) for name in PYTHON_MODULES},
        "commands": {
            "nvidia_smi_query": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "nvidia_smi_topology": run_command(["nvidia-smi", "topo", "-m"]),
            "nvcc": run_command(["nvcc", "--version"]),
            "huggingface_cli": run_command(["huggingface-cli", "--version"]),
            "git_lfs": run_command(["git-lfs", "--version"]),
        },
        "paths": {
            "qwen_debug_model": path_status(workspace / "qwen1.5-MoE-A2.7B" / "config.json"),
            "mixtral_config": path_status(
                workspace / "models" / "Mixtral-8x7B-Instruct-v0.1" / "config.json"
            ),
            "debug_requests": path_status(workspace / "data" / "prompts" / "debug_requests.jsonl"),
            "eval_requests": path_status(workspace / "data" / "prompts" / "eval_requests.jsonl"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
