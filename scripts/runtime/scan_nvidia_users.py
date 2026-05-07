#!/usr/bin/env python3
"""扫描当前哪些进程持有 NVIDIA 设备文件。

这个脚本用 `/proc/<pid>/fd` 做一次 best-effort 检查，目标是替代某些环境里
可能不可用或输出不直观的 `fuser -v /dev/nvidia*`。它会遍历进程打开的文件
描述符，找出哪些进程仍持有 `/dev/nvidia*` 相关设备节点，并打印 PID、
comm 和命令行。

典型用途：
1. `nvidia-smi` 看起来很干净，但显存就是释放不掉。
2. 想确认是哪一个后台进程占住了 GPU 设备文件。
3. 在共享机器上排查双卡实验无法启动的原因。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_DEVICES = [
    "/dev/nvidia0",
    "/dev/nvidia1",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
]


def read_cmdline(proc_dir: Path) -> str:
    try:
        data = (proc_dir / "cmdline").read_bytes()
    except OSError:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def read_comm(proc_dir: Path) -> str:
    try:
        return (proc_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def scan_devices(device_paths: list[str]) -> dict[str, list[dict[str, str]]]:
    target_devices = {os.path.realpath(path): path for path in device_paths}
    results: dict[str, list[dict[str, str]]] = {path: [] for path in device_paths}

    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue

        fd_dir = proc_entry / "fd"
        if not fd_dir.exists():
            continue

        hits: set[str] = set()
        try:
            for fd_entry in fd_dir.iterdir():
                try:
                    target = os.path.realpath(fd_entry)
                except OSError:
                    continue
                if target in target_devices:
                    hits.add(target_devices[target])
        except OSError:
            continue

        if not hits:
            continue

        cmdline = read_cmdline(proc_entry)
        comm = read_comm(proc_entry)
        pid = proc_entry.name
        for device in sorted(hits):
            results[device].append(
                {
                    "pid": pid,
                    "comm": comm,
                    "cmdline": cmdline,
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "devices",
        nargs="*",
        default=DEFAULT_DEVICES,
        help="Device files to scan. Default: common NVIDIA device files.",
    )
    args = parser.parse_args()

    results = scan_devices(args.devices)
    for device in args.devices:
        print(f"== {device} ==")
        entries = results.get(device, [])
        if not entries:
            print("  no visible holders found")
            continue
        for row in entries:
            print(f"  PID {row['pid']} | comm={row['comm'] or '[unknown]'}")
            if row["cmdline"]:
                print(f"    cmdline: {row['cmdline']}")


if __name__ == "__main__":
    main()
