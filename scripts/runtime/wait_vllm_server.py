#!/usr/bin/env python3
"""轮询等待 vLLM 服务就绪。

这个脚本会持续探测目标服务的 `/health` 和 `/v1/models` 接口，直到其中一个
能够访问，或者超时退出。可选地，它还能同时监控服务进程 PID：如果进程在
服务 ready 之前就已经退出，就会立刻失败，避免主实验脚本继续往下执行。

典型用途：
1. 给单卡或双卡启动脚本提供“服务已可用”的同步点。
2. 在后台启动 `vllm serve` 后，避免靠固定 sleep 猜测初始化时间。
3. 在服务异常崩溃时尽早失败，而不是让后续请求脚本卡住。
"""

from __future__ import annotations

import argparse
import http.client
import os
import time
import urllib.error
import urllib.request


HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def url_ok(url: str) -> bool:
    try:
        with HTTP_OPENER.open(url, timeout=5) as response:
            return 200 <= getattr(response, "status", 200) < 500
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional server PID. If provided, fail early when the process exits.",
    )
    args = parser.parse_args()

    started = time.time()
    health_url = f"{args.base_url.rstrip('/')}/health"
    models_url = f"{args.base_url.rstrip('/')}/v1/models"

    while time.time() - started <= args.timeout:
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except OSError as exc:
                raise SystemExit(
                    f"vLLM server process exited before becoming ready (pid={args.pid})"
                ) from exc
        if url_ok(health_url) or url_ok(models_url):
            print(f"server ready: {args.base_url}")
            return
        time.sleep(max(args.interval, 0.2))

    raise SystemExit(f"Timed out waiting for vLLM server at {args.base_url}")


if __name__ == "__main__":
    main()
