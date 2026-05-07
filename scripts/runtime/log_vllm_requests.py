#!/usr/bin/env python3
"""向 vLLM 服务发送请求，并记录逐请求时延与 GPU 摘要。

这是实验链路里的核心采集脚本。它会读取一个请求清单 JSONL，按给定并发
和到达间隔向本地 vLLM OpenAI 兼容接口发请求，记录每条请求的提交时间、
首 token 时间、完成时间、输入长度和输出长度，并最终写出原始请求日志。

注意：`--dispatch-strategy length_only` 只是客户端提交前重排请求顺序，用于
代理实验和链路验证；服务端内嵌调度实验应使用 `--strategy length_only`
记录结果标签，同时使用 `--dispatch-strategy fcfs` 保持客户端按原始顺序发请求。

如果同时提供 GPU 监控日志路径，它还会在请求窗口结束后汇总对应时间段内
的显存使用情况，生成一个简短的 GPU 摘要文件，便于把时延结果和资源使用
放在同一轮实验里对照分析。

输入：
- 请求清单 JSONL。
- 运行中的 vLLM 服务地址。
- 可选 tokenizer 路径和 GPU 监控日志。

输出：
- 请求级原始日志 `logs/raw/*.jsonl`。
- 可选 GPU 摘要 `logs/processed/*_gpu_summary.json`。
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_requests(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "prompt", "target_max_new_tokens"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing required key: {key}")
            rows.append(row)
    return rows


def maybe_build_tokenizer(tokenizer_path: str | None):
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_input_tokens(prompt: str, tokenizer) -> int:
    if tokenizer is None:
        return len(prompt)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def count_output_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def normalize_strategy_name(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {
        "fcfs": "fcfs",
        "ltr": "ltr",
        "moe_affinity": "moe_affinity",
        "moe-affinity": "moe_affinity",
        "length": "length_only",
        "length_only": "length_only",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported strategy: {strategy}")
    return aliases[normalized]


def prepare_requests(rows: list[dict], strategy: str, tokenizer) -> list[dict]:
    prepared = []
    for original_index, row in enumerate(rows):
        input_tokens = count_input_tokens(row["prompt"], tokenizer)
        estimated_total_tokens = input_tokens + int(row["target_max_new_tokens"])
        prepared_row = dict(row)
        prepared_row["_original_index"] = original_index
        prepared_row["_planned_input_tokens"] = input_tokens
        prepared_row["_estimated_total_tokens"] = estimated_total_tokens
        prepared.append(prepared_row)

    if strategy == "length_only":
        # Use prompt length plus target decode budget as a lightweight proxy for
        # total request cost, while preserving FCFS order for ties.
        prepared.sort(
            key=lambda row: (
                int(row["_estimated_total_tokens"]),
                int(row["_planned_input_tokens"]),
                int(row["_original_index"]),
            )
        )

    for dispatch_index, row in enumerate(prepared):
        row["_dispatch_index"] = dispatch_index
    return prepared


def load_gpu_samples(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = {"timestamp", "gpu_index", "memory_used_mib", "memory_total_mib"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def build_gpu_summary(samples: list[dict], window_start: float, window_end: float) -> dict:
    in_window = [row for row in samples if window_start <= row["timestamp"] <= window_end]
    if not in_window:
        in_window = samples

    grouped: dict[int, list[dict]] = {}
    for row in in_window:
        grouped.setdefault(int(row["gpu_index"]), []).append(row)

    per_gpu = []
    for gpu_index, bucket in sorted(grouped.items()):
        used_values = [float(row["memory_used_mib"]) for row in bucket]
        total_values = [float(row["memory_total_mib"]) for row in bucket]
        total_mib = max(total_values) if total_values else 0.0
        peak_mib = max(used_values) if used_values else 0.0
        avg_mib = sum(used_values) / len(used_values) if used_values else 0.0
        peak_ratio = (peak_mib / total_mib) if total_mib else 0.0
        per_gpu.append(
            {
                "gpu_index": gpu_index,
                "sample_count": len(bucket),
                "memory_total_mib": total_mib,
                "peak_memory_used_mib": peak_mib,
                "avg_memory_used_mib": avg_mib,
                "peak_memory_utilization": peak_ratio,
            }
        )

    return {
        "window_start": window_start,
        "window_end": window_end,
        "sample_count": len(in_window),
        "per_gpu": per_gpu,
    }


def post_streaming_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "request_id": request_id,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )

    submit_time = time.time()
    start_time = submit_time
    first_token_time = None
    final_text = ""
    finish_reason = None

    try:
        with HTTP_OPENER.open(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                text = choice.get("text", "")
                if text and first_token_time is None:
                    first_token_time = time.time()
                final_text += text
                finish_reason = choice.get("finish_reason")
    except urllib.error.HTTPError as exc:  # pragma: no cover - runtime diagnostic
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc

    finish_time = time.time()
    if first_token_time is None:
        first_token_time = finish_time

    return {
        "submit_time": submit_time,
        "start_time": start_time,
        "first_token_time": first_token_time,
        "finish_time": finish_time,
        "generated_text": final_text,
        "finish_reason": finish_reason,
    }


def build_log_row(
    row: dict,
    *,
    base_url: str,
    model_name: str,
    strategy: str,
    dispatch_strategy: str,
    server_scheduling_policy: str,
    load_level: str,
    tokenizer,
) -> dict:
    input_tokens = int(row.get("_planned_input_tokens", count_input_tokens(row["prompt"], tokenizer)))
    result = post_streaming_completion(
        base_url=base_url,
        model=model_name,
        prompt=row["prompt"],
        max_tokens=int(row["target_max_new_tokens"]),
        request_id=str(row["id"]),
    )
    return {
        "request_id": row["id"],
        "strategy": strategy,
        "dispatch_strategy": dispatch_strategy,
        "server_scheduling_policy": server_scheduling_policy,
        "load_level": load_level,
        "submit_time": result["submit_time"],
        "start_time": result["start_time"],
        "first_token_time": result["first_token_time"],
        "finish_time": result["finish_time"],
        "input_tokens": input_tokens,
        "output_tokens": count_output_tokens(result["generated_text"], tokenizer),
        "target_max_new_tokens": int(row["target_max_new_tokens"]),
        "estimated_total_tokens": int(row.get("_estimated_total_tokens", input_tokens + int(row["target_max_new_tokens"]))),
        "group": row.get("group", "unknown"),
        "original_request_index": int(row.get("_original_index", -1)),
        "dispatch_index": int(row.get("_dispatch_index", -1)),
        "finish_reason": result["finish_reason"],
        "generated_text": result["generated_text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True, help="Prompt manifest in JSONL format")
    parser.add_argument("--output-log", type=Path, required=True, help="Where to save request-level logs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server base URL")
    parser.add_argument("--model-name", required=True, help="Model name passed to the OpenAI-compatible endpoint")
    parser.add_argument(
        "--strategy",
        default="fcfs",
        help="Strategy label written into the output log.",
    )
    parser.add_argument(
        "--dispatch-strategy",
        default=None,
        help=(
            "Client-side request dispatch order. Defaults to --strategy for "
            "backward compatibility. Use fcfs for server-side scheduler tests."
        ),
    )
    parser.add_argument(
        "--server-scheduling-policy",
        default="fcfs",
        help="vLLM server-side scheduling policy label written into the output log.",
    )
    parser.add_argument("--load-level", default="debug", help="Load label written into the output log")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path used to compute input token lengths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of prompts to send",
    )
    parser.add_argument(
        "--gpu-metrics-log",
        type=Path,
        default=None,
        help="Optional JSONL file with sampled GPU metrics",
    )
    parser.add_argument(
        "--gpu-metrics-summary-output",
        type=Path,
        default=None,
        help="Optional JSON path used to save a summary derived from --gpu-metrics-log",
    )
    parser.add_argument(
        "--arrival-interval",
        type=float,
        default=0.0,
        help="Seconds between request submissions. Use 0 for burst submission.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of client requests in flight at once.",
    )
    args = parser.parse_args()
    if args.arrival_interval < 0:
        raise ValueError("--arrival-interval must be non-negative")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    strategy_name = normalize_strategy_name(args.strategy)
    dispatch_strategy_name = normalize_strategy_name(
        args.dispatch_strategy if args.dispatch_strategy is not None else args.strategy
    )
    server_scheduling_policy_name = args.server_scheduling_policy.strip().lower().replace("-", "_")
    requests = load_requests(args.request_file)
    if args.limit is not None:
        requests = requests[: args.limit]

    tokenizer = maybe_build_tokenizer(args.tokenizer_path)
    requests = prepare_requests(requests, dispatch_strategy_name, tokenizer)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"prepared {len(requests)} requests with strategy={strategy_name} "
        f"dispatch_strategy={dispatch_strategy_name} "
        f"server_scheduling_policy={server_scheduling_policy_name} "
        f"(arrival_interval={args.arrival_interval}, max_concurrency={args.max_concurrency})"
    )

    run_start = None
    run_end = None
    log_rows: list[dict] = []

    def collect_results(done_futures: set[Future]) -> None:
        nonlocal run_start, run_end
        for future in done_futures:
            log_row = future.result()
            if run_start is None or log_row["submit_time"] < run_start:
                run_start = log_row["submit_time"]
            if run_end is None or log_row["finish_time"] > run_end:
                run_end = log_row["finish_time"]
            log_rows.append(log_row)
            print(
                f"logged {log_row['request_id']} "
                f"(strategy={log_row['strategy']}, load={log_row['load_level']})"
            )

    pending: set[Future] = set()
    next_release = time.time()

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        for row in requests:
            if args.arrival_interval > 0:
                delay = next_release - time.time()
                if delay > 0:
                    time.sleep(delay)
                next_release += args.arrival_interval

            while len(pending) >= args.max_concurrency:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect_results(done)

            future = executor.submit(
                build_log_row,
                row,
                base_url=args.base_url,
                model_name=args.model_name,
                strategy=strategy_name,
                dispatch_strategy=dispatch_strategy_name,
                server_scheduling_policy=server_scheduling_policy_name,
                load_level=args.load_level,
                tokenizer=tokenizer,
            )
            pending.add(future)

            done_now = {item for item in pending if item.done()}
            if done_now:
                pending -= done_now
                collect_results(done_now)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect_results(done)

    log_rows.sort(key=lambda row: row["submit_time"])
    with args.output_log.open("w", encoding="utf-8") as fout:
        for log_row in log_rows:
            fout.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            fout.flush()

    if args.gpu_metrics_summary_output is not None:
        if args.gpu_metrics_log is None:
            raise ValueError("--gpu-metrics-summary-output requires --gpu-metrics-log")
        if run_start is None or run_end is None:
            raise ValueError("No requests were logged, cannot build GPU summary")

        samples = load_gpu_samples(args.gpu_metrics_log)
        summary = build_gpu_summary(samples, run_start, run_end)
        args.gpu_metrics_summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.gpu_metrics_summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved gpu summary to {args.gpu_metrics_summary_output}")


if __name__ == "__main__":
    main()
