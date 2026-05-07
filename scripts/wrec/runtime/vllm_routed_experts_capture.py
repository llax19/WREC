#!/usr/bin/env python3
"""Capture vLLM routed experts and export them as WREC runtime events.

This script is the first internal vLLM integration step for WREC. It relies on
vLLM's built-in `enable_return_routed_experts` path, then converts the returned
`[seq_len, layer_num, topk]` array into the sidecar event format used by
`runtime_sidecar.py`.
"""

from __future__ import annotations

import argparse
import gc
import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_OUT_DIR = Path("/root/workspace/results/wrec/runtime_vllm")


def shutdown_llm(llm: Any) -> None:
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def post_json(conn: http.client.HTTPConnection, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    raw = response.read()
    if response.status >= 400:
        raise RuntimeError(raw.decode("utf-8", errors="replace"))
    return json.loads(raw.decode("utf-8"))


def get_json(conn: http.client.HTTPConnection, path: str) -> dict[str, Any]:
    conn.request("GET", path)
    response = conn.getresponse()
    raw = response.read()
    if response.status >= 400:
        raise RuntimeError(raw.decode("utf-8", errors="replace"))
    return json.loads(raw.decode("utf-8"))


def iter_wrec_events(
    *,
    request_id: str,
    routed_experts: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import numpy as np

    array = np.asarray(routed_experts)
    if array.ndim != 3:
        raise ValueError(
            "Expected routed_experts with shape [seq_len, layer_num, topk], "
            f"got {array.shape}"
        )

    seq_len, layer_num, topk = (int(dim) for dim in array.shape)
    events: list[dict[str, Any]] = []
    event_index = 0
    for token_pos in range(seq_len):
        for layer in range(layer_num):
            selected = [int(expert) for expert in array[token_pos, layer].tolist() if int(expert) >= 0]
            if not selected:
                continue
            events.append(
                {
                    "request_id": request_id,
                    "event_index": event_index,
                    "layer": layer,
                    "token_pos": token_pos,
                    "selected_experts": selected,
                }
            )
            event_index += 1

    return events, {
        "seq_len": seq_len,
        "layer_num": layer_num,
        "topk": topk,
        "router_events": len(events),
        "expert_refs": sum(len(event["selected_experts"]) for event in events),
    }


def maybe_send_to_sidecar(
    *,
    sidecar_url: str | None,
    events: list[dict[str, Any]],
    max_events: int,
) -> dict[str, Any] | None:
    if not sidecar_url:
        return None

    parsed = urlparse(sidecar_url)
    conn = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=60)
    health = get_json(conn, "/health")
    sent = 0
    expert_refs = 0
    last_response: dict[str, Any] | None = None
    start = time.perf_counter()
    try:
        for event in events:
            if max_events > 0 and sent >= max_events:
                break
            last_response = post_json(conn, "/event", event)
            sent += 1
            expert_refs += len(event["selected_experts"])
        metrics = get_json(conn, "/metrics")
    finally:
        conn.close()

    elapsed = time.perf_counter() - start
    return {
        "health": health,
        "events_sent": sent,
        "expert_refs_sent": expert_refs,
        "elapsed_seconds": elapsed,
        "us_per_event": elapsed / max(1, sent) * 1e6,
        "last_response": last_response,
        "metrics": metrics,
    }


def write_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    routed = payload["routed_experts"]
    generation = payload["generation"]
    sidecar = payload.get("sidecar")
    lines = [
        "# vLLM Routed Experts Capture",
        "",
        "## Inputs",
        "",
        f"- Model: `{payload['inputs']['model']}`",
        f"- Prompt: `{payload['inputs']['prompt']}`",
        f"- Max tokens: `{payload['inputs']['max_tokens']}`",
        f"- Max model len: `{payload['inputs']['max_model_len']}`",
        "",
        "## vLLM Output",
        "",
        f"- Finish reason: `{generation['finish_reason']}`",
        f"- Generated token count: `{generation['generated_token_count']}`",
        f"- Generated text: `{generation['generated_text']}`",
        "",
        "## Routed Experts",
        "",
        f"- Shape: `{routed['shape']}`",
        f"- Seq len: `{routed['seq_len']}`",
        f"- Layers: `{routed['layer_num']}`",
        f"- Top-k: `{routed['topk']}`",
        f"- Router events exported: `{routed['router_events']}`",
        f"- Expert refs exported: `{routed['expert_refs']}`",
        f"- Events JSONL: `{payload['outputs']['events_jsonl']}`",
    ]
    if sidecar is not None:
        lines.extend(
            [
                "",
                "## Sidecar",
                "",
                f"- Events sent: `{sidecar['events_sent']}`",
                f"- Expert refs sent: `{sidecar['expert_refs_sent']}`",
                f"- Submit time: `{sidecar['us_per_event']:.3f}` us/event",
                f"- Sidecar expert refs processed: `{sidecar['metrics']['expert_refs']}`",
                f"- Sidecar shadow misses: `{sidecar['metrics']['shadow_misses']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- vLLM internal routed expert capture is available through `enable_return_routed_experts=True`.",
            "- The captured routed experts were converted into WREC runtime sidecar event format.",
            "- This verifies the internal event-source path; it does not yet control vLLM expert residency or loading.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/workspace/qwen1.5-MoE-A2.7B"))
    parser.add_argument("--prompt", default="请用一句话解释什么是混合专家模型。")
    parser.add_argument("--request-id", default="vllm-qwen-smoke-000000")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--sidecar-url", default=None)
    parser.add_argument("--sidecar-max-events", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUT_DIR / "vllm_routed_experts_qwen_smoke_20260505.json",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        default=DEFAULT_OUT_DIR / "vllm_routed_experts_qwen_events_20260505.jsonl",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUT_DIR / "vllm_routed_experts_qwen_smoke_20260505.md",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import numpy as np
    import torch

    llm = None
    try:
        load_start = time.perf_counter()
        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=args.cpu_offload_gb,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            enable_return_routed_experts=True,
        )
        load_seconds = time.perf_counter() - load_start
        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        generate_start = time.perf_counter()
        outputs = llm.generate([args.prompt], sampling_params)
        generate_seconds = time.perf_counter() - generate_start

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion outputs")
        completion = outputs[0].outputs[0]
        routed_experts = getattr(completion, "routed_experts", None)
        if routed_experts is None:
            raise RuntimeError("CompletionOutput.routed_experts is None")

        routed_array = np.asarray(routed_experts)
        events, event_stats = iter_wrec_events(
            request_id=args.request_id,
            routed_experts=routed_array,
        )
        write_events_jsonl(args.events_jsonl, events)
        sidecar_result = maybe_send_to_sidecar(
            sidecar_url=args.sidecar_url,
            events=events,
            max_events=args.sidecar_max_events,
        )

        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "model": str(args.model),
                "prompt": args.prompt,
                "request_id": args.request_id,
                "max_tokens": args.max_tokens,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "cpu_offload_gb": args.cpu_offload_gb,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "trust_remote_code": args.trust_remote_code,
                "enforce_eager": args.enforce_eager,
                "sidecar_url": args.sidecar_url,
            },
            "timing": {
                "load_seconds": load_seconds,
                "generate_seconds": generate_seconds,
            },
            "generation": {
                "generated_text": completion.text,
                "generated_token_count": len(completion.token_ids),
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            },
            "routed_experts": {
                "shape": list(routed_array.shape),
                **event_stats,
            },
            "outputs": {
                "json": str(args.output_json),
                "events_jsonl": str(args.events_jsonl),
                "markdown": str(args.output_md),
            },
            "sidecar": sidecar_result,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(args.output_md, payload)
        print(args.output_json)
        print(args.events_jsonl)
        print(args.output_md)
    finally:
        shutdown_llm(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
