#!/usr/bin/env python3
"""Build token/layer router event traces for MoE causal language models.

The first target is Qwen1.5-MoE-A2.7B Phase 2A tracing. The CLI also keeps the
offload-related arguments needed for the later Mixtral feasibility probe.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "prompt"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing required key: {key}")
            rows.append(row)
    return rows


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    if not text:
        return None
    values: dict[int | str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--max-memory entries must look like 0=22GiB,cpu=96GiB")
        key, value = item.split("=", 1)
        key = key.strip()
        values["cpu" if key == "cpu" else int(key)] = value.strip()
    return values


def input_device_for(model: torch.nn.Module) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten_router_logits(layer_logits: torch.Tensor, input_tokens: int, layer_idx: int) -> torch.Tensor:
    logits = layer_logits.detach().float().cpu()
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if logits.ndim != 2:
        raise ValueError(f"layer {layer_idx} router logits shape is {tuple(logits.shape)}")
    if logits.shape[0] < input_tokens:
        raise ValueError(
            f"layer {layer_idx} has {logits.shape[0]} router rows for {input_tokens} input tokens"
        )
    if logits.shape[0] > input_tokens:
        logits = logits[:input_tokens]
    return logits


def router_probabilities(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    row_sums = scores.sum(dim=-1)
    looks_like_probs = (
        float(scores.min().item()) >= 0.0
        and float(row_sums.min().item()) > 0.99
        and float(row_sums.max().item()) < 1.01
    )
    if looks_like_probs:
        return scores
    return torch.softmax(scores, dim=-1)


def dtype_from_name(name: str) -> Any:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def safe_model_name(model_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return model_path.name


def append_router_events(
    *,
    events: list[dict[str, Any]],
    router_logits: Any,
    routed_tokens: int,
    request_id: str,
    model_name: str,
    phase: str,
    step: int,
    token_pos_offset: int,
    router_top_k: int,
    layer_event_counts: Counter[int],
    layer_expert_counts: dict[int, Counter[int]],
    generated_token_id: int | None = None,
) -> int:
    if router_logits is None:
        raise RuntimeError("model did not return router_logits")

    request_events = 0
    for layer_idx, layer_logits in enumerate(router_logits):
        logits = flatten_router_logits(layer_logits, routed_tokens, layer_idx)
        probs = router_probabilities(logits)
        k = min(router_top_k, probs.shape[-1])
        values, indices = torch.topk(probs, k=k, dim=-1)
        for local_token_pos, (token_values, token_indices) in enumerate(zip(values, indices)):
            selected = [int(expert) for expert in token_indices.tolist()]
            expert_probs = [round(float(value), 8) for value in token_values.tolist()]
            event = {
                "request_id": request_id,
                "model": model_name,
                "phase": phase,
                "step": step,
                "layer": layer_idx,
                "token_pos": token_pos_offset + local_token_pos,
                "selected_experts": selected,
                "expert_probs": expert_probs,
                "num_routed_tokens": 1,
            }
            if generated_token_id is not None:
                event["generated_token_id"] = int(generated_token_id)
            events.append(event)
            request_events += 1
            layer_event_counts[layer_idx] += 1
            for expert in selected:
                layer_expert_counts[layer_idx][expert] += 1
    return request_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--failure-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument(
        "--phase",
        choices=("prefill", "prefill_decode", "decode"),
        default="prefill",
        help=(
            "`prefill` writes prompt router events only. `prefill_decode` writes "
            "prompt events at step 0 plus generated-token decode events at steps "
            "1..max_new_tokens. `decode` runs prefill to build KV cache but writes "
            "only decode events."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument(
        "--stop-on-eos",
        action="store_true",
        help="Stop decode tracing early when the greedy token is EOS.",
    )
    parser.add_argument("--router-top-k", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record request failures and continue instead of failing fast.",
    )
    parser.add_argument(
        "--disable-cuda-allocator-warmup",
        action="store_true",
        help=(
            "Disable Transformers' device-map CUDA caching allocator warmup. "
            "This is slower but can avoid large temporary allocations on 24GB GPUs."
        ),
    )
    args = parser.parse_args()

    if args.phase in {"prefill_decode", "decode"} and args.max_new_tokens <= 0:
        raise ValueError("--phase prefill_decode/decode requires --max-new-tokens > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.disable_cuda_allocator_warmup:
        import transformers.modeling_utils as modeling_utils

        def _skip_caching_allocator_warmup(*_unused_args: Any, **_unused_kwargs: Any) -> None:
            return None

        modeling_utils.caching_allocator_warmup = _skip_caching_allocator_warmup

    requests = load_requests(args.request_file)
    if args.offset:
        requests = requests[args.offset :]
    if args.limit is not None:
        requests = requests[: args.limit]
    if not requests:
        raise ValueError("no requests selected")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_name(args.dtype),
        "device_map": args.device_map,
        "max_memory": parse_max_memory(args.max_memory),
        "low_cpu_mem_usage": True,
    }
    if args.offload_folder is not None:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(args.offload_folder)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    config = model.config
    num_experts = int(getattr(config, "num_experts", getattr(config, "num_local_experts", 0)))
    num_experts_per_tok = int(getattr(config, "num_experts_per_tok"))
    router_top_k = int(args.router_top_k or num_experts_per_tok)
    model_name = safe_model_name(args.model_path, args.model_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    if args.failure_output:
        args.failure_output.parent.mkdir(parents=True, exist_ok=True)

    total_events = 0
    total_input_tokens = 0
    total_decode_tokens = 0
    request_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    layer_event_counts: Counter[int] = Counter()
    layer_expert_counts: dict[int, Counter[int]] = defaultdict(Counter)
    started = time.perf_counter()
    device = input_device_for(model)

    with args.output.open("w", encoding="utf-8") as out_f:
        for index, row in enumerate(requests, start=1):
            request_id = str(row.get("request_id", row["id"]))
            request_start = time.perf_counter()
            try:
                request_event_rows: list[dict[str, Any]] = []
                request_layer_event_counts: Counter[int] = Counter()
                request_layer_expert_counts: dict[int, Counter[int]] = defaultdict(Counter)
                encoded = tokenizer(
                    str(row["prompt"]),
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                input_tokens = int(encoded["input_ids"].shape[-1])
                encoded = {key: value.to(device) for key, value in encoded.items()}

                with torch.inference_mode():
                    outputs = model(
                        **encoded,
                        use_cache=args.phase != "prefill",
                        output_router_logits=True,
                        output_hidden_states=False,
                        output_attentions=False,
                        logits_to_keep=1,
                    )
                router_logits = getattr(outputs, "router_logits", None)

                request_events = 0
                if args.phase in {"prefill", "prefill_decode"}:
                    request_events += append_router_events(
                        events=request_event_rows,
                        router_logits=router_logits,
                        routed_tokens=input_tokens,
                        request_id=request_id,
                        model_name=model_name,
                        phase="prefill",
                        step=0,
                        token_pos_offset=0,
                        router_top_k=router_top_k,
                        layer_event_counts=request_layer_event_counts,
                        layer_expert_counts=request_layer_expert_counts,
                    )

                decode_tokens = 0
                if args.phase in {"prefill_decode", "decode"}:
                    past_key_values = getattr(outputs, "past_key_values", None)
                    if past_key_values is None:
                        raise RuntimeError("model did not return past_key_values for decode tracing")
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        raise RuntimeError("model did not return logits for greedy decode tracing")
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    for step in range(1, args.max_new_tokens + 1):
                        if args.stop_on_eos and tokenizer.eos_token_id is not None:
                            if bool((next_token == int(tokenizer.eos_token_id)).all().item()):
                                break

                        with torch.inference_mode():
                            decode_outputs = model(
                                input_ids=next_token.to(device),
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_router_logits=True,
                                output_hidden_states=False,
                                output_attentions=False,
                                logits_to_keep=1,
                            )
                        generated_token_id = int(next_token[0, -1].detach().cpu().item())
                        request_events += append_router_events(
                            events=request_event_rows,
                            router_logits=getattr(decode_outputs, "router_logits", None),
                            routed_tokens=1,
                            request_id=request_id,
                            model_name=model_name,
                            phase="decode",
                            step=step,
                            token_pos_offset=input_tokens + step - 1,
                            router_top_k=router_top_k,
                            layer_event_counts=request_layer_event_counts,
                            layer_expert_counts=request_layer_expert_counts,
                            generated_token_id=generated_token_id,
                        )
                        decode_tokens += 1
                        past_key_values = getattr(decode_outputs, "past_key_values", None)
                        if past_key_values is None:
                            raise RuntimeError("model did not return past_key_values during decode")
                        logits = getattr(decode_outputs, "logits", None)
                        if logits is None:
                            raise RuntimeError("model did not return logits during decode")
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                for event in request_event_rows:
                    out_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                for layer, count in request_layer_event_counts.items():
                    layer_event_counts[layer] += count
                for layer, expert_counter in request_layer_expert_counts.items():
                    layer_expert_counts[layer].update(expert_counter)
                total_events += request_events
                total_input_tokens += input_tokens
                total_decode_tokens += decode_tokens
                elapsed = time.perf_counter() - request_start
                request_summaries.append(
                    {
                        "request_id": request_id,
                        "input_tokens": input_tokens,
                        "decode_tokens": decode_tokens,
                        "events": request_events,
                        "trace_seconds": elapsed,
                    }
                )
                print(
                    f"[{index}/{len(requests)}] {request_id} tokens={input_tokens} "
                    f"decode_tokens={decode_tokens} events={request_events} seconds={elapsed:.2f}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                if not args.continue_on_error:
                    raise
                print(f"[{index}/{len(requests)}] {request_id} failed: {exc}", flush=True)

    if args.failure_output:
        with args.failure_output.open("w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.stats_output:
        stats = {
            "model": model_name,
            "model_path": str(args.model_path),
            "request_file": str(args.request_file),
            "output": str(args.output),
            "phase": args.phase,
            "num_requests": len(request_summaries),
            "num_failures": len(failures),
            "total_input_tokens": total_input_tokens,
            "total_decode_tokens": total_decode_tokens,
            "max_new_tokens": args.max_new_tokens,
            "stop_on_eos": bool(args.stop_on_eos),
            "total_events": total_events,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "router_top_k": router_top_k,
            "num_router_layers": len(layer_event_counts),
            "elapsed_seconds": time.perf_counter() - started,
            "request_summaries": request_summaries,
            "layer_event_counts": dict(sorted(layer_event_counts.items())),
            "top_experts_by_layer": {
                str(layer): counter.most_common(8)
                for layer, counter in sorted(layer_expert_counts.items())
            },
            "failures": failures,
        }
        args.stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
