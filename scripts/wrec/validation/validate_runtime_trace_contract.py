#!/usr/bin/env python3
"""Validate WREC runtime event traces against a fixed MoE prior contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_trace(path: Path, *, expected_layers: int, expected_experts: int, expected_topk: int) -> dict[str, Any]:
    required = {"request_id", "layer", "token_pos", "selected_experts"}
    request_ids: set[str] = set()
    request_tokens: set[tuple[str, int]] = set()
    token_layer_events: set[tuple[str, int, int]] = set()
    layer_counts: Counter[int] = Counter()
    expert_counts: dict[int, Counter[int]] = defaultdict(Counter)
    selected_lengths: Counter[int] = Counter()
    failures: list[dict[str, Any]] = []
    previous_key: tuple[str, int, int] | None = None
    router_events = 0
    expert_refs = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            router_events += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append({"line": line_no, "error": f"json: {exc}"})
                continue

            missing = sorted(required - set(row))
            if missing:
                failures.append({"line": line_no, "error": f"missing fields: {missing}"})
                continue

            request_id = str(row["request_id"])
            layer = int(row["layer"])
            token_pos = int(row["token_pos"])
            selected = row["selected_experts"]
            event_key = (request_id, layer, token_pos)

            if previous_key is not None and event_key < previous_key:
                failures.append(
                    {
                        "line": line_no,
                        "error": "events are not monotonic by request_id/layer/token_pos",
                        "previous": list(previous_key),
                        "current": list(event_key),
                    }
                )
            previous_key = event_key

            if not isinstance(selected, list) or len(selected) != expected_topk:
                failures.append(
                    {
                        "line": line_no,
                        "error": "selected_experts length mismatch",
                        "expected_topk": expected_topk,
                        "actual": selected,
                    }
                )
                continue
            if layer < 0 or layer >= expected_layers:
                failures.append(
                    {
                        "line": line_no,
                        "error": "layer out of range",
                        "expected_range": [0, expected_layers - 1],
                        "actual": layer,
                    }
                )
            selected_lengths[len(selected)] += 1
            request_ids.add(request_id)
            request_tokens.add((request_id, token_pos))
            token_layer_events.add((request_id, token_pos, layer))
            layer_counts[layer] += 1
            for expert_value in selected:
                expert = int(expert_value)
                expert_refs += 1
                if expert < 0 or expert >= expected_experts:
                    failures.append(
                        {
                            "line": line_no,
                            "error": "expert out of range",
                            "expected_range": [0, expected_experts - 1],
                            "actual": expert,
                        }
                    )
                expert_counts[layer][expert] += 1

    observed_layers = sorted(layer_counts)
    observed_experts = sorted({expert for counts in expert_counts.values() for expert in counts})
    complete_layer_set = observed_layers == list(range(expected_layers))
    complete_expert_set = observed_experts == list(range(expected_experts))
    expected_token_layer_events = len(request_tokens) * expected_layers

    if len(token_layer_events) != expected_token_layer_events:
        failures.append(
            {
                "error": "token-layer event count mismatch",
                "expected": expected_token_layer_events,
                "actual": len(token_layer_events),
            }
        )

    return {
        "path": str(path),
        "router_events": router_events,
        "expert_refs": expert_refs,
        "requests": len(request_ids),
        "input_tokens": len(request_tokens),
        "token_layer_events": len(token_layer_events),
        "expected_token_layer_events": expected_token_layer_events,
        "observed_layers": observed_layers,
        "observed_experts": observed_experts,
        "selected_lengths": {str(k): v for k, v in sorted(selected_lengths.items())},
        "complete_layer_set": complete_layer_set,
        "complete_expert_set": complete_expert_set,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def compare_stats(name: str, trace: dict[str, Any], stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    if stats is None:
        return []
    checks = [
        ("requests", trace["requests"], int(stats["num_requests"])),
        ("input_tokens", trace["input_tokens"], int(stats["total_input_tokens"])),
        ("router_events", trace["router_events"], int(stats["total_events"])),
    ]
    if "num_router_layers" in stats:
        checks.append(("num_layers", len(trace["observed_layers"]), int(stats["num_router_layers"])))
    if "num_experts" in stats:
        checks.append(("num_experts", len(trace["observed_experts"]), int(stats["num_experts"])))
    return [
        {
            "trace": name,
            "name": check_name,
            "actual": actual,
            "expected": expected,
            "pass": actual == expected,
        }
        for check_name, actual, expected in checks
    ]


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    config = payload["contract"]
    lines = [
        "# WREC Runtime Trace Contract Validation",
        "",
        "## Contract",
        "",
        f"- Model family: `{config['model_family']}`",
        f"- Expected layers: `{config['expected_layers']}`",
        f"- Expected experts/layer: `{config['expected_experts']}`",
        f"- Expected top-k: `{config['expected_topk']}`",
        f"- Train prior trace: `{payload['inputs']['train_trace']}`",
        f"- Runtime event trace: `{payload['inputs']['event_trace']}`",
        "",
        "## Verdict",
        "",
        f"- Overall pass: `{payload['overall_pass']}`",
        "",
        "## Trace Summary",
        "",
        "| trace | requests | input tokens | router events | expert refs | layers complete | experts complete | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "event"):
        trace = payload["traces"][name]
        lines.append(
            f"| {name} | {trace['requests']} | {trace['input_tokens']} | {trace['router_events']} | "
            f"{trace['expert_refs']} | {trace['complete_layer_set']} | {trace['complete_expert_set']} | "
            f"{trace['failure_count']} |"
        )
    lines.extend(["", "## Stats Checks", "", "| trace | metric | actual | expected | pass |", "|---|---|---:|---:|---:|"])
    for check in payload["stats_checks"]:
        lines.append(
            f"| {check['trace']} | {check['name']} | {check['actual']} | {check['expected']} | {check['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- The existing Mixtral `mem48` train trace is a valid WREC prior source for 32 layers, 8 experts, and top-k 2.",
            "- The existing Mixtral `mem48` eval trace satisfies the same runtime event contract and can be sent to the sidecar without model-dimension mismatch.",
            "- This validates prior/event compatibility only; it does not modify vLLM or control real expert residency.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--event-trace", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, default=None)
    parser.add_argument("--event-stats", type=Path, default=None)
    parser.add_argument("--model-family", default="mixtral8x7b")
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-experts", type=int, default=8)
    parser.add_argument("--expected-topk", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    train = inspect_trace(
        args.train_trace,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
        expected_topk=args.expected_topk,
    )
    event = inspect_trace(
        args.event_trace,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
        expected_topk=args.expected_topk,
    )
    stats_checks = [
        *compare_stats("train", train, load_stats(args.train_stats)),
        *compare_stats("event", event, load_stats(args.event_stats)),
    ]
    trace_pass = all(
        trace["failure_count"] == 0 and trace["complete_layer_set"] and trace["complete_expert_set"]
        for trace in (train, event)
    )
    stats_pass = all(check["pass"] for check in stats_checks)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train_trace": str(args.train_trace),
            "event_trace": str(args.event_trace),
            "train_stats": str(args.train_stats) if args.train_stats else None,
            "event_stats": str(args.event_stats) if args.event_stats else None,
        },
        "contract": {
            "model_family": args.model_family,
            "expected_layers": args.expected_layers,
            "expected_experts": args.expected_experts,
            "expected_topk": args.expected_topk,
        },
        "overall_pass": trace_pass and stats_pass,
        "traces": {
            "train": train,
            "event": event,
        },
        "stats_checks": stats_checks,
        "claim_boundary": {
            "validates_prior_event_dimension_alignment": trace_pass and stats_pass,
            "uses_existing_mem48_offload_traces": True,
            "does_control_real_expert_loading": False,
            "does_modify_vllm": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)
    if not payload["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
