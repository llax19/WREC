#!/usr/bin/env python3
"""Replay MoE affinity scheduling on sparse expert signatures.

This is a CPU-only locality simulator. It compares request orders produced by
FCFS, length-only, random, and an affinity reranker, then reports proxy metrics
that should correlate with expert-cache friendliness:

* average pairwise cosine similarity inside an admitted batch;
* average unique experts touched per layer per batch;
* per-layer LRU expert-cache hit/miss rate over admitted batches.

The simulator validates whether true route signatures contain exploitable
locality. It does not measure vLLM latency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    original_index: int
    prompt_tokens: int
    target_tokens: int
    total_tokens: int
    num_layers: int
    num_experts: int
    signature: dict[int, float]
    active_by_layer: tuple[frozenset[int], ...]
    norm: float


def vector_index(layer: int, expert: int, num_experts: int) -> int:
    return layer * num_experts + expert


def cosine(a: RequestTrace | dict[int, float], b: RequestTrace | dict[int, float]) -> float:
    if isinstance(a, RequestTrace):
        a_sig = a.signature
        a_norm = a.norm
    else:
        a_sig = a
        a_norm = math.sqrt(sum(value * value for value in a_sig.values()))

    if isinstance(b, RequestTrace):
        b_sig = b.signature
        b_norm = b.norm
    else:
        b_sig = b
        b_norm = math.sqrt(sum(value * value for value in b_sig.values()))

    if a_norm <= 0 or b_norm <= 0:
        return 0.0
    if len(a_sig) > len(b_sig):
        a_sig, b_sig = b_sig, a_sig
    dot = sum(value * b_sig.get(key, 0.0) for key, value in a_sig.items())
    return dot / (a_norm * b_norm)


def load_traces(path: Path, limit: int | None) -> list[RequestTrace]:
    traces: list[RequestTrace] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            experts = row.get("experts")
            if not isinstance(experts, list):
                raise ValueError(f"{path}:{line_no} missing experts list")

            num_layers = int(row.get("num_layers", 0))
            num_experts = int(row.get("num_experts", 0))
            if num_layers <= 0 or num_experts <= 0:
                max_layer = max(int(item[0]) for item in experts) if experts else -1
                max_expert = max(int(item[1]) for item in experts) if experts else -1
                num_layers = max(num_layers, max_layer + 1)
                num_experts = max(num_experts, max_expert + 1)

            signature: dict[int, float] = {}
            active_sets = [set() for _ in range(num_layers)]
            for item in experts:
                layer = int(item[0])
                expert = int(item[1])
                weight = float(item[2])
                if layer < 0 or expert < 0 or weight <= 0:
                    continue
                if layer >= num_layers:
                    continue
                key = vector_index(layer, expert, num_experts)
                signature[key] = signature.get(key, 0.0) + weight
                active_sets[layer].add(expert)

            prompt_tokens = int(row.get("input_tokens", row.get("prompt_tokens", 0)))
            target_tokens = int(
                row.get(
                    "target_tokens",
                    row.get("predicted_tokens", row.get("target_max_new_tokens", 0)),
                )
            )
            total_tokens = int(
                row.get(
                    "estimated_total_tokens",
                    prompt_tokens + target_tokens,
                )
            )
            norm = math.sqrt(sum(value * value for value in signature.values()))
            traces.append(
                RequestTrace(
                    request_id=str(row.get("request_id", f"trace-{line_no:06d}")),
                    original_index=len(traces),
                    prompt_tokens=prompt_tokens,
                    target_tokens=target_tokens,
                    total_tokens=total_tokens,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    signature=signature,
                    active_by_layer=tuple(frozenset(items) for items in active_sets),
                    norm=norm,
                )
            )
            if limit is not None and len(traces) >= limit:
                break
    if not traces:
        raise ValueError(f"no traces loaded from {path}")
    return traces


def aggregate_signature(requests: Iterable[RequestTrace]) -> dict[int, float]:
    aggregate: dict[int, float] = {}
    for request in requests:
        for key, value in request.signature.items():
            aggregate[key] = aggregate.get(key, 0.0) + value
    return aggregate


def base_sort_key(request: RequestTrace, base_policy: str) -> tuple[float, int, int, int]:
    if base_policy == "length":
        return (
            float(request.total_tokens),
            request.prompt_tokens,
            request.target_tokens,
            request.original_index,
        )
    if base_policy == "fcfs":
        return (0.0, 0, 0, request.original_index)
    raise ValueError(f"unknown base policy: {base_policy}")


def length_bucket(request: RequestTrace, bucket_size: int) -> int:
    if bucket_size <= 0:
        return 0
    return request.total_tokens // bucket_size


def schedule_in_batches(
    traces: list[RequestTrace],
    *,
    policy: str,
    batch_size: int,
    affinity_topk: int,
    base_policy: str,
    length_bucket_size: int,
    random_seed: int,
) -> list[list[RequestTrace]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if policy == "fcfs":
        ordered = sorted(traces, key=lambda req: req.original_index)
        return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]

    if policy == "length":
        ordered = sorted(traces, key=lambda req: base_sort_key(req, "length"))
        return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]

    if policy == "random":
        ordered = list(traces)
        random.Random(random_seed).shuffle(ordered)
        return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]

    if policy != "affinity":
        raise ValueError(f"unknown policy: {policy}")

    waiting = sorted(traces, key=lambda req: base_sort_key(req, base_policy))
    batches: list[list[RequestTrace]] = []
    topk = max(1, affinity_topk)
    while waiting:
        batch: list[RequestTrace] = []
        while waiting and len(batch) < batch_size:
            if not batch:
                pick_index = 0
            else:
                running_signature = aggregate_signature(batch)
                candidate_count = min(topk, len(waiting))
                best_index = 0
                best_key: tuple[int, float, tuple[float, int, int, int]] | None = None
                for index in range(candidate_count):
                    candidate = waiting[index]
                    affinity = cosine(candidate, running_signature)
                    candidate_key = (
                        length_bucket(candidate, length_bucket_size),
                        -affinity,
                        base_sort_key(candidate, base_policy),
                    )
                    if best_key is None or candidate_key < best_key:
                        best_index = index
                        best_key = candidate_key
                pick_index = best_index
            batch.append(waiting.pop(pick_index))
        batches.append(batch)
    return batches


def flatten_batches(batches: list[list[RequestTrace]]) -> list[RequestTrace]:
    return [request for batch in batches for request in batch]


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def batch_pairwise_cosines(batches: list[list[RequestTrace]]) -> list[float]:
    values: list[float] = []
    for batch in batches:
        if len(batch) < 2:
            continue
        total = 0.0
        pairs = 0
        for i, left in enumerate(batch):
            for right in batch[i + 1 :]:
                total += cosine(left, right)
                pairs += 1
        if pairs:
            values.append(total / float(pairs))
    return values


def adjacent_cosines(order: list[RequestTrace]) -> list[float]:
    return [cosine(left, right) for left, right in zip(order, order[1:])]


def batch_unique_experts(batches: list[list[RequestTrace]], num_layers: int) -> list[float]:
    values: list[float] = []
    for batch in batches:
        layer_counts: list[int] = []
        for layer in range(num_layers):
            experts: set[int] = set()
            for request in batch:
                if layer < len(request.active_by_layer):
                    experts.update(request.active_by_layer[layer])
            layer_counts.append(len(experts))
        values.append(mean([float(item) for item in layer_counts]))
    return values


def simulate_lru_cache(
    batches: list[list[RequestTrace]],
    *,
    num_layers: int,
    capacity: int,
) -> dict[str, float | int]:
    if capacity <= 0:
        return {
            "capacity": capacity,
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "miss_rate": 0.0,
            "misses_per_batch": 0.0,
        }

    caches = [deque() for _ in range(num_layers)]
    cache_sets = [set() for _ in range(num_layers)]
    hits = 0
    misses = 0
    lookups = 0

    for batch in batches:
        for layer in range(num_layers):
            needed: set[int] = set()
            for request in batch:
                if layer < len(request.active_by_layer):
                    needed.update(request.active_by_layer[layer])

            cache = caches[layer]
            cache_set = cache_sets[layer]
            for expert in sorted(needed):
                lookups += 1
                if expert in cache_set:
                    hits += 1
                    cache.remove(expert)
                    cache.appendleft(expert)
                    continue
                misses += 1
                cache.appendleft(expert)
                cache_set.add(expert)
                while len(cache) > capacity:
                    evicted = cache.pop()
                    cache_set.remove(evicted)

    hit_rate = hits / float(lookups) if lookups else 0.0
    return {
        "capacity": capacity,
        "lookups": lookups,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "miss_rate": 1.0 - hit_rate if lookups else 0.0,
        "misses_per_batch": misses / float(len(batches)) if batches else 0.0,
    }


def summarize_batches(
    batches: list[list[RequestTrace]],
    *,
    num_layers: int,
    cache_capacities: list[int],
) -> dict[str, object]:
    order = flatten_batches(batches)
    pairwise = batch_pairwise_cosines(batches)
    adjacent = adjacent_cosines(order)
    uniques = batch_unique_experts(batches, num_layers)
    displacement = [
        abs(request.original_index - scheduled_index)
        for scheduled_index, request in enumerate(order)
    ]
    return {
        "num_batches": len(batches),
        "num_requests": len(order),
        "avg_batch_pairwise_cosine": mean(pairwise),
        "p50_batch_pairwise_cosine": percentile(pairwise, 50),
        "p90_batch_pairwise_cosine": percentile(pairwise, 90),
        "avg_adjacent_cosine": mean(adjacent),
        "avg_unique_experts_per_layer_per_batch": mean(uniques),
        "p90_unique_experts_per_layer_per_batch": percentile(uniques, 90),
        "avg_abs_reorder_displacement": mean([float(item) for item in displacement]),
        "max_abs_reorder_displacement": max(displacement) if displacement else 0,
        "cache": {
            str(capacity): simulate_lru_cache(
                batches,
                num_layers=num_layers,
                capacity=capacity,
            )
            for capacity in cache_capacities
        },
        "first_16_request_ids": [request.request_id for request in order[:16]],
    }


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def print_table(results: dict[str, dict[str, object]], cache_capacity: int) -> None:
    print(
        "policy\tbatch_cos\tadj_cos\tunique_exp/layer/batch"
        f"\tcache_hit@{cache_capacity}\tcache_miss/batch@{cache_capacity}"
        "\tavg_displacement"
    )
    for policy, metrics in results.items():
        cache = metrics["cache"][str(cache_capacity)]
        print(
            policy
            + "\t"
            + f"{metrics['avg_batch_pairwise_cosine']:.6f}\t"
            + f"{metrics['avg_adjacent_cosine']:.6f}\t"
            + f"{metrics['avg_unique_experts_per_layer_per_batch']:.3f}\t"
            + f"{cache['hit_rate']:.6f}\t"
            + f"{cache['misses_per_batch']:.3f}\t"
            + f"{metrics['avg_abs_reorder_displacement']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--affinity-topk", type=int, default=32)
    parser.add_argument("--affinity-base-policy", choices=("fcfs", "length"), default="length")
    parser.add_argument(
        "--length-bucket-size",
        type=int,
        default=0,
        help="When positive, affinity cannot jump across total-token buckets.",
    )
    parser.add_argument("--cache-capacities", default="8,16,32")
    parser.add_argument("--random-seed", type=int, default=20260425)
    args = parser.parse_args()

    traces = load_traces(args.signature_file, args.limit)
    num_layers = max(request.num_layers for request in traces)
    num_experts = max(request.num_experts for request in traces)
    cache_capacities = parse_int_list(args.cache_capacities)

    policies = ["fcfs", "length", "random", "affinity"]
    results: dict[str, dict[str, object]] = {}
    for policy in policies:
        batches = schedule_in_batches(
            traces,
            policy=policy,
            batch_size=args.batch_size,
            affinity_topk=args.affinity_topk,
            base_policy=args.affinity_base_policy,
            length_bucket_size=args.length_bucket_size,
            random_seed=args.random_seed,
        )
        results[policy] = summarize_batches(
            batches,
            num_layers=num_layers,
            cache_capacities=cache_capacities,
        )

    summary = {
        "signature_file": str(args.signature_file),
        "num_requests": len(traces),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "batch_size": args.batch_size,
        "affinity_topk": args.affinity_topk,
        "affinity_base_policy": args.affinity_base_policy,
        "length_bucket_size": args.length_bucket_size,
        "cache_capacities": cache_capacities,
        "random_seed": args.random_seed,
        "policies": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote replay summary to {args.output}")
    print_table(results, cache_capacities[min(1, len(cache_capacities) - 1)])


if __name__ == "__main__":
    main()
