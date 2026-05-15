#!/usr/bin/env python3
"""HTTP sidecar for online WREC runtime shadow integration.

The sidecar exposes a small runtime-facing API:

  POST /event
    {"request_id": "...", "layer": 0, "token_pos": 0,
     "selected_experts": [1, 4], "event_index": 0}

It returns WREC shadow decisions for each selected expert while maintaining a
shadow resident set. This process still does not control real expert loading;
it is the integration boundary a runtime hook can call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from moe_affinity.simulate_expert_cache_offload import ExpertRef, infer_expert_bytes, load_event_trace
from wrec import WrecExpertCachePolicy, WrecPolicyConfig, build_wrec_stats


@dataclass
class RuntimeCache:
    resident: set[tuple[int, int]]
    last_touch: dict[tuple[int, int], int]
    lru_order: OrderedDict[tuple[int, int], None]
    prefetched_unused: set[tuple[int, int]]


@dataclass
class RequestHistory:
    request_counts: dict[int, Counter[int]]
    request_totals: dict[int, int]
    token_layer_experts: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)
    current_event_key: tuple[int, int] | None = None
    current_event_experts: list[int] = field(default_factory=list)


def top_global_hot(refs: list[ExpertRef], total_slots: int) -> set[tuple[int, int]]:
    counts = Counter((ref.layer, ref.expert) for ref in refs)
    return {key for key, _ in counts.most_common(total_slots)}


def init_cache(initial: set[tuple[int, int]]) -> RuntimeCache:
    return RuntimeCache(
        resident=set(initial),
        last_touch={key: 0 for key in initial},
        lru_order=OrderedDict((key, None) for key in initial),
        prefetched_unused=set(),
    )


class WrecRuntimeSidecarEngine:
    def __init__(
        self,
        *,
        train_trace: Path,
        model_path: Path | None,
        expert_bytes_override: float | None,
        dtype: str,
        bandwidth_gbps: float,
        total_slots: int,
        window_size: int,
        history_size: int,
        recent_weight: float,
        request_weight: float,
        cross_layer_weight: float,
        contention_penalty: float,
        ranking_score_threshold: float | None,
    ) -> None:
        load_start = time.perf_counter()
        train_refs, train_meta = load_event_trace(train_trace)
        self.load_seconds = time.perf_counter() - load_start
        self.train_trace = train_trace
        self.model_path = model_path
        self.expert_bytes = infer_expert_bytes(model_path, dtype, expert_bytes_override)
        self.bandwidth_gbps = bandwidth_gbps
        self.num_layers = int(train_meta["num_layers"])
        self.num_experts = int(train_meta["num_experts"])
        self.total_slots = max(0, min(total_slots, self.num_layers * self.num_experts))
        self.history_size = history_size
        self.ranking_score_threshold = ranking_score_threshold
        self.created_at_utc = datetime.now(timezone.utc).isoformat()

        prior_start = time.perf_counter()
        self.stats = build_wrec_stats(
            train_refs,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            window_size=window_size,
            expert_bytes=self.expert_bytes,
            bandwidth_gbps=bandwidth_gbps,
        )
        self.prior_seconds = time.perf_counter() - prior_start
        self.policy = WrecExpertCachePolicy(
            WrecPolicyConfig(
                recent_weight=recent_weight,
                request_weight=request_weight,
                cross_layer_weight=cross_layer_weight,
                contention_penalty=contention_penalty,
            )
        )
        self.cache = init_cache(top_global_hot(train_refs, self.total_slots))
        self.recent_refs = {layer: deque(maxlen=history_size) for layer in range(self.num_layers)}
        self.recent_counts = {layer: Counter() for layer in range(self.num_layers)}
        self.requests: dict[str, RequestHistory] = {}
        self.ref_index = 0
        self.router_events = 0
        self.shadow_hits = 0
        self.shadow_misses = 0
        self.would_admit = 0
        self.would_bypass = 0
        self.would_evict = 0
        self.loop_ns = 0
        self.decision_ns = 0
        self.history_update_ns = 0
        self.ranking_ns = 0

    def _empty_online_view(self) -> Any:
        return SimpleNamespace(
            recent_refs={
                layer: deque(maxlen=self.history_size)
                for layer in range(self.num_layers)
            },
            recent_counts={layer: Counter() for layer in range(self.num_layers)},
            request_counts={layer: Counter() for layer in range(self.num_layers)},
            request_totals={layer: 0 for layer in range(self.num_layers)},
            token_layer_experts={},
        )

    def _request_history(self, request_id: str) -> RequestHistory:
        history = self.requests.get(request_id)
        if history is None:
            history = RequestHistory(
                request_counts={layer: Counter() for layer in range(self.num_layers)},
                request_totals={layer: 0 for layer in range(self.num_layers)},
            )
            self.requests[request_id] = history
        return history

    def _online_view(self, request_id: str) -> Any:
        history = self._request_history(request_id)
        return SimpleNamespace(
            recent_refs=self.recent_refs,
            recent_counts=self.recent_counts,
            request_counts=history.request_counts,
            request_totals=history.request_totals,
            token_layer_experts=history.token_layer_experts,
        )

    def _update_history(self, ref: ExpertRef) -> None:
        history = self._request_history(ref.request_id)
        event_key = (ref.layer, ref.token_pos)
        if history.current_event_key is None:
            history.current_event_key = event_key
        elif history.current_event_key != event_key:
            history.token_layer_experts[history.current_event_key] = tuple(
                history.current_event_experts
            )
            history.current_event_key = event_key
            history.current_event_experts = []

        recent = self.recent_refs[ref.layer]
        counts = self.recent_counts[ref.layer]
        if recent.maxlen is not None and len(recent) >= recent.maxlen:
            expired = recent[0]
            counts[expired] -= 1
            if counts[expired] <= 0:
                counts.pop(expired, None)
        recent.append(ref.expert)
        counts[ref.expert] += 1
        history.request_counts[ref.layer][ref.expert] += 1
        history.request_totals[ref.layer] += 1
        history.current_event_experts.append(ref.expert)

    def rank_layer_experts(
        self,
        *,
        layer: int,
        token_pos: int = 0,
        timestamp: int | None = None,
        request_id: str | None = None,
    ) -> list[int]:
        online = (
            self._online_view(request_id)
            if request_id is not None
            else self._empty_online_view()
        )
        effective_timestamp = self.ref_index if timestamp is None else timestamp
        scored = [
            (
                self.policy.score(
                    layer=layer,
                    token_pos=token_pos,
                    expert=expert,
                    timestamp=effective_timestamp,
                    cache=self.cache,
                    online=online,
                    stats=self.stats,
                ),
                expert,
            )
            for expert in range(self.num_experts)
        ]
        if self.ranking_score_threshold is not None:
            scored = [
                item
                for item in scored
                if item[0] >= self.ranking_score_threshold
            ]
        return [
            expert
            for _, expert in sorted(
                scored,
                key=lambda item: (-item[0], item[1]),
            )
        ]

    def ranked_experts_payload(
        self,
        *,
        layers: list[int] | None = None,
        token_pos: int = 0,
        request_id: str | None = None,
    ) -> dict[str, list[int]]:
        ranking_start = time.perf_counter_ns()
        selected_layers = layers if layers is not None else list(range(self.num_layers))
        payload = {
            str(layer): self.rank_layer_experts(
                layer=layer,
                token_pos=token_pos,
                request_id=request_id,
            )
            for layer in selected_layers
            if 0 <= layer < self.num_layers
        }
        self.ranking_ns += time.perf_counter_ns() - ranking_start
        return payload

    def process_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload["request_id"])
        layer = int(payload["layer"])
        token_pos = int(payload["token_pos"])
        selected = payload.get("selected_experts")
        if not isinstance(selected, list) or not selected:
            raise ValueError("selected_experts must be a non-empty list")
        event_index = int(payload.get("event_index", self.router_events))
        event_start = time.perf_counter_ns()
        decisions: list[dict[str, Any]] = []
        self.router_events += 1

        for expert_value in selected:
            expert = int(expert_value)
            ref = ExpertRef(
                ref_index=self.ref_index,
                event_index=event_index,
                request_id=request_id,
                layer=layer,
                token_pos=token_pos,
                expert=expert,
            )
            self.ref_index += 1
            key = (layer, expert)
            online = self._online_view(request_id)
            if key in self.cache.resident:
                self.shadow_hits += 1
                self.policy.touch(self.cache, key, ref.ref_index)
                self.cache.prefetched_unused.discard(key)
                update_start = time.perf_counter_ns()
                self._update_history(ref)
                self.history_update_ns += time.perf_counter_ns() - update_start
                decisions.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "shadow_hit": True,
                        "would_admit": False,
                        "would_bypass": False,
                        "would_evict": None,
                    }
                )
                continue

            self.shadow_misses += 1
            update_start = time.perf_counter_ns()
            self._update_history(ref)
            self.history_update_ns += time.perf_counter_ns() - update_start
            before = set(self.cache.resident)
            online = self._online_view(request_id)
            decision_start = time.perf_counter_ns()
            self.policy.admit_or_bypass(
                self.cache,
                ref=ref,
                total_slots=self.total_slots,
                online=online,
                stats=self.stats,
            )
            self.decision_ns += time.perf_counter_ns() - decision_start
            after = set(self.cache.resident)
            admitted = key in after and key not in before
            evicted = sorted(before - after)
            if admitted:
                self.would_admit += 1
            else:
                self.would_bypass += 1
            self.would_evict += len(evicted)
            decisions.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "shadow_hit": False,
                    "would_admit": admitted,
                    "would_bypass": not admitted,
                    "would_evict": evicted[0] if evicted else None,
                }
            )

        self.loop_ns += time.perf_counter_ns() - event_start
        ranked_layers = [layer]
        if layer + 1 < self.num_layers:
            ranked_layers.append(layer + 1)
        return {
            "request_id": request_id,
            "event_index": event_index,
            "decisions": decisions,
            "ranked_experts_by_layer": self.ranked_experts_payload(
                layers=ranked_layers,
                token_pos=token_pos,
                request_id=request_id,
            ),
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict[str, Any]:
        expert_refs = max(1, self.ref_index)
        router_events = max(1, self.router_events)
        demand_transfer_bytes = self.shadow_misses * self.expert_bytes
        stall_ms = demand_transfer_bytes / (self.bandwidth_gbps * 1e9) * 1000.0
        return {
            "created_at_utc": self.created_at_utc,
            "train_trace": str(self.train_trace),
            "model_path": str(self.model_path) if self.model_path else None,
            "total_slots": self.total_slots,
            "ranking_score_threshold": self.ranking_score_threshold,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "expert_bytes": self.expert_bytes,
            "bandwidth_gbps": self.bandwidth_gbps,
            "expert_refs": self.ref_index,
            "router_events": self.router_events,
            "shadow_hits": self.shadow_hits,
            "shadow_misses": self.shadow_misses,
            "shadow_hit_rate": self.shadow_hits / expert_refs,
            "shadow_miss_rate": self.shadow_misses / expert_refs,
            "would_admit": self.would_admit,
            "would_bypass": self.would_bypass,
            "would_evict": self.would_evict,
            "final_resident": len(self.cache.resident),
            "demand_transfer_bytes": demand_transfer_bytes,
            "stall_ms": stall_ms,
            "timing": {
                "load_seconds": self.load_seconds,
                "prior_seconds": self.prior_seconds,
                "online_loop_seconds": self.loop_ns / 1e9,
                "online_loop_us_per_expert_ref": self.loop_ns / expert_refs / 1000.0,
                "online_loop_us_per_router_event": self.loop_ns / router_events / 1000.0,
                "history_update_us_per_expert_ref": self.history_update_ns / expert_refs / 1000.0,
                "decision_us_per_miss": self.decision_ns / max(1, self.shadow_misses) / 1000.0,
                "ranking_seconds": self.ranking_ns / 1e9,
                "ranking_us_per_router_event": self.ranking_ns / router_events / 1000.0,
            },
            "claim_boundary": {
                "does_control_real_expert_loading": False,
                "does_measure_end_to_end_latency": False,
                "integration_boundary": "HTTP sidecar /event API for runtime hooks",
            },
        }


ENGINE: WrecRuntimeSidecarEngine | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "WrecRuntimeSidecar/0.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if ENGINE is None:
            self._send(503, {"error": "engine not initialized"})
            return
        if self.path == "/health":
            self._send(200, {"status": "ok", "total_slots": ENGINE.total_slots})
            return
        if self.path == "/metrics":
            self._send(200, ENGINE.metrics())
            return
        if self.path == "/rankings":
            self._send(
                200,
                {
                    "ranked_experts_by_layer": ENGINE.ranked_experts_payload(),
                    "metrics": ENGINE.metrics(),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if ENGINE is None:
            self._send(503, {"error": "engine not initialized"})
            return
        if self.path != "/event":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = ENGINE.process_event(payload)
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            self._send(400, {"error": repr(exc)})
            return
        self._send(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bandwidth-gbps", type=float, default=41.37220609315469)
    parser.add_argument("--total-slots", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--recent-weight", type=float, default=0.0)
    parser.add_argument("--request-weight", type=float, default=1024.0)
    parser.add_argument("--cross-layer-weight", type=float, default=1024.0)
    parser.add_argument("--contention-penalty", type=float, default=0.0)
    parser.add_argument("--ranking-score-threshold", type=float, default=None)
    args = parser.parse_args()

    global ENGINE
    ENGINE = WrecRuntimeSidecarEngine(
        train_trace=args.train_trace,
        model_path=args.model_path,
        expert_bytes_override=args.expert_bytes,
        dtype=args.dtype,
        bandwidth_gbps=args.bandwidth_gbps,
        total_slots=args.total_slots,
        window_size=args.window_size,
        history_size=args.history_size,
        recent_weight=args.recent_weight,
        request_weight=args.request_weight,
        cross_layer_weight=args.cross_layer_weight,
        contention_penalty=args.contention_penalty,
        ranking_score_threshold=args.ranking_score_threshold,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "listening", "host": args.host, "port": args.port}, ensure_ascii=False))
    server.serve_forever()


if __name__ == "__main__":
    main()
