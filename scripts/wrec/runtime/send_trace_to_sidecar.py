#!/usr/bin/env python3
"""Send routed expert trace events to a WREC runtime sidecar."""

from __future__ import annotations

import argparse
import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["sidecar_metrics"]
    client = payload["client"]
    timing = metrics["timing"]
    lines = [
        "# WREC Runtime Sidecar Smoke",
        "",
        "## Inputs",
        "",
        f"- Trace: `{payload['inputs']['trace']}`",
        f"- Sidecar URL: `{payload['inputs']['sidecar_url']}`",
        f"- Max events: `{payload['config']['max_events']}`",
        "",
        "## Client",
        "",
        f"- Events sent: `{client['events_sent']}`",
        f"- Expert refs sent: `{client['expert_refs_sent']}`",
        f"- Client elapsed: `{client['elapsed_seconds']:.6f}` s",
        f"- Mean client submit time: `{client['us_per_event']:.3f}` us/event",
        "",
        "## Sidecar Metrics",
        "",
        f"- Expert refs processed: `{metrics['expert_refs']}`",
        f"- Router events processed: `{metrics['router_events']}`",
        f"- Shadow hits: `{metrics['shadow_hits']}`",
        f"- Shadow misses: `{metrics['shadow_misses']}`",
        f"- Shadow miss rate: `{metrics['shadow_miss_rate']:.9f}`",
        f"- Would-admit: `{metrics['would_admit']}`",
        f"- Would-bypass: `{metrics['would_bypass']}`",
        f"- Would-evict: `{metrics['would_evict']}`",
        f"- Sidecar loop overhead: `{timing['online_loop_us_per_router_event']:.3f}` us/router event",
        f"- Decision overhead: `{timing['decision_us_per_miss']:.3f}` us/miss",
        "",
        "## Conclusion",
        "",
        "- The trace client successfully pushed routed expert events into the HTTP sidecar integration boundary.",
        "- The sidecar returned online WREC decisions while maintaining shadow cache state.",
        "- This is an external runtime integration smoke, not a vLLM internal expert-loading hook.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8765")
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    parsed = urlparse(args.sidecar_url)
    conn = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=60)
    health = get_json(conn, "/health")

    events_sent = 0
    expert_refs_sent = 0
    start = time.perf_counter()
    last_response: dict[str, Any] | None = None
    with args.trace.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if args.max_events > 0 and events_sent >= args.max_events:
                break
            row = json.loads(line)
            selected = row.get("selected_experts")
            if not isinstance(selected, list) or not selected:
                raise ValueError(f"{args.trace}:{line_no + 1} missing selected_experts")
            payload = {
                "request_id": str(row["request_id"]),
                "event_index": int(row.get("event_index", events_sent)),
                "layer": int(row["layer"]),
                "token_pos": int(row["token_pos"]),
                "selected_experts": [int(item) for item in selected],
            }
            last_response = post_json(conn, "/event", payload)
            events_sent += 1
            expert_refs_sent += len(selected)
    elapsed = time.perf_counter() - start
    metrics = get_json(conn, "/metrics")
    conn.close()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "trace": str(args.trace),
            "sidecar_url": args.sidecar_url,
        },
        "config": {
            "max_events": args.max_events,
        },
        "health": health,
        "client": {
            "events_sent": events_sent,
            "expert_refs_sent": expert_refs_sent,
            "elapsed_seconds": elapsed,
            "us_per_event": elapsed / max(1, events_sent) * 1e6,
        },
        "last_response": last_response,
        "sidecar_metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
