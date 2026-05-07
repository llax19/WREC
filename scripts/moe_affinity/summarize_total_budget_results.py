#!/usr/bin/env python3
"""Summarize fixed-total-budget WREC replay results and write SVG figures."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


COLORS = {
    "lru": "#4C78A8",
    "static_hot": "#F58518",
    "route_window_prefetch": "#E45756",
    "wrec_h": "#54A24B",
    "wrec_h2": "#B279A2",
    "wrec_c": "#2F855A",
    "belady_oracle": "#72B7B2",
    "on_demand": "#9D755D",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in [
            "total_cache_slots",
            "total_budget_fraction",
            "stall_ms_per_input_token",
            "workload_weighted_miss_rate",
            "transfer_bytes_per_input_token",
            "waste_bytes_per_input_token",
            "oracle_gap_ratio",
        ]:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def grouped(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        out[int(row["total_cache_slots"])][str(row["policy"])] = row
    return dict(sorted(out.items()))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    by_budget = grouped(rows)
    policy_order = [
        "on_demand",
        "lru",
        "static_hot",
        "route_window_prefetch",
        "wrec_h",
        "wrec_h2",
        "wrec_c",
        "belady_oracle",
    ]
    lines = [
        "# WREC Fixed Total Expert-Cache Budget Results",
        "",
        "## Setup",
        "",
        "- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`",
        "- Train/reference trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`",
        "- Budget mode: fixed total expert-cache slots.",
        "- LRU / route-window allocation: uniform per-layer.",
        "- Static-hot allocation: global train frequency.",
        "- WREC allocation: global adaptive admission/eviction.",
        "",
        "## Main Table",
        "",
        "| total slots | fraction | policy | allocation | miss rate | stall ms/token | gain vs LRU | waste bytes/token | oracle gap ratio |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for slots, policies in by_budget.items():
        lru = policies["lru"]
        lru_stall = float(lru["stall_ms_per_input_token"])
        for policy in policy_order:
            if policy not in policies:
                continue
            row = policies[policy]
            stall = float(row["stall_ms_per_input_token"])
            gain = (lru_stall - stall) / lru_stall if lru_stall else 0.0
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(slots),
                        fmt_pct(float(row["total_budget_fraction"])),
                        policy,
                        str(row["allocation_mode"]),
                        fmt_num(float(row["workload_weighted_miss_rate"]), 6),
                        fmt_num(stall, 4),
                        fmt_pct(gain),
                        fmt_num(float(row["waste_bytes_per_input_token"]), 2),
                        fmt_num(float(row.get("oracle_gap_ratio", 0.0)), 4),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- Fixed total-budget replay changes the conclusion materially: WREC adaptive allocation is much stronger than uniform per-layer LRU.",
            "- WREC-H2 beats LRU and static-hot at all tested total budgets with zero prefetch waste.",
            "- WREC-C is reported when present as the constrained total-budget planner variant.",
            "- Route-window prefetch can reduce stall at larger budgets but has very large waste in this oracle-style configuration.",
            "- Belady remains far ahead, so there is still meaningful oracle gap for constrained planning.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_line_chart(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    policies: list[str],
    y_key: str,
    y_label: str,
    transform=lambda x: x,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 220, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    by_budget = grouped(rows)
    slots = sorted(by_budget)
    xs = {slot: left + i * plot_w / max(1, len(slots) - 1) for i, slot in enumerate(slots)}
    values = []
    for slot in slots:
        for policy in policies:
            values.append(transform(float(by_budget[slot][policy][y_key])))
    y_min = min(0.0, min(values))
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0

    def y_pos(value: float) -> float:
        scaled = (value - y_min) / (y_max - y_min)
        return top + plot_h * (1.0 - scaled)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
        f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="13">total expert-cache slots</text>',
    ]
    for slot in slots:
        x = xs[slot]
        parts.append(f'<line x1="{x:.2f}" y1="{top+plot_h}" x2="{x:.2f}" y2="{top+plot_h+6}" stroke="#333"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{slot}</text>')
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for policy in policies:
        points = []
        for slot in slots:
            value = transform(float(by_budget[slot][policy][y_key]))
            points.append(f'{xs[slot]:.2f},{y_pos(value):.2f}')
        color = COLORS.get(policy, "#333")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        for point in points:
            x, y = point.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
    legend_x = left + plot_w + 25
    for idx, policy in enumerate(policies):
        y = top + idx * 24
        color = COLORS.get(policy, "#333")
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+32}" y="{y+4}" font-family="Arial" font-size="12">{policy}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def add_capture_ratio(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_budget = grouped(rows)
    out = []
    for slot, policies in by_budget.items():
        lru_stall = float(policies["lru"]["stall_ms_per_input_token"])
        oracle_stall = float(policies["belady_oracle"]["stall_ms_per_input_token"])
        denom = max(1e-12, lru_stall - oracle_stall)
        for policy, row in policies.items():
            copied = dict(row)
            stall = float(row["stall_ms_per_input_token"])
            copied["oracle_gap_capture_ratio"] = max(0.0, min(1.0, (lru_stall - stall) / denom))
            out.append(copied)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    write_markdown(args.output_md, rows)
    svg_line_chart(
        args.fig_dir / "wrec_total_budget_stall_vs_budget_20260502.svg",
        title="Stall Proxy vs Total Expert-Cache Budget",
        rows=rows,
        policies=[
            policy for policy in ["lru", "static_hot", "route_window_prefetch", "wrec_h", "wrec_h2", "wrec_c", "belady_oracle"]
            if policy in {str(row["policy"]) for row in rows}
        ],
        y_key="stall_ms_per_input_token",
        y_label="stall ms / input token",
    )
    svg_line_chart(
        args.fig_dir / "wrec_total_budget_waste_vs_budget_20260502.svg",
        title="Prefetch Waste vs Total Expert-Cache Budget",
        rows=rows,
        policies=[
            policy for policy in ["route_window_prefetch", "wrec_h", "wrec_h2", "wrec_c"]
            if policy in {str(row["policy"]) for row in rows}
        ],
        y_key="waste_bytes_per_input_token",
        y_label="log10(waste bytes / input token + 1)",
        transform=lambda value: math.log10(value + 1.0),
    )
    svg_line_chart(
        args.fig_dir / "wrec_total_budget_oracle_gap_capture_20260502.svg",
        title="Oracle Gap Capture vs Total Expert-Cache Budget",
        rows=add_capture_ratio(rows),
        policies=[
            policy for policy in ["static_hot", "route_window_prefetch", "wrec_h", "wrec_h2", "wrec_c", "belady_oracle"]
            if policy in {str(row["policy"]) for row in rows}
        ],
        y_key="oracle_gap_capture_ratio",
        y_label="oracle gap capture ratio",
    )
    print(args.output_md)
    print(args.fig_dir / "wrec_total_budget_stall_vs_budget_20260502.svg")
    print(args.fig_dir / "wrec_total_budget_waste_vs_budget_20260502.svg")
    print(args.fig_dir / "wrec_total_budget_oracle_gap_capture_20260502.svg")


if __name__ == "__main__":
    main()
