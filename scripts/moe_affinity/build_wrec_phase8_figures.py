#!/usr/bin/env python3
"""Build minimal paper-ready SVG figures for WREC Phase 8."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def build_hrm_transfer_heatmap(
    rows: list[dict[str, str]],
    *,
    dtype: str,
    bandwidth_gbps: float,
    output: Path,
) -> None:
    filtered = [
        row for row in rows
        if row["dtype"] == dtype and abs(float(row["bandwidth_gbps"]) - bandwidth_gbps) < 1e-9
    ]
    gpu_values = sorted({float(row["gpu_memory_gb"]) for row in filtered})
    kv_values = sorted({float(row["kv_cache_budget_gb"]) for row in filtered})

    heatmap: dict[tuple[float, float], float] = {}
    for gpu in gpu_values:
        for kv in kv_values:
            bucket = [
                row for row in filtered
                if float(row["gpu_memory_gb"]) == gpu and float(row["kv_cache_budget_gb"]) == kv
            ]
            total = len(bucket)
            transfer_bound = sum(1 for row in bucket if row["bottleneck_type"] == "expert-transfer-bound")
            heatmap[(gpu, kv)] = transfer_bound / total if total else 0.0

    width, height = 900, 520
    left, top = 120, 90
    cell_w, cell_h = 150, 82
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="34" text-anchor="middle" font-family="Arial" font-size="22">Mixtral BF16 Transfer-Bound HRM Heatmap</text>',
        '<text x="450" y="58" text-anchor="middle" font-family="Arial" font-size="13">Cell value = fraction of tested resident-fraction and decode-token settings that are expert-transfer-bound at measured 41.37 GB/s</text>',
        f'<text x="40" y="{top + 2*cell_h}" transform="rotate(-90 40 {top + 2*cell_h})" text-anchor="middle" font-family="Arial" font-size="13">GPU memory budget (GB)</text>',
        f'<text x="{left + 2*cell_w}" y="{top + len(gpu_values)*cell_h + 55}" text-anchor="middle" font-family="Arial" font-size="13">KV reservation (GB)</text>',
    ]

    for col, kv in enumerate(kv_values):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.2f}" y="{top-16}" text-anchor="middle" font-family="Arial" font-size="13">{kv:g}</text>')
    for row_idx, gpu in enumerate(gpu_values):
        y = top + row_idx * cell_h + cell_h / 2
        parts.append(f'<text x="{left-20}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="13">{gpu:g}</text>')

    for row_idx, gpu in enumerate(gpu_values):
        for col, kv in enumerate(kv_values):
            x = left + col * cell_w
            y = top + row_idx * cell_h
            value = heatmap[(gpu, kv)]
            # White -> red
            shade = int(255 - 155 * value)
            color = f"rgb(255,{shade},{shade})"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#666"/>')
            parts.append(f'<text x="{x + cell_w/2:.2f}" y="{y + cell_h/2 - 4:.2f}" text-anchor="middle" font-family="Arial" font-size="18">{value:.2f}</text>')
            parts.append(f'<text x="{x + cell_w/2:.2f}" y="{y + cell_h/2 + 18:.2f}" text-anchor="middle" font-family="Arial" font-size="11">transfer-bound ratio</text>')

    legend_x, legend_y = 730, 120
    for idx, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        shade = int(255 - 155 * value)
        color = f"rgb(255,{shade},{shade})"
        y = legend_y + idx * 26
        parts.append(f'<rect x="{legend_x}" y="{y}" width="24" height="18" fill="{color}" stroke="#666"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y + 13}" font-family="Arial" font-size="12">{value:.2f}</text>')
    parts.append("</svg>")
    write_text(output, "\n".join(parts))


def build_locality_summary_figure(stats: dict[str, Any], *, output: Path) -> None:
    freq_layers = stats["train_expert_frequency"]["layers"]
    layer_ids = sorted(int(layer) for layer in freq_layers)
    top2_shares = [float(freq_layers[str(layer)]["top2_share"]) for layer in layer_ids]
    reuse = stats["train_reuse_distance"]["same_layer_expert_ref_distance"]
    window_stats = stats["train_working_set"]["by_layer_window"]
    overlap = stats["train_eval_hotness_shift"]["topk_overlap_by_layer"]
    windows = sorted(int(key) for key in window_stats)

    width, height = 1160, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="580" y="32" text-anchor="middle" font-family="Arial" font-size="22">Mixtral Prefill Expert Locality Summary</text>',
        '<text x="580" y="56" text-anchor="middle" font-family="Arial" font-size="13">Train trace: 29,123 input tokens, 931,936 router events, 1,863,872 expert refs</text>',
    ]

    # Panel 1: top2 share by layer
    p1 = {"x": 70, "y": 110, "w": 470, "h": 220}
    parts += [
        f'<text x="{p1["x"] + p1["w"]/2}" y="{p1["y"]-14}" text-anchor="middle" font-family="Arial" font-size="15">Top-2 expert share by layer</text>',
        f'<line x1="{p1["x"]}" y1="{p1["y"]}" x2="{p1["x"]}" y2="{p1["y"] + p1["h"]}" stroke="#333"/>',
        f'<line x1="{p1["x"]}" y1="{p1["y"] + p1["h"]}" x2="{p1["x"] + p1["w"]}" y2="{p1["y"] + p1["h"]}" stroke="#333"/>',
    ]
    points = []
    for idx, value in enumerate(top2_shares):
        x = p1["x"] + idx * p1["w"] / max(1, len(top2_shares) - 1)
        y = p1["y"] + (1.0 - value) * p1["h"]
        points.append(f"{x:.2f},{y:.2f}")
        if idx in {0, 15, 31}:
            parts.append(f'<text x="{x:.2f}" y="{p1["y"] + p1["h"] + 18}" text-anchor="middle" font-family="Arial" font-size="11">{idx}</text>')
    for tick in [0.25, 0.5, 0.75, 1.0]:
        y = p1["y"] + (1.0 - tick) * p1["h"]
        parts.append(f'<line x1="{p1["x"]-5}" y1="{y:.2f}" x2="{p1["x"] + p1["w"]}" y2="{y:.2f}" stroke="#eee"/>')
        parts.append(f'<text x="{p1["x"]-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    parts.append(f'<polyline fill="none" stroke="#4C78A8" stroke-width="2.5" points="{" ".join(points)}"/>')

    # Panel 2: reuse distance quantiles
    p2 = {"x": 620, "y": 110, "w": 230, "h": 220}
    parts += [
        f'<text x="{p2["x"] + p2["w"]/2}" y="{p2["y"]-14}" text-anchor="middle" font-family="Arial" font-size="15">Reuse distance quantiles</text>',
        f'<line x1="{p2["x"]}" y1="{p2["y"]}" x2="{p2["x"]}" y2="{p2["y"] + p2["h"]}" stroke="#333"/>',
        f'<line x1="{p2["x"]}" y1="{p2["y"] + p2["h"]}" x2="{p2["x"] + p2["w"]}" y2="{p2["y"] + p2["h"]}" stroke="#333"/>',
    ]
    reuse_vals = [("p50", float(reuse["p50"])), ("p90", float(reuse["p90"])), ("p99", float(reuse["p99"]))]
    max_reuse = max(value for _, value in reuse_vals)
    for idx, (label, value) in enumerate(reuse_vals):
        x = p2["x"] + 35 + idx * 65
        h = p2["h"] * value / max_reuse
        y = p2["y"] + p2["h"] - h
        parts.append(f'<rect x="{x}" y="{y:.2f}" width="42" height="{h:.2f}" fill="#F58518"/>')
        parts.append(f'<text x="{x + 21}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.0f}</text>')
        parts.append(f'<text x="{x + 21}" y="{p2["y"] + p2["h"] + 18}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append(f'<text x="{p2["x"] + p2["w"]/2}" y="{p2["y"] + p2["h"] + 38}" text-anchor="middle" font-family="Arial" font-size="11">same-layer expert ref distance</text>')

    # Panel 3: short-window locality
    p3 = {"x": 70, "y": 390, "w": 470, "h": 170}
    parts += [
        f'<text x="{p3["x"] + p3["w"]/2}" y="{p3["y"]-14}" text-anchor="middle" font-family="Arial" font-size="15">Short-window locality</text>',
        f'<line x1="{p3["x"]}" y1="{p3["y"]}" x2="{p3["x"]}" y2="{p3["y"] + p3["h"]}" stroke="#333"/>',
        f'<line x1="{p3["x"]}" y1="{p3["y"] + p3["h"]}" x2="{p3["x"] + p3["w"]}" y2="{p3["y"] + p3["h"]}" stroke="#333"/>',
    ]
    max_unique = max(float(window_stats[str(w)]["unique_experts_per_layer_window"]["p50"]) for w in windows)
    points_unique = []
    points_prob = []
    for idx, window in enumerate(windows):
        x = p3["x"] + idx * p3["w"] / max(1, len(windows) - 1)
        unique = float(window_stats[str(window)]["unique_experts_per_layer_window"]["p50"])
        prob = float(window_stats[str(window)]["expert_use_probability_per_window"]["p50"])
        y_unique = p3["y"] + (1.0 - unique / max_unique) * p3["h"]
        y_prob = p3["y"] + (1.0 - prob) * p3["h"]
        points_unique.append(f"{x:.2f},{y_unique:.2f}")
        points_prob.append(f"{x:.2f},{y_prob:.2f}")
        parts.append(f'<text x="{x:.2f}" y="{p3["y"] + p3["h"] + 18}" text-anchor="middle" font-family="Arial" font-size="11">{window}</text>')
    parts.append(f'<polyline fill="none" stroke="#54A24B" stroke-width="2.5" points="{" ".join(points_unique)}"/>')
    parts.append(f'<polyline fill="none" stroke="#E45756" stroke-width="2.5" points="{" ".join(points_prob)}"/>')
    parts.append(f'<text x="{p3["x"] + 18}" y="{p3["y"] + 18}" font-family="Arial" font-size="11" fill="#54A24B">p50 unique experts / window</text>')
    parts.append(f'<text x="{p3["x"] + 18}" y="{p3["y"] + 36}" font-family="Arial" font-size="11" fill="#E45756">p50 expert-use probability</text>')

    # Panel 4: train/eval overlap
    p4 = {"x": 620, "y": 390, "w": 230, "h": 170}
    parts += [
        f'<text x="{p4["x"] + p4["w"]/2}" y="{p4["y"]-14}" text-anchor="middle" font-family="Arial" font-size="15">Train/eval hotness stability</text>',
        f'<line x1="{p4["x"]}" y1="{p4["y"]}" x2="{p4["x"]}" y2="{p4["y"] + p4["h"]}" stroke="#333"/>',
        f'<line x1="{p4["x"]}" y1="{p4["y"] + p4["h"]}" x2="{p4["x"] + p4["w"]}" y2="{p4["y"] + p4["h"]}" stroke="#333"/>',
    ]
    overlap_vals = [("top2", float(overlap["2"]["mean"])), ("top4", float(overlap["4"]["mean"]))]
    for idx, (label, value) in enumerate(overlap_vals):
        x = p4["x"] + 45 + idx * 85
        h = p4["h"] * value
        y = p4["y"] + p4["h"] - h
        parts.append(f'<rect x="{x}" y="{y:.2f}" width="48" height="{h:.2f}" fill="#B279A2"/>')
        parts.append(f'<text x="{x + 24}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
        parts.append(f'<text x="{x + 24}" y="{p4["y"] + p4["h"] + 18}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append(f'<text x="{p4["x"] + p4["w"]/2}" y="{p4["y"] + p4["h"] + 38}" text-anchor="middle" font-family="Arial" font-size="11">mean train/eval overlap by layer</text>')

    # Key numbers
    parts.append('<text x="910" y="148" font-family="Arial" font-size="14">Key numbers</text>')
    parts.append('<text x="910" y="172" font-family="Arial" font-size="12">reuse p50/p90/p99: 5 / 27 / 7097</text>')
    parts.append('<text x="910" y="194" font-family="Arial" font-size="12">window-4 unique experts p50: 5</text>')
    parts.append('<text x="910" y="216" font-family="Arial" font-size="12">window-4 use probability p50: 0.625</text>')
    parts.append('<text x="910" y="238" font-family="Arial" font-size="12">top-2 overlap mean: 0.938</text>')
    parts.append('<text x="910" y="260" font-family="Arial" font-size="12">top-4 overlap mean: 0.938</text>')

    parts.append("</svg>")
    write_text(output, "\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrm-csv", type=Path, required=True)
    parser.add_argument("--workload-json", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args()

    hrm_rows = read_csv(args.hrm_csv)
    stats = read_json(args.workload_json)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    build_hrm_transfer_heatmap(
        hrm_rows,
        dtype="bf16",
        bandwidth_gbps=41.37220609315469,
        output=args.fig_dir / "wrec_hrm_transfer_bound_heatmap_mixtral8x7b_20260503.svg",
    )
    build_locality_summary_figure(
        stats,
        output=args.fig_dir / "wrec_locality_summary_mixtral8x7b_dolly_train_20260503.svg",
    )
    print(args.fig_dir / "wrec_hrm_transfer_bound_heatmap_mixtral8x7b_20260503.svg")
    print(args.fig_dir / "wrec_locality_summary_mixtral8x7b_dolly_train_20260503.svg")


if __name__ == "__main__":
    main()
