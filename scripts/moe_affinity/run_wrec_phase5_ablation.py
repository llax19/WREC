#!/usr/bin/env python3
"""Run WREC Phase 5 ablation and sensitivity sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from simulate_expert_cache_offload import infer_expert_bytes, load_event_trace
from simulate_expert_cache_total_budget import (
    add_oracle_gap,
    simulate_total_budget_policy,
    total_slots_for_fraction,
)
from wrec import WrecStats, build_wrec_stats


@dataclass(frozen=True)
class Variant:
    suite: str
    variant: str
    policy: str
    score_mode: str = "full"
    bandwidth_gbps: float = 41.37220609315469
    window_size: int = 4
    history_size: int = 8
    recent_weight: float = 512.0
    request_weight: float = 1024.0
    cross_layer_weight: float = 1024.0
    contention_penalty: float = 0.0
    wrec_c_prefetch_queue_depth: int = 0
    wrec_c_overlap_ms: float = 0.0


def score_mode_stats(
    stats: WrecStats,
    *,
    expert_bytes: float,
    bandwidth_gbps: float,
    mode: str,
) -> WrecStats:
    transfer_ms = expert_bytes / (bandwidth_gbps * 1e9) * 1000.0
    base_score: dict[int, dict[int, float]] = defaultdict(dict)
    for layer, by_expert in stats.p_window_use.items():
        for expert, p_use in by_expert.items():
            expected = stats.expected_routed_tokens[layer].get(expert, 0.0)
            if mode == "full":
                score = p_use * expected * transfer_ms - transfer_ms
            elif mode == "no_workload":
                score = p_use * transfer_ms - transfer_ms
            elif mode == "no_transfer":
                score = p_use * expected * transfer_ms
            else:
                raise ValueError(f"unknown score mode: {mode}")
            base_score[layer][expert] = score
    return WrecStats(
        p_window_use=stats.p_window_use,
        expected_routed_tokens=stats.expected_routed_tokens,
        base_score=base_score,
        train_frequency=stats.train_frequency,
        cross_layer_transition=stats.cross_layer_transition,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def enrich_with_gains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["suite"]), float(row["bandwidth_gbps"]), int(row["total_cache_slots"]))
        if row.get("variant") == "lru_baseline":
            by_key[key] = row

    enriched = []
    for row in rows:
        copied = dict(row)
        key = (str(row["suite"]), float(row["bandwidth_gbps"]), int(row["total_cache_slots"]))
        lru = by_key.get(key)
        if lru is not None:
            lru_stall = float(lru["stall_ms_per_input_token"])
            stall = float(row["stall_ms_per_input_token"])
            copied["lru_stall_ms_per_input_token"] = lru_stall
            copied["gain_vs_lru"] = (lru_stall - stall) / lru_stall if lru_stall else 0.0
            copied["saved_ms_per_input_token_vs_lru"] = lru_stall - stall
        enriched.append(copied)
    return enriched


def grouped(rows: list[dict[str, Any]], key_name: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key_name])].append(row)
    return dict(out)


def svg_bar(path: Path, *, title: str, values: list[tuple[str, float]], y_label: str) -> None:
    width, height = 980, 520
    left, right, top, bottom = 90, 40, 60, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max([value for _, value in values] + [0.01])
    bar_w = plot_w / max(1, len(values)) * 0.68
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = top + plot_h * (1 - value / max_value)
        parts.append(f'<line x1="{left-5}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for idx, (label, value) in enumerate(values):
        x = left + (idx + 0.5) * plot_w / max(1, len(values))
        h = plot_h * value / max_value
        y = top + plot_h - h
        parts.append(f'<rect x="{x-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#4C78A8"/>')
        parts.append(f'<text x="{x:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        parts.append(
            f'<text x="{x:.2f}" y="{top+plot_h+18}" transform="rotate(35 {x:.2f} {top+plot_h+18})" '
            f'text-anchor="start" font-family="Arial" font-size="11">{label}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_line(path: Path, *, title: str, rows: list[tuple[float, float]], x_label: str, y_label: str) -> None:
    width, height = 860, 500
    left, right, top, bottom = 90, 50, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [x for x, _ in rows]
    ys = [y for _, y in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1

    def xp(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def yp(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(f"{xp(x):.2f},{yp(y):.2f}" for x, y in rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
        f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>',
    ]
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = yp(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for x, y in rows:
        parts.append(f'<line x1="{xp(x):.2f}" y1="{top+plot_h}" x2="{xp(x):.2f}" y2="{top+plot_h+6}" stroke="#333"/>')
        parts.append(f'<text x="{xp(x):.2f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{x:g}</text>')
    parts.append(f'<polyline fill="none" stroke="#B279A2" stroke-width="2.5" points="{points}"/>')
    for x, y in rows:
        parts.append(f'<circle cx="{xp(x):.2f}" cy="{yp(y):.2f}" r="3.5" fill="#B279A2"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(path: Path, *, title: str, rows: list[dict[str, Any]], group_key: str) -> None:
    lines = [f"# {title}", ""]
    for group, group_rows in sorted(grouped(rows, group_key).items()):
        tested = [row for row in group_rows if str(row.get("variant")) != "lru_baseline"]
        gains = [float(row.get("gain_vs_lru", 0.0)) for row in tested]
        saved = [float(row.get("saved_ms_per_input_token_vs_lru", 0.0)) for row in tested]
        lines.extend(
            [
                f"## {group}",
                "",
                f"- rows: {len(group_rows)}",
                f"- avg gain vs LRU: {100 * average(gains):.2f}%",
                f"- avg saved stall: {average(saved):.4f} ms/input-token",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- WREC-H2 is demand-load admission/eviction in this simulator, so prefetch-only ablation is not emitted here.",
            "- `no_transfer` changes the WREC score formula, but same-sized experts make the transfer term mostly a constant-ranking term.",
            "- `route_window_prefetch` uses eval future window in the existing simulator and should be read as oracle-style stress test, not deployable online policy.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(
    *,
    refs: list[Any],
    metadata: dict[str, Any],
    static_refs: list[Any],
    stats_cache: dict[tuple[int, float, str], WrecStats],
    variant: Variant,
    fractions: list[float],
    expert_bytes: float,
) -> list[dict[str, Any]]:
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])
    stats_key = (variant.window_size, variant.bandwidth_gbps, variant.score_mode)
    if stats_key not in stats_cache:
        base = build_wrec_stats(
            static_refs,
            num_layers=num_layers,
            num_experts=num_experts,
            window_size=variant.window_size,
            expert_bytes=expert_bytes,
            bandwidth_gbps=variant.bandwidth_gbps,
        )
        stats_cache[stats_key] = score_mode_stats(
            base,
            expert_bytes=expert_bytes,
            bandwidth_gbps=variant.bandwidth_gbps,
            mode=variant.score_mode,
        )
    wrec_stats = stats_cache[stats_key]
    rows = []
    for fraction in fractions:
        print(
            f"{variant.suite}:{variant.variant}: fraction={fraction:g}, "
            f"bw={variant.bandwidth_gbps:g}, window={variant.window_size}, history={variant.history_size}",
            flush=True,
        )
        total_slots = total_slots_for_fraction(num_layers, num_experts, fraction)
        row = simulate_total_budget_policy(
            refs,
            metadata=metadata,
            static_refs=static_refs,
            policy=variant.policy,
            total_slots=total_slots,
            total_budget_fraction=fraction,
            expert_bytes=expert_bytes,
            bandwidth_gbps=variant.bandwidth_gbps,
            window_size=variant.window_size,
            prefetch_queue_depth=1,
            wrec_stats=wrec_stats if variant.policy in {"wrec_h", "wrec_h2", "wrec_c"} else None,
            wrec_recent_weight=variant.recent_weight,
            wrec_request_weight=variant.request_weight,
            wrec_cross_layer_weight=variant.cross_layer_weight,
            wrec_contention_penalty=variant.contention_penalty,
            wrec_history_size=variant.history_size,
            wrec_c_prefetch_queue_depth=variant.wrec_c_prefetch_queue_depth,
            wrec_c_overlap_ms=variant.wrec_c_overlap_ms,
            wrec_c_min_slots_per_layer=0,
            wrec_c_max_slots_per_layer=None,
            wrec_c_candidates_per_layer=4,
            wrec_c_replan_interval=16,
        )
        row["suite"] = variant.suite
        row["variant"] = variant.variant
        row["score_mode"] = variant.score_mode
        row["history_size"] = variant.history_size
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--static-hot-trace", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--expert-bytes", type=float, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/results/wrec/phase5"))
    parser.add_argument("--fig-dir", type=Path, default=Path("workspace/figures/wrec/phase5_20260503"))
    parser.add_argument("--date", default="20260503")
    args = parser.parse_args()

    refs, metadata = load_event_trace(args.trace)
    static_refs, _ = load_event_trace(args.static_hot_trace)
    expert_bytes = infer_expert_bytes(args.model_path, args.dtype, args.expert_bytes)
    # Phase 5 first-pass sweeps use three representative budgets. The full
    # five-point budget sensitivity is already covered by the main total-budget
    # table and is referenced in the markdown notes.
    fractions = [0.25, 0.375, 0.5]
    measured_bw = 41.37220609315469
    stats_cache: dict[tuple[int, float, str], WrecStats] = {}

    ablation_variants = [
        Variant("ablation", "lru_baseline", "lru"),
        Variant("ablation", "full_wrec_h2", "wrec_h2"),
        Variant("ablation", "wrec_h_recent_only", "wrec_h", request_weight=0.0, cross_layer_weight=0.0),
        Variant("ablation", "no_workload_term", "wrec_h2", score_mode="no_workload"),
        Variant("ablation", "no_transfer_term", "wrec_h2", score_mode="no_transfer"),
        Variant("ablation", "no_recent_signal", "wrec_h2", recent_weight=0.0),
        Variant("ablation", "no_request_signal", "wrec_h2", request_weight=0.0),
        Variant("ablation", "no_cross_layer_signal", "wrec_h2", cross_layer_weight=0.0),
        Variant("ablation", "train_window_only", "wrec_h2", recent_weight=0.0, request_weight=0.0, cross_layer_weight=0.0),
    ]
    ablation_rows = []
    for variant in ablation_variants:
        ablation_rows.extend(
            run_variant(
                refs=refs,
                metadata=metadata,
                static_refs=static_refs,
                stats_cache=stats_cache,
                variant=variant,
                fractions=fractions,
                expert_bytes=expert_bytes,
            )
        )
    add_oracle_gap(ablation_rows)
    ablation_rows = enrich_with_gains(ablation_rows)

    sensitivity_variants = [Variant("bandwidth", "lru_baseline", "lru", bandwidth_gbps=bw) for bw in [8.0, 16.0, 32.0, measured_bw]]
    sensitivity_variants += [Variant("bandwidth", f"full_wrec_h2_bw_{bw:g}", "wrec_h2", bandwidth_gbps=bw) for bw in [8.0, 16.0, 32.0, measured_bw]]
    sensitivity_variants += [Variant("window", "lru_baseline", "lru")]
    sensitivity_variants += [Variant("window", f"full_wrec_h2_window_{window}", "wrec_h2", window_size=window) for window in [1, 4, 8, 16, 32]]
    sensitivity_variants += [Variant("history", "lru_baseline", "lru")]
    sensitivity_variants += [Variant("history", f"full_wrec_h2_history_{history}", "wrec_h2", history_size=history) for history in [1, 4, 8, 16, 32]]

    sensitivity_rows = []
    for variant in sensitivity_variants:
        sensitivity_rows.extend(
            run_variant(
                refs=refs,
                metadata=metadata,
                static_refs=static_refs,
                stats_cache=stats_cache,
                variant=variant,
                fractions=fractions,
                expert_bytes=expert_bytes,
            )
        )
    add_oracle_gap(sensitivity_rows)
    sensitivity_rows = enrich_with_gains(sensitivity_rows)

    ablation_csv = args.output_dir / f"wrec_ablation_mixtral8x7b_dolly_{args.date}.csv"
    sensitivity_csv = args.output_dir / f"wrec_sensitivity_mixtral8x7b_dolly_{args.date}.csv"
    write_csv(ablation_csv, ablation_rows)
    write_csv(sensitivity_csv, sensitivity_rows)
    (args.output_dir / f"wrec_phase5_metadata_{args.date}.json").write_text(
        json.dumps(
            {
                "trace": str(args.trace),
                "static_hot_trace": str(args.static_hot_trace),
                "expert_bytes": expert_bytes,
                "fractions": fractions,
                "ablation_rows": len(ablation_rows),
                "sensitivity_rows": len(sensitivity_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / f"wrec_ablation_mixtral8x7b_dolly_{args.date}.md",
        title="WREC Phase 5 Ablation",
        rows=ablation_rows,
        group_key="variant",
    )
    write_markdown(
        args.output_dir / f"wrec_sensitivity_mixtral8x7b_dolly_{args.date}.md",
        title="WREC Phase 5 Sensitivity",
        rows=sensitivity_rows,
        group_key="variant",
    )

    ablation_values = []
    for variant, rows in grouped(ablation_rows, "variant").items():
        if variant == "lru_baseline":
            continue
        ablation_values.append((variant, 100.0 * average([float(row["gain_vs_lru"]) for row in rows])))
    svg_bar(
        args.fig_dir / f"wrec_ablation_bar_{args.date}.svg",
        title="WREC Ablation Average Gain vs LRU",
        values=ablation_values,
        y_label="avg gain vs LRU (%)",
    )

    bw_points = []
    for bw in [8.0, 16.0, 32.0, measured_bw]:
        rows = [row for row in sensitivity_rows if row["variant"] == f"full_wrec_h2_bw_{bw:g}"]
        bw_points.append((bw, average([float(row["saved_ms_per_input_token_vs_lru"]) for row in rows])))
    svg_line(
        args.fig_dir / f"wrec_bandwidth_sensitivity_{args.date}.svg",
        title="WREC-H2 Saved Stall vs Bandwidth",
        rows=bw_points,
        x_label="bandwidth GB/s",
        y_label="avg saved ms / input token",
    )

    window_points = []
    for window in [1, 4, 8, 16, 32]:
        rows = [row for row in sensitivity_rows if row["variant"] == f"full_wrec_h2_window_{window}"]
        window_points.append((float(window), 100.0 * average([float(row["gain_vs_lru"]) for row in rows])))
    svg_line(
        args.fig_dir / f"wrec_window_sensitivity_{args.date}.svg",
        title="WREC-H2 Gain vs Train Window",
        rows=window_points,
        x_label="train window size",
        y_label="avg gain vs LRU (%)",
    )

    history_points = []
    for history in [1, 4, 8, 16, 32]:
        rows = [row for row in sensitivity_rows if row["variant"] == f"full_wrec_h2_history_{history}"]
        history_points.append((float(history), 100.0 * average([float(row["gain_vs_lru"]) for row in rows])))
    svg_line(
        args.fig_dir / f"wrec_history_sensitivity_{args.date}.svg",
        title="WREC-H2 Gain vs Online History",
        rows=history_points,
        x_label="history size",
        y_label="avg gain vs LRU (%)",
    )
    print(ablation_csv)
    print(sensitivity_csv)
    print(args.fig_dir)


if __name__ == "__main__":
    main()
