#!/usr/bin/env python3
"""Build publication-style WREC figures from fixed total-budget replay CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # Registers SciencePlots styles.


POLICY_LABELS = {
    "lru": "LRU",
    "static_hot": "Static-hot",
    "route_window_prefetch": "Window prefetch",
    "wrec_h2": "WREC-H2",
    "belady_oracle": "Belady oracle",
}

POLICY_COLORS = {
    "lru": "#4C78A8",
    "static_hot": "#F58518",
    "route_window_prefetch": "#E45756",
    "wrec_h2": "#2F855A",
    "belady_oracle": "#6F4E7C",
}

POLICY_MARKERS = {
    "lru": "o",
    "static_hot": "s",
    "route_window_prefetch": "^",
    "wrec_h2": "D",
    "belady_oracle": "P",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = {
        "total_cache_slots",
        "stall_ms_per_input_token",
        "cache_hit_rate",
        "oracle_gap_ratio",
        "waste_bytes_per_input_token",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def by_budget(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["total_cache_slots"])][str(row["policy"])] = row
    return dict(sorted(grouped.items()))


def setup_style() -> None:
    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.top": False,
            "ytick.right": False,
            "xtick.minor.top": False,
            "ytick.minor.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_stall(rows: list[dict[str, Any]], output_base: Path) -> None:
    grouped = by_budget(rows)
    slots = sorted(grouped)
    policies = ["lru", "static_hot", "route_window_prefetch", "wrec_h2", "belady_oracle"]
    fig, ax = plt.subplots(figsize=(3.55, 2.28))
    for policy in policies:
        if policy not in grouped[slots[0]]:
            continue
        values = [grouped[slot][policy]["stall_ms_per_input_token"] for slot in slots]
        ax.plot(
            slots,
            values,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
        )
    ax.set_xlabel("Total expert-cache slots")
    ax.set_ylabel("Stall proxy (ms / input token)")
    ax.set_xticks(slots)
    ax.tick_params(which="both", top=False, right=False)
    ax.legend(frameon=True, framealpha=0.92, ncol=1, loc="upper right")
    save_figure(fig, output_base)


def plot_gain(rows: list[dict[str, Any]], output_base: Path) -> None:
    grouped = by_budget(rows)
    slots = sorted(grouped)
    policies = ["static_hot", "route_window_prefetch", "wrec_h2", "belady_oracle"]
    fig, ax = plt.subplots(figsize=(3.55, 2.24))
    for policy in policies:
        if policy not in grouped[slots[0]]:
            continue
        values = []
        for slot in slots:
            lru = grouped[slot]["lru"]["stall_ms_per_input_token"]
            current = grouped[slot][policy]["stall_ms_per_input_token"]
            values.append((lru - current) / lru * 100.0)
        ax.plot(
            slots,
            values,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Total expert-cache slots")
    ax.set_ylabel("Reduction vs. LRU (%)")
    ax.set_xticks(slots)
    ax.set_ylim(bottom=min(-10, ax.get_ylim()[0]), top=100)
    ax.tick_params(which="both", top=False, right=False)
    ax.legend(frameon=True, framealpha=0.92, ncol=1, loc="lower right")
    save_figure(fig, output_base)


def plot_oracle_gap(rows: list[dict[str, Any]], output_base: Path) -> None:
    grouped = by_budget(rows)
    slots = sorted(grouped)
    policies = ["static_hot", "route_window_prefetch", "wrec_h2"]
    fig, ax = plt.subplots(figsize=(3.55, 2.24))
    for policy in policies:
        if policy not in grouped[slots[0]]:
            continue
        values = []
        for slot in slots:
            lru = grouped[slot]["lru"]["stall_ms_per_input_token"]
            oracle = grouped[slot]["belady_oracle"]["stall_ms_per_input_token"]
            current = grouped[slot][policy]["stall_ms_per_input_token"]
            denom = max(1e-12, lru - oracle)
            values.append((lru - current) / denom * 100.0)
        ax.plot(
            slots,
            values,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
        )
    ax.set_xlabel("Total expert-cache slots")
    ax.set_ylabel("Oracle gap captured (%)")
    ax.set_xticks(slots)
    ax.set_ylim(0, 105)
    ax.tick_params(which="both", top=False, right=False)
    ax.legend(frameon=True, framealpha=0.92, ncol=1, loc="lower right")
    save_figure(fig, output_base)


def plot_miss_rate(rows: list[dict[str, Any]], output_base: Path) -> None:
    grouped = by_budget(rows)
    slots = sorted(grouped)
    policies = ["lru", "static_hot", "route_window_prefetch", "wrec_h2", "belady_oracle"]
    fig, ax = plt.subplots(figsize=(3.55, 2.24))
    for policy in policies:
        if policy not in grouped[slots[0]]:
            continue
        values = [(1.0 - grouped[slot][policy]["cache_hit_rate"]) * 100.0 for slot in slots]
        ax.plot(
            slots,
            values,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
        )
    ax.set_xlabel("Total expert-cache slots")
    ax.set_ylabel("Expert miss rate (%)")
    ax.set_xticks(slots)
    ax.tick_params(which="both", top=False, right=False)
    ax.legend(frameon=True, framealpha=0.92, ncol=1, loc="upper right")
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args()

    setup_style()
    rows = read_rows(args.csv)
    plot_stall(rows, args.fig_dir / "wrec_stall_vs_budget_publication")
    plot_gain(rows, args.fig_dir / "wrec_gain_vs_lru_publication")
    plot_oracle_gap(rows, args.fig_dir / "wrec_oracle_gap_capture_publication")
    plot_miss_rate(rows, args.fig_dir / "wrec_miss_rate_publication")
    for name in [
        "wrec_stall_vs_budget_publication",
        "wrec_gain_vs_lru_publication",
        "wrec_oracle_gap_capture_publication",
        "wrec_miss_rate_publication",
    ]:
        print(args.fig_dir / f"{name}.pdf")
        print(args.fig_dir / f"{name}.svg")


if __name__ == "__main__":
    main()
