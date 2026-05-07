#!/usr/bin/env python3
"""Build a lightweight WREC scale-out appendix from config-only HRM screens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from estimate_moe_hrm_bottleneck import build_rows, infer_model_profile, read_json


MODELS = [
    {
        "label": "Mixtral-8x7B",
        "repo": "local",
        "config_path": "/root/workspace/models/Mixtral-8x7B-Instruct-v0.1/config.json",
        "proxy_note": "official local config",
    },
    {
        "label": "Mixtral-8x22B",
        "repo": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "config_path": "/root/workspace/models/hf_config_cache/models--mistralai--Mixtral-8x22B-Instruct-v0.1/snapshots/cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d/config.json",
        "proxy_note": "official Hugging Face config",
    },
    {
        "label": "DBRX",
        "repo": "alpindale/dbrx-instruct",
        "config_path": "/root/workspace/models/hf_config_cache/models--alpindale--dbrx-instruct/snapshots/8007650525bf3b67d6a4763caf02230061452d45/config.json",
        "proxy_note": "public converted config proxy, not official Databricks repo",
    },
    {
        "label": "DeepSeek-MoE-16B",
        "repo": "deepseek-ai/deepseek-moe-16b-chat",
        "config_path": "/root/workspace/models/hf_config_cache/models--deepseek-ai--deepseek-moe-16b-chat/snapshots/eefd8ac7e8dc90e095129fe1a537d5e236b2e57c/config.json",
        "proxy_note": "official Hugging Face config",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_grouped_bars(
    path: Path,
    *,
    title: str,
    models: list[str],
    budgets: list[float],
    values: dict[tuple[str, float], float],
    y_label: str,
    percent: bool = False,
) -> None:
    width, height = 1040, 520
    left, right, top, bottom = 90, 40, 70, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max([values[(model, budget)] for model in models for budget in budgets] + [1e-6])
    group_w = plot_w / max(1, len(models))
    bar_w = group_w / (len(budgets) + 1)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = top + plot_h * (1.0 - value / max_value)
        label = f"{100*value:.0f}%" if percent else f"{value:.2f}"
        parts.append(f'<line x1="{left-5}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')

    for m_idx, model in enumerate(models):
        gx = left + m_idx * group_w
        parts.append(f'<text x="{gx + group_w/2:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{model}</text>')
        for b_idx, budget in enumerate(budgets):
            value = values[(model, budget)]
            h = plot_h * value / max_value if max_value > 0 else 0.0
            x = gx + (b_idx + 0.5) * bar_w
            y = top + plot_h - h
            color = colors[b_idx % len(colors)]
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w*0.8:.2f}" height="{h:.2f}" fill="{color}"/>')
            text = f"{100*value:.0f}%" if percent else f"{value:.2f}"
            parts.append(f'<text x="{x + bar_w*0.4:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="10">{text}</text>')

    legend_x = width - 170
    for idx, budget in enumerate(budgets):
        y = 86 + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y-12}" width="18" height="12" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{y-2}" font-family="Arial" font-size="12">{budget:g} GB GPU</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_markdown(path: Path, rows: list[dict[str, Any]], assumptions: dict[str, Any]) -> None:
    lines = [
        "# WREC Scale-Out Appendix",
        "",
        "## Setup",
        "",
        f"- dtype: `{assumptions['dtype']}`",
        f"- measured CPU-GPU bandwidth: `{assumptions['bandwidth_gbps']}` GB/s",
        f"- KV reservation: `{assumptions['kv_cache_budget_gb']}` GB",
        f"- active decode tokens: `{assumptions['decode_active_tokens']}`",
        "- resident fraction request used for transfer-pressure comparison: `1.0`",
        "- DBRX uses a public converted config proxy because an official Databricks HF repo was not available in this environment.",
        "",
        "## Summary Table",
        "",
        "| model | repo/config source | model total GiB | expert total GiB | GPU budget GiB | max feasible resident fraction | transfer ratio at max resident | all-resident feasible |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_label"],
                    row["repo_source"],
                    f"{float(row['model_total_gb']):.2f}",
                    f"{float(row['expert_total_gb']):.2f}",
                    f"{float(row['gpu_memory_gb']):.0f}",
                    f"{100*float(row['max_effective_resident_fraction']):.1f}%",
                    f"{100*float(row['transfer_ratio_at_max_resident']):.1f}%",
                    str(row["all_resident_feasible"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- This appendix does not replace the Mixtral-8x7B main experiment. It only broadens the motivation by showing that larger MoE models become harder to keep resident under the same memory budget.",
            "- `transfer_ratio_at_max_resident` should be read as a best-case under that GPU budget, because it assumes the cache is filled up to the model's maximum feasible resident fraction.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--bandwidth-gbps", type=float, default=41.37220609315469)
    parser.add_argument("--gpu-memory-gb", default="24,48,96")
    parser.add_argument("--kv-cache-budget-gb", type=float, default=8.0)
    parser.add_argument("--decode-active-tokens", type=int, default=32)
    parser.add_argument("--gpu-tflops", type=float, default=382.45478034491344)
    args = parser.parse_args()

    budgets = [float(x.strip()) for x in args.gpu_memory_gb.split(",") if x.strip()]
    summary_rows: list[dict[str, Any]] = []
    resident_values: dict[tuple[str, float], float] = {}
    transfer_values: dict[tuple[str, float], float] = {}

    for model in MODELS:
        config = read_json(Path(model["config_path"]))
        profile = infer_model_profile(config)
        rows = build_rows(
            model_name=model["label"],
            profile=profile,
            dtype_names=["bf16"],
            gpu_memory_gb=budgets,
            kv_cache_budget_gb=[args.kv_cache_budget_gb],
            bandwidth_gbps=[args.bandwidth_gbps],
            resident_fractions=[1.0],
            active_tokens=[args.decode_active_tokens],
            gpu_tflops=args.gpu_tflops,
            transfer_threshold=0.30,
        )
        for row in rows:
            budget = float(row["gpu_memory_gb"])
            max_resident = float(row["effective_resident_fraction"])
            transfer_ratio = float(row["expert_transfer_ratio"])
            resident_values[(model["label"], budget)] = max_resident
            transfer_values[(model["label"], budget)] = transfer_ratio
            summary_rows.append(
                {
                    "model_label": model["label"],
                    "repo_source": model["repo"],
                    "proxy_note": model["proxy_note"],
                    "gpu_memory_gb": budget,
                    "kv_cache_budget_gb": float(row["kv_cache_budget_gb"]),
                    "bandwidth_gbps": float(row["bandwidth_gbps"]),
                    "decode_active_tokens": int(row["decode_active_tokens"]),
                    "model_total_gb": float(row["model_total_gb"]),
                    "expert_total_gb": float(row["expert_total_gb"]),
                    "max_effective_resident_fraction": max_resident,
                    "transfer_ratio_at_max_resident": transfer_ratio,
                    "all_resident_feasible": row["all_experts_resident_feasible"],
                }
            )

    write_csv(args.output_csv, summary_rows)
    build_markdown(
        args.output_md,
        summary_rows,
        {
            "dtype": "bf16",
            "bandwidth_gbps": args.bandwidth_gbps,
            "kv_cache_budget_gb": args.kv_cache_budget_gb,
            "decode_active_tokens": args.decode_active_tokens,
        },
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    model_labels = [model["label"] for model in MODELS]
    svg_grouped_bars(
        args.fig_dir / "scaleout_resident_fraction_20260503.svg",
        title="Max Feasible Expert Resident Fraction at Fixed GPU Budget",
        models=model_labels,
        budgets=budgets,
        values=resident_values,
        y_label="max effective resident fraction",
        percent=True,
    )
    svg_grouped_bars(
        args.fig_dir / "scaleout_expert_transfer_ratio_20260503.svg",
        title="Transfer Pressure at Max Resident Fraction",
        models=model_labels,
        budgets=budgets,
        values=transfer_values,
        y_label="expert transfer ratio",
        percent=True,
    )
    print(args.output_csv)
    print(args.output_md)
    print(args.fig_dir / "scaleout_resident_fraction_20260503.svg")
    print(args.fig_dir / "scaleout_expert_transfer_ratio_20260503.svg")


if __name__ == "__main__":
    main()
