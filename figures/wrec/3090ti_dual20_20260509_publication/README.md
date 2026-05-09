# WREC publication-style figures

## Source

- CSV: `results/wrec/wrec_total_budget_3090ti_dual20_20260509/main_recent0.csv`
- Script: `scripts/moe_affinity/plot_wrec_publication_figures.py`
- Style: Matplotlib + SciencePlots, no-LaTeX mode.

## Figures

| figure | formats | purpose |
|---|---|---|
| `wrec_stall_vs_budget_publication` | PDF, SVG | Main transfer-stall proxy comparison. |
| `wrec_gain_vs_lru_publication` | PDF, SVG | WREC-H2 reduction relative to LRU. |
| `wrec_oracle_gap_capture_publication` | PDF, SVG | Fraction of LRU-to-Belady headroom captured. |
| `wrec_miss_rate_publication` | PDF, SVG | Expert-cache miss-rate comparison. |

## Notes

- PDF is preferred for LaTeX inclusion.
- SVG is kept for quick browser inspection and editing.
- Figure titles are intentionally omitted; use thesis captions for titles.
