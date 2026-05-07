# WREC Phase 5 Findings

## Setup

- Eval trace: `workspace/logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train trace: `workspace/logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Representative budgets: total slots `64/96/128`
- Full total-budget sweep remains in `workspace/results/wrec/wrec_total_budget/wrec_total_budget_mixtral8x7b_dolly_n256_20260502.md`.

## Main conclusions

- Full WREC-H2 average gain vs LRU over `64/96/128` slots is `77.80%`.
- `no_recent_signal` is slightly stronger than full WREC-H2: average gain `79.40%`.
- Removing cross-layer signal hurts clearly: average gain drops to `68.54%`.
- WREC-H / recent-only is much weaker: average gain `43.51%`.
- Train-window-only nearly collapses to LRU: average gain `-0.93%`.
- `no_workload_term` and `no_transfer_term` are nearly identical to full WREC-H2 in this setting, because request/cross-layer weights dominate and all experts have equal transfer size.

## Sensitivity

- Bandwidth sensitivity matches the transfer-bound story in absolute terms: average saved stall is about `1198.96 / 601.54 / 301.26 / 233.04 ms/input-token` at `8 / 16 / 32 / 41.37 GB/s`.
- Relative gain stays near `77-78%` because LRU and WREC demand-transfer stalls scale together with bandwidth.
- Window sensitivity is flat for `1/4/8/16/32`, with average gain between `77.15%` and `77.85%`.
- History sensitivity is mild; history `8` is best among tested values, but history `1/16/32` remain close.

## Interpretation

- Current WREC-H2 should be described as request/cross-layer-driven, not as strongly dependent on train-window workload score.
- Recent history is not a reliable positive term for the current total-budget prefill replay; either set it to zero in the next main-table refresh or report it as a negative ablation.
- Transfer-aware score ablation is inconclusive under homogeneous expert size. A stronger transfer ablation needs heterogeneous expert sizes, layer-specific bandwidth, or explicit prefetch overlap.

## Limitations

- This Phase 5 pass uses three representative budgets to control runtime. The full five-budget sensitivity is already covered by the main total-budget result.
- WREC-H2 currently has no active prefetch path, so prefetch-only ablation is not emitted.
- WREC-C comparison should use the existing `wrec_c_total_budget_*` and `wrec_c_prefetch_*` results; the constrained-prefetch version is a negative result.
