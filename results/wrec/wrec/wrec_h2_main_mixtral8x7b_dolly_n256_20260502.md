# WREC-H2 Main Result

## Setup

- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train/reference trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Bandwidth: `41.37220609315469` GB/s
- WREC-H2 signals:
  - train global/window hotness
  - online recent-history
  - current-request layer-local frequency
  - cross-layer transition from previous-layer experts
- WREC-H2 params: `window=4`, `history=8`, `recent_weight=512`, `request_weight=1024`, `cross_layer_weight=1024`, `prefetch_queue=0`

## Results

| cache | policy | miss rate | stall ms/token | gain vs LRU | gain vs static_hot | waste bytes/token |
|---:|---|---:|---:|---:|---:|---:|
| 1 | lru | 0.921096 | 502.0137 | 0.00% | -10.03% | 0.00 |
| 1 | static_hot | 0.837102 | 456.2354 | 9.12% | 0.00% | 0.00 |
| 1 | route_window_prefetch | 0.860722 | 469.1088 | 6.55% | -2.82% | 19183854090.89 |
| 1 | wrec_h | 0.838857 | 457.1919 | 8.93% | -0.21% | 0.00 |
| 1 | wrec_h2 | 0.835624 | 455.4300 | 9.28% | 0.18% | 0.00 |
| 1 | belady_oracle | 0.675870 | 368.3609 | 26.62% | 19.26% | 0.00 |
| 2 | lru | 0.692906 | 377.6461 | 0.00% | -0.23% | 0.00 |
| 2 | static_hot | 0.691349 | 376.7974 | 0.22% | 0.00% | 0.00 |
| 2 | route_window_prefetch | 0.719147 | 391.9475 | -3.79% | -4.02% | 17147327528.03 |
| 2 | wrec_h | 0.676757 | 368.8443 | 2.33% | 2.11% | 0.00 |
| 2 | wrec_h2 | 0.671588 | 366.0272 | 3.08% | 2.86% | 0.00 |
| 2 | belady_oracle | 0.461339 | 251.4378 | 33.42% | 33.27% | 0.00 |
| 4 | lru | 0.418951 | 228.3359 | 0.00% | 2.40% | 0.00 |
| 4 | static_hot | 0.429235 | 233.9407 | -2.45% | 0.00% | 0.00 |
| 4 | route_window_prefetch | 0.264134 | 143.9576 | 36.95% | 38.46% | 6039797760.00 |
| 4 | wrec_h | 0.405204 | 220.8433 | 3.28% | 5.60% | 0.00 |
| 4 | wrec_h2 | 0.402584 | 219.4155 | 3.91% | 6.21% | 0.00 |
| 4 | belady_oracle | 0.219416 | 119.5856 | 47.63% | 48.88% | 0.00 |

## Conclusion

WREC-H2 improves WREC-H v1 and beats `static_hot` at all tested cache capacities without prefetch waste. It still does not meet the planned `>= 10%` stall reduction versus LRU. The remaining gap suggests that the next method change should be a constrained keep/prefetch planner rather than additional scalar weight tuning.
