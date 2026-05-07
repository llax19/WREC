# WREC-H Main Result

## Setup

- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train/reference trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Bandwidth: `41.37220609315469` GB/s
- Expert bytes: `352321536`
- WREC-H: train hotness prior + online recent-history admission/eviction
- WREC-H params: `window=4`, `history=8`, `recent_weight=512`, `prefetch_queue=0`
- Route-window baseline params: `window=4`, `prefetch_queue=1`

## Results

| cache | policy | miss rate | stall ms/token | gain vs LRU | prefetch loads | waste bytes/token |
|---:|---|---:|---:|---:|---:|---:|
| 1 | on_demand | 1.000000 | 545.0175 | -8.57% | 0 | 0.00 |
| 1 | lru | 0.921096 | 502.0137 | 0.00% | 0 | 0.00 |
| 1 | static_hot | 0.837102 | 456.2354 | 9.12% | 0 | 0.00 |
| 1 | route_window_prefetch | 0.860722 | 469.1088 | 6.55% | 896405 | 19183854090.89 |
| 1 | wrec_h | 0.838857 | 457.1919 | 8.93% | 0 | 0.00 |
| 1 | belady_oracle | 0.675870 | 368.3609 | 26.62% | 0 | 0.00 |
| 2 | on_demand | 1.000000 | 545.0175 | -44.32% | 0 | 0.00 |
| 2 | lru | 0.692906 | 377.6461 | 0.00% | 0 | 0.00 |
| 2 | static_hot | 0.691349 | 376.7974 | 0.22% | 0 | 0.00 |
| 2 | route_window_prefetch | 0.719147 | 391.9475 | -3.79% | 867106 | 17147327528.03 |
| 2 | wrec_h | 0.676757 | 368.8443 | 2.33% | 0 | 0.00 |
| 2 | belady_oracle | 0.461339 | 251.4378 | 33.42% | 0 | 0.00 |
| 4 | on_demand | 1.000000 | 545.0175 | -138.69% | 0 | 0.00 |
| 4 | lru | 0.418951 | 228.3359 | 0.00% | 0 | 0.00 |
| 4 | static_hot | 0.429235 | 233.9407 | -2.45% | 0 | 0.00 |
| 4 | route_window_prefetch | 0.264134 | 143.9576 | 36.95% | 558804 | 6039797760.00 |
| 4 | wrec_h | 0.405204 | 220.8433 | 3.28% | 0 | 0.00 |
| 4 | belady_oracle | 0.219416 | 119.5856 | 47.63% | 0 | 0.00 |

## Conclusion

WREC-H v1 is implemented and produces a measurable improvement over LRU at cache capacities `2` and `4`, but it does not meet the planned success bar of `>= 10%` stall reduction versus LRU. The prefetch-enabled variant was worse because it introduced large waste. The next step should be to revise the WREC score or move to a constrained WREC-C planner rather than training a predictor.
