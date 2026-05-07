# Mixtral Fair Cache Baselines

## Inputs

- Train trace for `static_hot`: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Bandwidth: `41.37220609315469` GB/s
- Cache capacities: `1,2,4` experts/layer
- Policies: `on_demand`, `lru`, `static_hot`, `belady_oracle`

## Main Results

| cache | policy | miss rate | hit rate | stall ms/input token | oracle gap ratio |
|---:|---|---:|---:|---:|---:|
| 1 | on_demand | 1.000000 | 0.000000 | 545.0175 | 0.3241 |
| 1 | lru | 0.921096 | 0.078904 | 502.0137 | 0.2662 |
| 1 | static_hot | 0.837102 | 0.162898 | 456.2354 | 0.1926 |
| 1 | belady_oracle | 0.675870 | 0.324130 | 368.3609 | 0.0000 |
| 2 | on_demand | 1.000000 | 0.000000 | 545.0175 | 0.5387 |
| 2 | lru | 0.692906 | 0.307094 | 377.6461 | 0.3342 |
| 2 | static_hot | 0.691349 | 0.308651 | 376.7974 | 0.3327 |
| 2 | belady_oracle | 0.461339 | 0.538661 | 251.4378 | 0.0000 |
| 4 | on_demand | 1.000000 | 0.000000 | 545.0175 | 0.7806 |
| 4 | lru | 0.418951 | 0.581049 | 228.3359 | 0.4763 |
| 4 | static_hot | 0.429235 | 0.570765 | 233.9407 | 0.4888 |
| 4 | belady_oracle | 0.219416 | 0.780584 | 119.5856 | 0.0000 |

## Notes

- `static_hot` uses only train trace frequency.
- `belady_oracle` uses eval future information for initial cache and admission/bypass, so it is an upper bound rather than a deployable policy.
- At capacity 1, `static_hot` is better than LRU; at capacity 4, LRU is slightly better than `static_hot`.
