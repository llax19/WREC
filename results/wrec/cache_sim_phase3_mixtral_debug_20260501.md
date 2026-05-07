# Phase 3 simulator on Mixtral debug trace - 2026-05-01

## Scope

This run replays the existing Mixtral debug trace:

- trace: `logs/processed/wrec/mixtral8x7b_dolly_debug_router_events_n64_mem48_20260501.jsonl`
- requests: `64`
- total input tokens: `3347`
- total router events: `107104`

The simulator sweep covered:

- bandwidth: `8`, `16`, `32`, `41.37220609315469` GB/s
- cache capacities: `1`, `2`, `4`, `25%`, `50%`
- policies:
  - `on_demand`
  - `lru`
  - `static_hot`
  - `belady_oracle`
  - `route_window_prefetch`

Note: because Mixtral has `8` experts per layer, `25% -> 2` and `50% -> 4`, so those capacities duplicate `2` and `4`.

## Output files

- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw8_20260501.json`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw8_20260501.csv`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw16_20260501.json`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw16_20260501.csv`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw32_20260501.json`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw32_20260501.csv`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw41p37_20260501.json`
- `results/wrec/cache_sim_baselines_mixtral8x7b_debug_n64_bw41p37_20260501.csv`

## Key findings at measured bandwidth

Measured bandwidth run: `41.37220609315469 GB/s`

Unique capacities:

| cache/layer | policy | stall ms/token | miss rate | transfer bytes/token | improvement vs on_demand |
|---:|---|---:|---:|---:|---:|
| 1 | on_demand | 545.0175 | 1.0000 | 22548578304.00 | 0.0000 |
| 1 | lru | 501.7663 | 0.9206 | 20759180697.05 | 0.0794 |
| 1 | static_hot | 455.1617 | 0.8351 | 18831043985.09 | 0.1649 |
| 1 | belady_oracle | 501.7663 | 0.9206 | 20759180697.05 | 0.0794 |
| 2 | on_demand | 545.0175 | 1.0000 | 22548578304.00 | 0.0000 |
| 2 | lru | 376.0201 | 0.6899 | 15556780054.03 | 0.3101 |
| 2 | static_hot | 374.3713 | 0.6869 | 15488568415.15 | 0.3131 |
| 2 | belady_oracle | 296.3085 | 0.5437 | 12258936791.00 | 0.4563 |
| 4 | on_demand | 545.0175 | 1.0000 | 22548578304.00 | 0.0000 |
| 4 | lru | 225.9118 | 0.4145 | 9346468234.67 | 0.5855 |
| 4 | static_hot | 229.5909 | 0.4213 | 9498681243.65 | 0.5787 |
| 4 | belady_oracle | 130.7128 | 0.2398 | 5407877678.66 | 0.7602 |

Belady relative to LRU:

- cache `1`: `0.0%`
- cache `2`: `21.2%`
- cache `4`: `42.14%`

LRU relative to on-demand:

- cache `1`: `7.94%`
- cache `2`: `31.01%`
- cache `4`: `58.55%`

Static-hot relative to LRU:

- cache `1`: `+9.29%`
- cache `2`: `+0.44%`
- cache `4`: `-1.63%`

## Interpretation

The debug trace already shows meaningful locality once cache capacity reaches `2` or `4` experts per layer:

- Belady beats LRU by more than `20%` at capacities `2` and `4`
- LRU beats on-demand clearly at all three unique capacities

That means the Phase 3 gate is provisionally satisfied on the debug trace:

- there is a real oracle gap
- at least one deployable baseline converts locality into lower transfer/stall

## Caveats

This is not yet the final paper-quality baseline table.

1. `static_hot` currently uses the same debug trace as its reference trace.
   This is convenient for debug, but it leaks evaluation information and is optimistic.

2. `route_window_prefetch` in the current script uses future accesses inside the replay trace.
   That makes it an oracle-style upper bound, not a fair online baseline.

3. `belady_oracle` here is an eviction oracle under the current simulator model.
   It is not a universal upper bound over all future-aware prefetch policies.

4. The results are from the debug split, not the final train/eval split.

## Conclusion

The Mixtral debug trace is already sufficient to justify continuing the WREC pipeline. The next sensible step is to keep Phase 3 moving using this debug trace as a development harness, while preparing the larger Mixtral traces needed for fair `static_hot` and final eval measurements.

