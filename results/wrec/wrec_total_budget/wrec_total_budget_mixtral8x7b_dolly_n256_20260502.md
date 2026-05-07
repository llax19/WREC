# WREC Fixed Total Expert-Cache Budget Results

## Setup

- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train/reference trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Budget mode: fixed total expert-cache slots.
- LRU / route-window allocation: uniform per-layer.
- Static-hot allocation: global train frequency.
- WREC allocation: global adaptive admission/eviction.

## Main Table

| total slots | fraction | policy | allocation | miss rate | stall ms/token | gain vs LRU | waste bytes/token | oracle gap ratio |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 32 | 12.50% | on_demand | none | 1.000000 | 545.0175 | -8.57% | 0.00 | 0.9357 |
| 32 | 12.50% | lru | uniform_per_layer | 0.921096 | 502.0137 | 0.00% | 0.00 | 0.9302 |
| 32 | 12.50% | static_hot | global_static_hot | 0.834057 | 454.5758 | 9.45% | 0.00 | 0.9229 |
| 32 | 12.50% | route_window_prefetch | uniform_per_layer | 0.860722 | 469.1088 | 6.55% | 19183854090.89 | 0.9253 |
| 32 | 12.50% | wrec_h | global_wrec_adaptive | 0.607444 | 331.0678 | 34.05% | 0.00 | 0.8941 |
| 32 | 12.50% | wrec_h2 | global_wrec_adaptive | 0.303074 | 165.1809 | 67.10% | 0.00 | 0.7878 |
| 32 | 12.50% | belady_oracle | global_belady | 0.064298 | 35.0436 | 93.02% | 0.00 | 0.0000 |
| 64 | 25.00% | on_demand | none | 1.000000 | 545.0175 | -44.32% | 0.00 | 0.9451 |
| 64 | 25.00% | lru | uniform_per_layer | 0.692906 | 377.6461 | 0.00% | 0.00 | 0.9207 |
| 64 | 25.00% | static_hot | global_static_hot | 0.687638 | 374.7748 | 0.76% | 0.00 | 0.9201 |
| 64 | 25.00% | route_window_prefetch | uniform_per_layer | 0.746905 | 407.0765 | -7.79% | 17856777135.44 | 0.9265 |
| 64 | 25.00% | wrec_h | global_wrec_adaptive | 0.460854 | 251.1736 | 33.49% | 0.00 | 0.8808 |
| 64 | 25.00% | wrec_h2 | global_wrec_adaptive | 0.183682 | 100.1098 | 73.49% | 0.00 | 0.7010 |
| 64 | 25.00% | belady_oracle | global_belady | 0.054920 | 29.9324 | 92.07% | 0.00 | 0.0000 |
| 96 | 37.50% | on_demand | none | 1.000000 | 545.0175 | -81.79% | 0.00 | 0.9544 |
| 96 | 37.50% | lru | uniform_per_layer | 0.550089 | 299.8082 | 0.00% | 0.00 | 0.9171 |
| 96 | 37.50% | static_hot | global_static_hot | 0.554380 | 302.1468 | -0.78% | 0.00 | 0.9178 |
| 96 | 37.50% | route_window_prefetch | uniform_per_layer | 0.404581 | 220.5038 | 26.45% | 9845104449.49 | 0.8874 |
| 96 | 37.50% | wrec_h | global_wrec_adaptive | 0.321261 | 175.0927 | 41.60% | 0.00 | 0.8581 |
| 96 | 37.50% | wrec_h2 | global_wrec_adaptive | 0.115621 | 63.0154 | 78.98% | 0.00 | 0.6058 |
| 96 | 37.50% | belady_oracle | global_belady | 0.045575 | 24.8392 | 91.71% | 0.00 | 0.0000 |
| 128 | 50.00% | on_demand | none | 1.000000 | 545.0175 | -138.69% | 0.00 | 0.9638 |
| 128 | 50.00% | lru | uniform_per_layer | 0.418951 | 228.3359 | 0.00% | 0.00 | 0.9135 |
| 128 | 50.00% | static_hot | global_static_hot | 0.427522 | 233.0071 | -2.05% | 0.00 | 0.9153 |
| 128 | 50.00% | route_window_prefetch | uniform_per_layer | 0.258527 | 140.9015 | 38.29% | 6575066493.53 | 0.8599 |
| 128 | 50.00% | wrec_h | global_wrec_adaptive | 0.186662 | 101.7339 | 55.45% | 0.00 | 0.8059 |
| 128 | 50.00% | wrec_h2 | global_wrec_adaptive | 0.079921 | 43.5583 | 80.92% | 0.00 | 0.5466 |
| 128 | 50.00% | belady_oracle | global_belady | 0.036232 | 19.7472 | 91.35% | 0.00 | 0.0000 |
| 192 | 75.00% | on_demand | none | 1.000000 | 545.0175 | -431.52% | 0.00 | 0.9824 |
| 192 | 75.00% | lru | uniform_per_layer | 0.188138 | 102.5387 | 0.00% | 0.00 | 0.9065 |
| 192 | 75.00% | static_hot | global_static_hot | 0.192508 | 104.9201 | -2.32% | 0.00 | 0.9086 |
| 192 | 75.00% | route_window_prefetch | uniform_per_layer | 0.060614 | 33.0355 | 67.78% | 397522892.32 | 0.7099 |
| 192 | 75.00% | wrec_h | global_wrec_adaptive | 0.033603 | 18.3140 | 82.14% | 0.00 | 0.4766 |
| 192 | 75.00% | wrec_h2 | global_wrec_adaptive | 0.043450 | 23.6810 | 76.91% | 0.00 | 0.5952 |
| 192 | 75.00% | belady_oracle | global_belady | 0.017587 | 9.5850 | 90.65% | 0.00 | 0.0000 |

## Summary

- Fixed total-budget replay changes the conclusion materially: WREC adaptive allocation is much stronger than uniform per-layer LRU.
- WREC-H2 beats LRU and static-hot at all tested total budgets with zero prefetch waste.
- Route-window prefetch can reduce stall at larger budgets but has very large waste in this oracle-style configuration.
- Belady remains far ahead, so there is still meaningful oracle gap for constrained planning.
