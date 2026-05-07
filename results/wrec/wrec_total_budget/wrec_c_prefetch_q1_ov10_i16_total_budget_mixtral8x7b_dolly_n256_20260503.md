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
| 32 | 12.50% | lru | uniform_per_layer | 0.921096 | 502.0137 | 0.00% | 0.00 | 0.9302 |
| 32 | 12.50% | static_hot | global_static_hot | 0.834057 | 454.5758 | 9.45% | 0.00 | 0.9229 |
| 32 | 12.50% | wrec_h2 | global_wrec_adaptive | 0.303074 | 165.1809 | 67.10% | 0.00 | 0.7878 |
| 32 | 12.50% | wrec_c | global_wrec_constrained | 0.353201 | 192.5007 | 61.65% | 664024112.13 | 0.8180 |
| 32 | 12.50% | belady_oracle | global_belady | 0.064298 | 35.0436 | 93.02% | 0.00 | 0.0000 |
| 64 | 25.00% | lru | uniform_per_layer | 0.692906 | 377.6461 | 0.00% | 0.00 | 0.9207 |
| 64 | 25.00% | static_hot | global_static_hot | 0.687638 | 374.7748 | 0.76% | 0.00 | 0.9201 |
| 64 | 25.00% | wrec_h2 | global_wrec_adaptive | 0.183682 | 100.1098 | 73.49% | 0.00 | 0.7010 |
| 64 | 25.00% | wrec_c | global_wrec_constrained | 0.215075 | 117.2199 | 68.96% | 618299764.92 | 0.7446 |
| 64 | 25.00% | belady_oracle | global_belady | 0.054920 | 29.9324 | 92.07% | 0.00 | 0.0000 |
| 96 | 37.50% | lru | uniform_per_layer | 0.550089 | 299.8082 | 0.00% | 0.00 | 0.9171 |
| 96 | 37.50% | static_hot | global_static_hot | 0.554380 | 302.1468 | -0.78% | 0.00 | 0.9178 |
| 96 | 37.50% | wrec_h2 | global_wrec_adaptive | 0.115621 | 63.0154 | 78.98% | 0.00 | 0.6058 |
| 96 | 37.50% | wrec_c | global_wrec_constrained | 0.138123 | 75.2797 | 74.89% | 566224813.92 | 0.6700 |
| 96 | 37.50% | belady_oracle | global_belady | 0.045575 | 24.8392 | 91.71% | 0.00 | 0.0000 |
| 128 | 50.00% | lru | uniform_per_layer | 0.418951 | 228.3359 | 0.00% | 0.00 | 0.9135 |
| 128 | 50.00% | static_hot | global_static_hot | 0.427522 | 233.0071 | -2.05% | 0.00 | 0.9153 |
| 128 | 50.00% | wrec_h2 | global_wrec_adaptive | 0.079921 | 43.5583 | 80.92% | 0.00 | 0.5466 |
| 128 | 50.00% | wrec_c | global_wrec_constrained | 0.093044 | 50.7107 | 77.79% | 494674677.99 | 0.6106 |
| 128 | 50.00% | belady_oracle | global_belady | 0.036232 | 19.7472 | 91.35% | 0.00 | 0.0000 |
| 192 | 75.00% | lru | uniform_per_layer | 0.188138 | 102.5387 | 0.00% | 0.00 | 0.9065 |
| 192 | 75.00% | static_hot | global_static_hot | 0.192508 | 104.9201 | -2.32% | 0.00 | 0.9086 |
| 192 | 75.00% | wrec_h2 | global_wrec_adaptive | 0.043450 | 23.6810 | 76.91% | 0.00 | 0.5952 |
| 192 | 75.00% | wrec_c | global_wrec_constrained | 0.045024 | 24.5388 | 76.07% | 300271489.33 | 0.6094 |
| 192 | 75.00% | belady_oracle | global_belady | 0.017587 | 9.5850 | 90.65% | 0.00 | 0.0000 |

## Summary

- Fixed total-budget replay changes the conclusion materially: WREC adaptive allocation is much stronger than uniform per-layer LRU.
- WREC-H2 beats LRU and static-hot at all tested total budgets with zero prefetch waste.
- WREC-C is reported when present as the constrained total-budget planner variant.
- Route-window prefetch can reduce stall at larger budgets but has very large waste in this oracle-style configuration.
- Belady remains far ahead, so there is still meaningful oracle gap for constrained planning.
