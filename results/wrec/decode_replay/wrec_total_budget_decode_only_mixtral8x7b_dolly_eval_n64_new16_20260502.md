# WREC Decode-Only Fixed Total Expert-Cache Budget Replay

## Setup

- Eval trace: `logs/processed/wrec/decode_eval/mixtral8x7b_dolly_eval_decode_only_n64_new16_mem48_20260502.jsonl`
- Source trace: `logs/processed/wrec/decode_eval/mixtral8x7b_dolly_eval_decode_n64_new16_mem48_20260502.jsonl` filtered to `phase=decode`.
- Train/reference trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`.
- Workload: `64` eval requests, `1024` decode tokens, `32768` decode router events, `65536` expert refs.
- Budget mode: fixed total expert-cache slots, fractions `12.5%/25%/37.5%/50%/75%`.
- WREC stats/reference are still built from train prefill trace; this tests prefill-prior transfer to decode workload.

## Main Table

| slots | policy | miss rate | stall ms/decode token | gain vs LRU | waste bytes/decode token | oracle gap ratio |
|---:|---|---:|---:|---:|---:|---:|
| 32 | on_demand | 1.000000 | 545.0175 | -8.55% | 0.00 | 0.3922 |
| 32 | lru | 0.921204 | 502.0721 | 0.00% | 0.00 | 0.3402 |
| 32 | static_hot | 0.843155 | 459.5342 | 8.47% | 0.00 | 0.2791 |
| 32 | route_window_prefetch | 1.000000 | 545.0175 | -8.55% | 9382625280.00 | 0.3922 |
| 32 | wrec_h | 0.836426 | 455.8667 | 9.20% | 0.00 | 0.2733 |
| 32 | wrec_h2 | 0.877777 | 478.4039 | 4.71% | 0.00 | 0.3075 |
| 32 | belady_oracle | 0.607819 | 331.2718 | 34.02% | 0.00 | 0.0000 |
| 64 | on_demand | 1.000000 | 545.0175 | -42.98% | 0.00 | 0.5928 |
| 64 | lru | 0.699402 | 381.1863 | 0.00% | 0.00 | 0.4177 |
| 64 | static_hot | 0.699753 | 381.3776 | -0.05% | 0.00 | 0.4180 |
| 64 | route_window_prefetch | 0.446274 | 243.2271 | 36.19% | 0.00 | 0.0875 |
| 64 | wrec_h | 0.690552 | 376.3628 | 1.27% | 0.00 | 0.4103 |
| 64 | wrec_h2 | 0.739120 | 402.8336 | -5.68% | 0.00 | 0.4490 |
| 64 | belady_oracle | 0.407242 | 221.9539 | 41.77% | 0.00 | 0.0000 |
| 96 | on_demand | 1.000000 | 545.0175 | -79.78% | 0.00 | 0.7204 |
| 96 | lru | 0.556229 | 303.1544 | 0.00% | 0.00 | 0.4973 |
| 96 | static_hot | 0.567307 | 309.1920 | -1.99% | 0.00 | 0.5071 |
| 96 | route_window_prefetch | 0.320862 | 174.8753 | 42.31% | 0.00 | 0.1285 |
| 96 | wrec_h | 0.547241 | 298.2561 | 1.62% | 0.00 | 0.4890 |
| 96 | wrec_h2 | 0.603394 | 328.8601 | -8.48% | 0.00 | 0.5366 |
| 96 | belady_oracle | 0.279633 | 152.4047 | 49.73% | 0.00 | 0.0000 |
| 128 | on_demand | 1.000000 | 545.0175 | -132.51% | 0.00 | 0.8136 |
| 128 | lru | 0.430084 | 234.4035 | 0.00% | 0.00 | 0.5666 |
| 128 | static_hot | 0.438614 | 239.0523 | -1.98% | 0.00 | 0.5751 |
| 128 | route_window_prefetch | 0.244171 | 133.0776 | 43.23% | 0.00 | 0.2367 |
| 128 | wrec_h | 0.419449 | 228.6070 | 2.47% | 0.00 | 0.5556 |
| 128 | wrec_h2 | 0.468643 | 255.4188 | -8.97% | 0.00 | 0.6023 |
| 128 | belady_oracle | 0.186386 | 101.5837 | 56.66% | 0.00 | 0.0000 |
| 192 | on_demand | 1.000000 | 545.0175 | -398.64% | 0.00 | 0.9364 |
| 192 | lru | 0.200546 | 109.3012 | 0.00% | 0.00 | 0.6826 |
| 192 | static_hot | 0.202866 | 110.5653 | -1.16% | 0.00 | 0.6863 |
| 192 | route_window_prefetch | 0.112167 | 61.1332 | 44.07% | 0.00 | 0.4326 |
| 192 | wrec_h | 0.186783 | 101.7999 | 6.86% | 0.00 | 0.6593 |
| 192 | wrec_h2 | 0.215240 | 117.3098 | -7.33% | 0.00 | 0.7043 |
| 192 | belady_oracle | 0.063644 | 34.6873 | 68.26% | 0.00 | 0.0000 |

## WREC vs LRU

| slots | WREC-H gain | WREC-H2 gain | WREC-H stall | WREC-H2 stall | LRU stall | Belady stall |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 9.20% | 4.71% | 455.8667 | 478.4039 | 502.0721 | 331.2718 |
| 64 | 1.27% | -5.68% | 376.3628 | 402.8336 | 381.1863 | 221.9539 |
| 96 | 1.62% | -8.48% | 298.2561 | 328.8601 | 303.1544 | 152.4047 |
| 128 | 2.47% | -8.97% | 228.6070 | 255.4188 | 234.4035 | 101.5837 |
| 192 | 6.86% | -7.33% | 101.7999 | 117.3098 | 109.3012 | 34.6873 |

## Conclusion

- Decode-only replay does not reproduce the strong total-budget prefill result.
- WREC-H gives small positive gains over uniform LRU across budgets, from `1.27%` to `9.20%`.
- WREC-H2 is worse than LRU for budgets `64/96/128/192`, which suggests the request/cross-layer weights tuned on prefill over-adapt on decode.
- Static-hot from train prefill is competitive or better than WREC-H/H2 on this small decode workload at most budgets.
- Belady remains much better, so decode has exploitable locality, but current WREC-H2 scoring is not the right decode policy without retuning or decode-specific reference statistics.
