# Mixtral Train Workload Statistics
## Inputs
- Train trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Eval trace: `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Static-hot capacities: `[1, 2, 4]`
- Window sizes: `[4, 8, 16]`
## Summary
- Requests: `29123` input-token units across `931936` router events.
- Expert refs: `1863872`.
- Selected experts per router event: `2.0000`.
- Router events per input token: `32.0000`.
- Expert refs per input token: `64.0000`.
- Same-layer expert reuse distance p50/p90/p99: `5.00` / `27.00` / `7097.00` expert refs.
- Per-request-layer working set p50/p90/p99: `8.00` / `8.00` / `8.00` unique experts.
- Train/eval hotness total variation p50/p90: `0.0104` / `0.0164`.
- Train/eval top-2 overlap mean: `0.9375`.
- Train/eval top-4 overlap mean: `0.9375`.
## Notes
- `static_hot_by_capacity` in the JSON is computed only from train trace.
- Eval trace is used here only for train/eval hotness shift diagnostics.
