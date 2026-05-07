# Mixtral debug trace at 48GiB - 2026-05-01

## Configuration

- model: `models/Mixtral-8x7B-Instruct-v0.1`
- request file: `data/prompts/wrec_dolly_debug_n64.jsonl`
- limit: `64`
- max input tokens: `128`
- dtype: `float16`
- device map: `auto`
- memory cap: `0=48GiB,cpu=110GiB`
- allocator setting: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- other flags:
  - `--continue-on-error`
  - `--disable-cuda-allocator-warmup`

## Outputs

- `logs/processed/wrec/mixtral8x7b_dolly_debug_router_events_n64_mem48_20260501.jsonl`
- `results/wrec/router_trace_stats_mixtral8x7b_dolly_debug_n64_mem48_20260501.json`
- `results/wrec/router_trace_failures_mixtral8x7b_dolly_debug_n64_mem48_20260501.jsonl`

## Result summary

- requests attempted: `64`
- failures: `0`
- failure rate: `0%`
- total input tokens: `3347`
- router layers: `32`
- experts per token: `2`
- total router events: `107104`
- total elapsed time: `216.773s`

Derived check:

- `3347 * 32 = 107104`, so event count is exactly consistent with one router event per token per MoE layer.

## Timing observations

- request `dolly-debug-000000`: `10.488s`
- most later requests: about `3.2s` to `3.3s`
- last request `dolly-debug-000063`: `3.287s`

Interpretation:

- the first request still absorbed post-load/offload warmup overhead
- after that, per-request trace time was stable

## Resource observations

- GPU memory stayed around the 48GiB cap in the loaded phase, observed around `45.6GB`
- CPU memory pressure rose but remained feasible
- no GPU OOM
- no CPU OOM
- no request-level failures

## Conclusion

The `48GiB` constrained-memory configuration passed Mixtral Phase 2C debug tracing on the full `n=64` debug split. This is sufficient to continue to the next planned stage: building the train/eval traces or moving into Phase 3 simulator work, depending on whether you want to first collect the larger Mixtral traces or validate baseline/oracle replay on the debug trace.

