# Mixtral memory cap probe - 2026-05-01

## Goal

Validate whether Mixtral-8x7B-Instruct can load, CPU-offload, and emit router event traces under intentionally constrained effective GPU memory on the new 96GB GPU environment.

This is a feasibility probe, not a throughput benchmark.

## Environment and dependencies

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Physical GPU memory: 97887 MiB
- Baseline idle usage before probe: about 493 MiB
- Python environment: base `/opt/conda/bin/python`
- Installed missing probe dependencies before running:
  - `transformers==4.57.6`
  - `accelerate==1.13.0`
  - `safetensors==0.7.0`
  - `numpy==2.2.6`

## Probe command shape

All runs used:

- model: `models/Mixtral-8x7B-Instruct-v0.1`
- request file: `data/prompts/wrec_dolly_debug_n64.jsonl`
- limit: `1`
- max input tokens: `128`
- dtype: `float16`
- device map: `auto`
- CPU memory cap: `180GiB`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--disable-cuda-allocator-warmup`
- `--continue-on-error`

The varied parameter was `--max-memory` for GPU 0.

## Results

| GPU memory cap | Observed peak-ish GPU memory | Requests | Failures | Input tokens | Router layers | Events | Trace seconds | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 80GiB | ~78.8GB | 1 | 0 | 19 | 32 | 608 | 4.073 | success |
| 72GiB | ~70.5GB | 1 | 0 | 19 | 32 | 608 | 1.788 | success |
| 64GiB | ~33.9GB observed during polling | 1 | 0 | 19 | 32 | 608 | 2.393 | success |
| 48GiB | ~45.6GB | 1 | 0 | 19 | 32 | 608 | 6.878 | success |
| 48GiB | ~45.8GB | 4 | 0 | 189 | 32 | 6048 | 42.844 total | success |

All four runs printed:

```text
Some parameters are on the meta device because they were offloaded to the cpu.
```

This confirms that the constrained runs used CPU offload rather than pure all-resident GPU placement.

## Output files

- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n1_mem80_20260501.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n1_mem72_20260501.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n1_mem64_20260501.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n1_mem48_20260501.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n4_mem48_20260501.jsonl`
- `results/wrec/router_trace_stats_mixtral8x7b_probe_n1_mem80_20260501.json`
- `results/wrec/router_trace_stats_mixtral8x7b_probe_n1_mem72_20260501.json`
- `results/wrec/router_trace_stats_mixtral8x7b_probe_n1_mem64_20260501.json`
- `results/wrec/router_trace_stats_mixtral8x7b_probe_n1_mem48_20260501.json`
- `results/wrec/router_trace_stats_mixtral8x7b_probe_n4_mem48_20260501.json`
- `results/wrec/router_trace_failures_mixtral8x7b_probe_n1_mem80_20260501.jsonl`
- `results/wrec/router_trace_failures_mixtral8x7b_probe_n1_mem72_20260501.jsonl`
- `results/wrec/router_trace_failures_mixtral8x7b_probe_n1_mem64_20260501.jsonl`
- `results/wrec/router_trace_failures_mixtral8x7b_probe_n1_mem48_20260501.jsonl`
- `results/wrec/router_trace_failures_mixtral8x7b_probe_n4_mem48_20260501.jsonl`

All failure files are empty.

## Conclusion

The new environment can run Mixtral router-event tracing under explicit GPU memory caps of 80GiB, 72GiB, 64GiB, and 48GiB. The `48GiB` cap also passed the Phase 2B n=4 feasibility check. This validates the practical path for testing expert-transfer-bound settings by constraining effective GPU memory.

For the next stage, use `48GiB` as the controlled-memory setting for Mixtral debug trace, with `64GiB` as the fallback if debug-scale tracing is too slow or unstable.

## 48GiB n=4 details

The n=4 run used `--max-memory 0=48GiB,cpu=110GiB`.

| request | input tokens | events | trace seconds |
|---|---:|---:|---:|
| dolly-debug-000000 | 19 | 608 | 32.768 |
| dolly-debug-000001 | 21 | 672 | 3.283 |
| dolly-debug-000002 | 128 | 4096 | 3.386 |
| dolly-debug-000003 | 21 | 672 | 3.406 |

The first request absorbed most of the post-load warmup/offload overhead. Later requests were much faster.
