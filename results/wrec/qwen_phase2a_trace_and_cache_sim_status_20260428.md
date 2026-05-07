# Qwen Phase 2A Trace And Cache Simulator Status

Date: 2026-04-28

## Goal

Complete the non-Mixtral-blocked WREC tooling path:

1. Build token/layer router event traces with Qwen1.5-MoE-A2.7B.
2. Replay those events through a first expert cache/offload simulator.

## Trace Command

```bash
conda run -n vllm_moe python scripts/moe_affinity/build_moe_router_event_trace.py \
  --model-path qwen1.5-MoE-A2.7B \
  --model-name qwen1.5-moe-a2.7b \
  --request-file data/prompts/debug_requests.jsonl \
  --output logs/processed/wrec/qwen_debug_router_events_n24_20260428.jsonl \
  --stats-output results/wrec/router_trace_stats_qwen_debug_n24_20260428.json \
  --failure-output results/wrec/router_trace_failures_qwen_debug_n24_20260428.jsonl \
  --limit 24 \
  --max-input-tokens 1024 \
  --phase prefill \
  --dtype auto \
  --device-map auto \
  --trust-remote-code
```

## Trace Outputs

- `logs/processed/wrec/qwen_debug_router_events_n24_20260428.jsonl`
- `results/wrec/router_trace_stats_qwen_debug_n24_20260428.json`
- `results/wrec/router_trace_failures_qwen_debug_n24_20260428.jsonl`

Trace validation:

| metric | value |
| --- | ---: |
| requests | 24 |
| failures | 0 |
| input tokens | 1234 |
| router layers | 24 |
| router events | 29616 |
| expert refs | 118464 |
| selected experts per event | 4 |

## Simulator Command

```bash
python scripts/moe_affinity/simulate_expert_cache_offload.py \
  --trace logs/processed/wrec/qwen_debug_router_events_n24_20260428.jsonl \
  --model-path qwen1.5-MoE-A2.7B \
  --cache-capacities 0,1,2,4,25%,50%,100% \
  --output-json results/wrec/cache_sim_baselines_qwen_debug_n24_20260428.json \
  --output-csv results/wrec/cache_sim_baselines_qwen_debug_n24_20260428.csv \
  --run-sanity-tests
```

## Simulator Outputs

- `results/wrec/cache_sim_baselines_qwen_debug_n24_20260428.json`
- `results/wrec/cache_sim_baselines_qwen_debug_n24_20260428.csv`

Internal sanity tests passed:

- Full expert cache gives zero misses.
- Zero cache gives all misses.
- Belady oracle stall is not higher than LRU stall.
- Static-hot beats LRU on a high-skew synthetic trace.
- Prefetch waste is non-negative.
- Total transfer bytes are not lower than demand transfer bytes.

Selected Qwen debug replay results:

| cache experts/layer | LRU miss rate | Belady miss rate | route-window miss rate |
| ---: | ---: | ---: | ---: |
| 1 | 0.9932 | 0.9932 | 0.9829 |
| 2 | 0.9759 | 0.8531 | 0.9530 |
| 4 | 0.8843 | 0.6962 | 0.6889 |
| 15 | 0.6316 | 0.3596 | 0.0872 |
| 30 | 0.3683 | 0.1622 | 0.0149 |
| 60 | 0.0000 | 0.0000 | 0.0000 |

## Conclusion

Qwen Phase 2A is complete as a tooling validation step. The trace schema, selected expert extraction, and cache/offload simulator input path are now working.

This is not a WREC main-result table because Qwen is not the main transfer-bound model on the current dual-GPU setting. The next mainline work is to use this trace path and simulator with Mixtral only after the Mixtral weight download finishes and the Phase 2B offload probe succeeds.
