# WREC Lazy Init Smoke

## Setup

- Model: Qwen1.5-MoE-A2.7B
- Method: finite-slot selective prefetch
- Slot capacity: 32
- max_num_batched_tokens: 16
- Requests: 2 short prompts, max_new_tokens=1
- Output root: `results/wrec/runtime_lazy_init_smoke_20260515T`
- Server log: `logs/server/qwen_lazy_init_smoke_20260515T/finite_slot_slot32_mbt16_r1_server.log`

## Result

- Status: success
- Requests completed: 2
- Eager registered MoE layers: 24
- Reused vLLM CPU offload storage logs: 48
- Extra CPU-copy fallback logs: 0
- Slot overflow: 0
- Residency stats rows: 96
- `sum(H) / sum(U)`: 166 / 908 = 0.1828
- Row-average hit rate: 0.1170
- Max row hit rate: 0.6923
- Fallback rows: 0

## Conclusion

The lazy-init path is fixed at the runtime integration level: all 24 Qwen MoE layers are registered during UVA offloader post-init, before the first MoE forward.

The first few residency rows still have `H=0` because the sidecar has no desired experts before it observes routed experts from the first request. This remaining cold-start effect is sidecar-history cold start, not missing layer state.
