# WREC Ranked Prefetch Smoke

## Setup

- Model: Qwen1.5-MoE-A2.7B
- Method: finite-slot selective prefetch with sidecar ranked experts
- Slot capacity: 32
- max_num_batched_tokens: 16
- Requests: 2 repeated short prompts, max_new_tokens=1
- Output root: `results/wrec/runtime_ranked_prefetch_smoke_20260515T`
- Server log: `logs/server/qwen_ranked_prefetch_smoke_20260515T/finite_slot_slot32_mbt16_r1_server.log`

## Implementation Change

- Sidecar exposes `/rankings`, returning `ranked_experts_by_layer` for all layers.
- Each `/event` response returns ranked experts for the current layer and next layer.
- vLLM sidecar client fetches initial rankings at startup.
- WREC runtime attention hook uses `ranked_experts_by_layer[layer][:active_slots]` before falling back to old `would_admit` desired experts.

## Result

- Status: success
- Requests completed: 2
- Eager registered MoE layers: 24
- Extra CPU-copy fallback logs: 0
- Slot overflow: 0
- Residency stats rows: 96
- `sum(H) / sum(U)`: 601 / 908 = 0.6619
- Row-average hit rate: 0.4192
- Max row hit rate: 1.0000
- Fallback rows: 0
- Sidecar ranking overhead: 0.1021 s total, 425.43 us/router event

Block-level stats:

| block | U | H | M | hit |
|---|---:|---:|---:|---:|
| 1 | 96 | 0 | 96 | 0.0000 |
| 2 | 96 | 0 | 96 | 0.0000 |
| 3 | 358 | 243 | 115 | 0.6788 |
| 4 | 358 | 358 | 0 | 1.0000 |

## Boundary

This smoke validates that ranked-expert prefetch changes the runtime mechanism. It does not prove end-to-end latency improvement. The 2-request smoke became slower than the previous lazy-init smoke because top-32 prefetch is aggressive and adds H2D transfer work.
