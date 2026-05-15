# Qwen Real Inference Prefill n16 Selected Memory Run

## Setup
- Request file: `/root/WREC/data/prompts/wrec_dolly_prefill_eval_long_n16_maxnew1_20260514.jsonl`
- Requests: 16 long-prompt Dolly eval requests
- Output budget: 1 token
- Model len: 1280
- GPU monitor interval: 1 s
- Resource setting: `gpu_memory_utilization=0.70`

## Completed Cases

| case | total tok/s | p95 TTFT (s) | avg GPU MiB | peak GPU MiB |
|---|---:|---:|---:|---:|
| no_wrec_mbt4_r1 | 4.819 | 169.0 | 17441 | 17441 |
| finite_slot_slot24_mbt4_r1 | 5.049 | 163.0 | 17441 | 17441 |
| finite_slot_slot32_mbt4_r1 | 5.900 | 139.7 | 17441 | 17441 |
| no_wrec_mbt8_r1 | 5.683 | 144.7 | 17441 | 17441 |
| finite_slot_slot32_mbt8_r1 | 6.477 | 126.7 | 17441 | 17441 |

## Fair Comparisons

| comparison | total tok/s ratio | p95 TTFT ratio |
|---|---:|---:|
| slot24/mbt4 vs no_wrec/mbt4 | 1.048x | 0.965x |
| slot32/mbt4 vs no_wrec/mbt4 | 1.224x | 0.827x |
| slot32/mbt8 vs no_wrec/mbt8 | 1.140x | 0.876x |

## Failed Case

- `finite_slot_slot48_mbt8_r1` failed before the vLLM server became ready.
- Root cause in server log: `Available KV cache memory: -5.7 GiB`, followed by `ValueError: No available memory for the cache blocks`.
- Under the same `gpu_memory_utilization=0.70` resource budget, `slot48/mbt8` is not a feasible configuration.

## Notes

- The GPU memory monitor reports vLLM's allocated process memory during the request window. For these successful cases it is flat at about 17.0 GiB, so the main observed difference is latency/throughput rather than peak `nvidia-smi` memory.
- The `slot48/mbt8` startup failure is still useful: it marks a resource boundary under the selected single-GPU budget.
