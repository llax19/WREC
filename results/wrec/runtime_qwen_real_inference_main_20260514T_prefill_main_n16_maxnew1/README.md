# Qwen Real Inference Prefill Main n16 maxnew1

## Setup
- Request file: `/root/WREC/data/prompts/wrec_dolly_prefill_eval_long_n16_maxnew1_20260514.jsonl`
- Requests: 16 long-prompt Dolly eval requests
- Output budget: 1 token
- Model len: 1280
- Repeats: 1
- Methods: `no_wrec`, `finite_slot`

## Result

| case | total tok/s | p95 TTFT (s) | vs no_wrec tok/s | vs no_wrec TTFT | shadow miss |
|---|---:|---:|---:|---:|---:|
| no_wrec_mbt4_r1 | 4.825 | 168.8 | 1.000x | 1.000x | n/a |
| finite_slot_slot16_mbt4_r1 | 4.185 | 193.9 | 0.867x | 1.148x | 0.773 |
| finite_slot_slot24_mbt4_r1 | 4.995 | 164.5 | 1.035x | 0.974x | 0.773 |
| finite_slot_slot32_mbt4_r1 | 5.795 | 142.1 | 1.201x | 0.842x | 0.773 |
| finite_slot_slot32_mbt8_r1 | 6.492 | 126.9 | 1.346x | 0.752x | 0.773 |

## Notes
- `slot16/mbt4` is slower than no-WREC on this long-prompt prefill workload.
- `slot32/mbt8` is the best case in this run: 1.346x total token throughput and 24.8% lower p95 TTFT versus no-WREC.
- `shadow_miss_rate` should not be used to compare finite slot sizes in this run because `SIDECAR_TOTAL_SLOTS` was fixed at 192.
- This is a single-repeat n16 run. Treat it as a complete first real-inference pass, not the final statistical table.
