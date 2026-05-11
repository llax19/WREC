# Qwen finite-slot short eval n=8 summary

| mode | requests | mean latency s | max latency s | mean TTFT s | input tokens | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| finite_slot_s16_mbt4 | 8 | 2.399009 | 4.722728 | 2.398946 | 117 | 8 |
| stateonly_mbt4 | 8 | 4.198002 | 6.823200 | 4.197953 | 117 | 8 |
| no_wrec_mbt4 | 8 | 3.099667 | 4.912214 | 3.099624 | 117 | 8 |

Notes:
- `finite_slot_s16_mbt4` used `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY=16` and `max_num_batched_tokens=4`.
- Earlier finite-slot attempts with `slot=16, max_num_batched_tokens=32` and `slot=40, max_num_batched_tokens=32` failed with slot overflow.
- This is a controlled smoke comparison on short eval prompts, not a final throughput benchmark.
