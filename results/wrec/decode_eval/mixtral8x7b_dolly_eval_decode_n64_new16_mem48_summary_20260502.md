# Mixtral Eval Decode Trace Summary

- Trace: `/root/workspace/logs/processed/wrec/decode_eval/mixtral8x7b_dolly_eval_decode_n64_new16_mem48_20260502.jsonl`
- Stats: `/root/workspace/results/wrec/decode_eval/router_trace_stats_mixtral8x7b_dolly_eval_decode_n64_new16_mem48_20260502.json`
- Requests: `64`; failures: `0`.
- Max input tokens: `128`; max new tokens: `16`.
- Prompt tokens: `3908`; decode tokens: `1024`.
- Router events: `157824` = prefill `125056` + decode `32768`.
- Decode steps per request: `16` to `16`.
- Router events per decode token: `32.00`; expert refs per decode token: `64.00`.
- Elapsed: `1149.32` seconds; seconds/request: `17.96`.

## Prefill vs Decode Hotness

- Per-layer TV shift p50: `0.0549`; p90: `0.0746`; max: `0.1181`.
- Top-2 expert overlap mean: `0.5625`; min: `0.0000`.

| layer | TV | prefill top-2 | decode top-2 | top-2 overlap |
|---:|---:|---|---|---:|
| 8 | 0.1181 | `[4, 6]` | `[5, 6]` | 0.50 |
| 10 | 0.0820 | `[3, 7]` | `[0, 1]` | 0.00 |
| 11 | 0.0815 | `[2, 6]` | `[1, 2]` | 0.50 |
| 16 | 0.0746 | `[0, 1]` | `[0, 5]` | 0.50 |
| 14 | 0.0744 | `[4, 6]` | `[4, 5]` | 0.50 |
| 15 | 0.0709 | `[1, 4]` | `[1, 7]` | 0.50 |
| 12 | 0.0704 | `[1, 3]` | `[0, 1]` | 0.50 |
| 13 | 0.0679 | `[0, 1]` | `[0, 2]` | 0.50 |

## Conclusion

- Eval decode trace collection is stable at `n=64`, `max_new_tokens=16`, `max_input_tokens=128`, `48GiB` memory cap.
- Decode expert hotness remains different from prefill, so decode cache replay should be evaluated separately from prefill replay.
