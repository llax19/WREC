# Mixtral Decode Pilot Trace Summary

- Trace: `/root/workspace/logs/processed/wrec/decode_pilot/mixtral8x7b_dolly_decode_pilot_n16_new16_mem48_20260502.jsonl`
- Stats: `/root/workspace/results/wrec/decode_pilot/router_trace_stats_mixtral8x7b_dolly_decode_pilot_n16_new16_mem48_20260502.json`
- Requests: `16`; failures: `0`.
- Prompt tokens: `742`; decode tokens: `256`.
- Router events: `31936` = prefill `23744` + decode `8192`.
- Decode steps per request: `16` to `16`.
- Router events per decode token: `32.00`; expert refs per decode token: `64.00`.

## Prefill vs decode hotness

- Per-layer TV shift p50: `0.0795`; p90: `0.1275`; max: `0.1493`.
- Top-2 expert overlap mean: `0.4688`; min: `0.0000`.

| layer | TV | prefill top-2 | decode top-2 | top-2 overlap |
|---:|---:|---|---|---:|
| 6 | 0.1493 | `[2, 7]` | `[5, 6]` | 0.00 |
| 13 | 0.1485 | `[1, 7]` | `[1, 2]` | 0.50 |
| 10 | 0.1407 | `[3, 7]` | `[0, 1]` | 0.00 |
| 12 | 0.1277 | `[1, 3]` | `[0, 1]` | 0.50 |
| 9 | 0.1260 | `[6, 7]` | `[0, 4]` | 0.00 |

## Conclusion

- Decode tracing is functional on the Mixtral Dolly debug pilot with 48GiB GPU memory cap and CPU offload.
- The pilot is small, but decode hotness differs materially from prompt prefill on several layers; decode should be evaluated separately before treating prefill replay as a runtime proxy.
