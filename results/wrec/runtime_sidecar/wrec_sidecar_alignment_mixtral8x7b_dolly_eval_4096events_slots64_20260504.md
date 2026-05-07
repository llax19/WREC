# WREC Runtime Sidecar Alignment Validation

## Inputs

- Sidecar smoke: `/root/workspace/results/wrec/runtime_sidecar/wrec_sidecar_smoke_mixtral8x7b_dolly_eval_4096events_slots64_20260504.json`
- Local shadow: `/root/workspace/results/wrec/runtime_sidecar/wrec_shadow_prefix8192_mixtral8x7b_dolly_eval_slots64_20260504.json`

## Verdict

- Overall pass: `True`

## Metrics

| metric | sidecar | local shadow | delta | abs delta | tolerance | pass |
|---|---:|---:|---:|---:|---:|---:|
| expert_refs | 8192 | 8192 | 0 | 0 | 0 | True |
| router_events | 4096 | 4096 | 0 | 0 | 0 | True |
| shadow_hits | 7237 | 7237 | 0 | 0 | 0 | True |
| shadow_misses | 955 | 955 | 0 | 0 | 0 | True |
| shadow_hit_rate | 0.883422851562 | 0.883422851562 | 0 | 0 | 1e-12 | True |
| shadow_miss_rate | 0.116577148438 | 0.116577148438 | 0 | 0 | 1e-12 | True |
| would_admit | 651 | 651 | 0 | 0 | 0 | True |
| would_bypass | 304 | 304 | 0 | 0 | 0 | True |
| would_evict | 651 | 651 | 0 | 0 | 0 | True |
| demand_transfer_bytes | 336467066880 | 336467066880 | 0 | 0 | 1e-06 | True |
| stall_ms | 8132.6837182 | 8132.6837182 | 0 | 0 | 1e-06 | True |

## Conclusion

- Passing this validation means the HTTP sidecar preserves local shadow behavior for the same event prefix.
- This validates the external runtime integration boundary, not vLLM internal expert loading control.
