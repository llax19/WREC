# WREC Runtime Shadow Alignment Validation

## Inputs

- Shadow result: `/root/workspace/results/wrec/runtime_shadow/wrec_shadow_mixtral8x7b_dolly_eval_n256_slots64_20260504.json`
- Replay CSV: `/root/workspace/results/wrec/wrec_total_budget/wrec_total_budget_recent0_mixtral8x7b_dolly_n256_20260503.csv`
- Replay policy: `wrec_h2`
- Total slots: `64`

## Verdict

- Overall pass: `True`

## Metrics

| metric | shadow | replay | delta | abs delta | tolerance | pass |
|---|---:|---:|---:|---:|---:|---:|
| expert_refs | 905408 | 905408 | 0 | 0 | 0 | True |
| demand_hits | 757524 | 757524 | 0 | 0 | 0 | True |
| demand_misses | 147884 | 147884 | 0 | 0 | 0 | True |
| hit_rate | 0.836665900898 | 0.836665900898 | 0 | 0 | 1e-12 | True |
| miss_rate | 0.163334099102 | 0.163334099102 | 0 | 0 | 1e-12 | True |
| demand_transfer_bytes | 5.21027180298e+13 | 5.21027180298e+13 | 0 | 0 | 1e-06 | True |
| stall_ms | 1259365.23454 | 1259365.23454 | 0 | 0 | 1e-06 | True |
| stall_ms_per_input_token | 89.0199501334 | 89.0199501334 | 0 | 0 | 1e-06 | True |

## Conclusion

- The validation compares runtime shadow output against the existing fixed total-budget replay row.
- Passing this validation means the shadow runtime preserves replay policy behavior under the same trace, prior, budget, and weights.
- This does not imply real expert loading control or end-to-end serving latency improvement.
