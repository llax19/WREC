# WREC Runtime Shadow Alignment Validation

## Inputs

- Shadow result: `results/wrec/runtime_shadow_3090ti_dual20_20260509/shadow_slots64.json`
- Replay CSV: `results/wrec/wrec_total_budget_3090ti_dual20_20260509/main_recent0.csv`
- Replay policy: `wrec_h2`
- Total slots: `64`

## Verdict

- Overall pass: `True`

## Metrics

| metric | shadow | replay | delta | abs delta | tolerance | pass |
|---|---:|---:|---:|---:|---:|---:|
| expert_refs | 905408 | 905408 | 0 | 0 | 0 | True |
| demand_hits | 757395 | 757395 | 0 | 0 | 0 | True |
| demand_misses | 148013 | 148013 | 0 | 0 | 0 | True |
| hit_rate | 0.836523423694 | 0.836523423694 | 0 | 0 | 1e-12 | True |
| miss_rate | 0.163476576306 | 0.163476576306 | 0 | 0 | 1e-12 | True |
| demand_transfer_bytes | 5.2148167508e+13 | 5.2148167508e+13 | 0 | 0 | 1e-06 | True |
| stall_ms | 1260463.78553 | 1260463.78553 | 0 | 0 | 1e-06 | True |
| stall_ms_per_input_token | 89.0976027095 | 89.0976027095 | 0 | 0 | 1e-06 | True |

## Conclusion

- The validation compares runtime shadow output against the existing fixed total-budget replay row.
- Passing this validation means the shadow runtime preserves replay policy behavior under the same trace, prior, budget, and weights.
- This does not imply real expert loading control or end-to-end serving latency improvement.
