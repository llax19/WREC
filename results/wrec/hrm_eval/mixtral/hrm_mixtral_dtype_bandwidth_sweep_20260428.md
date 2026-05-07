# Mixtral HRM Dtype And Bandwidth Sweep

Date: 2026-04-28

## Goal

Add lightweight config-only Phase 0 supplements while full Mixtral weights are still downloading:

- dtype sweep: BF16, INT8, INT4.
- CPU-GPU bandwidth sweep: measured 6.50 GB/s, 8 GB/s, 16 GB/s, 32 GB/s.

These runs only read `models/Mixtral-8x7B-Instruct-v0.1/config.json` and the measured hardware profile. They do not require full model weights.

## Outputs

- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_int8_calibrated_20260428.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_int8_calibrated_20260428.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_int4_calibrated_20260428.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_int4_calibrated_20260428.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw8_20260428.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw8_20260428.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw16_20260428.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw16_20260428.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw32_20260428.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_bw32_20260428.csv`

## Dtype Sweep

Two-GPU aggregate capacity approximation, measured hardware constants.

| dtype | total weight GiB | feasible policies | infeasible GPU policies | cpu-gpu-transfer policies | best policy bottleneck |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 86.99 | 442 | 314 | 304 | gpu-compute-or-memory |
| INT8 | 43.50 | 756 | 0 | 494 | gpu-compute-or-memory |
| INT4 | 21.75 | 756 | 0 | 494 | gpu-compute-or-memory |

Interpretation:

- BF16 remains the clearest WREC target because the full model cannot fit in the two-GPU aggregate capacity budget.
- INT8/INT4 remove the all-resident memory pressure in this formula-level estimate, so they are not good settings for proving the first-order expert offload bottleneck.
- The INT8/INT4 rows are structural byte estimates only; they do not model real quantized kernels or dequantization overhead.

## Bandwidth Sweep

BF16, two-GPU aggregate capacity approximation.

| CPU-GPU bandwidth | feasible policies | cpu-gpu-transfer policies | gpu-compute-or-memory policies |
| ---: | ---: | ---: | ---: |
| measured 6.50 GB/s | 442 | 304 | 12 |
| 8 GB/s | 442 | 304 | 12 |
| 16 GB/s | 442 | 304 | 12 |
| 32 GB/s | 442 | 302 | 14 |

Interpretation:

- The BF16 Mixtral configuration remains largely transfer-bound across realistic PCIe bandwidth assumptions.
- Even at 32 GB/s, most feasible partial-residency `F_g=1` policies remain `cpu-gpu-transfer` bottlenecked.
- This supports keeping Mixtral BF16 as the main bottleneck-evaluation configuration, while using quantized variants only as a sensitivity boundary.

## Conclusion

The sweep strengthens the existing Phase 0 decision:

```text
Main bottleneck target: Mixtral-8x7B BF16 under constrained GPU residency.
Not main bottleneck target: Qwen dual-GPU all-resident; Mixtral INT8/INT4 all-resident estimates.
```

Full Mixtral weights are still only needed for Phase 2B offload probe and real router traces.
