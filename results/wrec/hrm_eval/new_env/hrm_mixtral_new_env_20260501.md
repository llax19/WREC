# Mixtral HRM rerun on new environment - 2026-05-01

## Scope

This rerun uses the local `models/Mixtral-8x7B-Instruct-v0.1/config.json` profile and the current machine inventory. It does not load the full Mixtral weights and should be read as formula-level HRM / capacity analysis, not measured serving throughput.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- GPU memory reported by `nvidia-smi`: 97887 MiB total, 96721 MiB free during inventory
- HRM calibrated GPU constants:
  - HBM copy bandwidth: 1451.13 GB/s
  - CPU to GPU bandwidth: 41.37 GB/s
  - BF16 GPU matmul: 382.45 TFLOP/s
- CPU bandwidth / CPU FLOP calibration was skipped because the base environment lacks `numpy`; policy evaluation therefore used the script defaults for CPU-side constants.

## Pre-HRM Matrix

Command output files:

- `results/wrec/hrm_eval/mixtral/hrm_bottleneck_matrix_mixtral8x7b_20260501.json`
- `results/wrec/hrm_eval/mixtral/hrm_bottleneck_matrix_mixtral8x7b_20260501.csv`

Matrix settings:

- dtype: bf16, int8, int4
- GPU memory budgets: 24, 48, 96 GB
- KV cache reservations: 4, 8, 16, 32 GB
- CPU-GPU bandwidth assumptions: 8, 16, 32, 41.37 GB/s
- resident fractions: 0.1, 0.25, 0.5, 0.75, 1.0
- active decode tokens: 1, 8, 32, 64

Result summary:

- Total rows: 2880
- All-resident feasible rows: 1120
- Expert-transfer-bound rows: 1760
- Mixtral BF16 estimated size: 86.99 GiB total, including 84.00 GiB expert weights and 2.99 GiB dense non-expert weights.

Key interpretation:

- BF16 at 24 GB or 48 GB remains expert-transfer-bound across this matrix.
- BF16 at 96 GB becomes all-resident only when KV reservation is 4 or 8 GB. With 16 or 32 GB reserved for KV/cache/overhead, full BF16 all-resident no longer fits under this simple model.
- INT8 at 96 GB is all-resident across the tested KV reservations.
- INT4 is all-resident at 48 GB except the 32 GB KV reservation case, and all-resident across all tested 96 GB cases.

## MoE-Lightning-style Policy Evaluation

Command output files:

- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_bf16_1gpu_20260501.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_bf16_1gpu_20260501.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int8_1gpu_20260501.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int8_1gpu_20260501.csv`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int4_1gpu_20260501.json`
- `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int4_1gpu_20260501.csv`

Policy search settings:

- batch sizes: 1, 4, 8, 16, 32, 64
- micro-batch sizes: 1, 4, 8, 16, 32, 64
- attention placement: CPU or GPU
- FFN placement: CPU or GPU
- GPU-resident weight ratio `r_w`: 0, 0.25, 0.5, 0.75, 1.0
- GPU-resident KV ratio `r_c`: 0, 0.25, 0.5, 0.75, 1.0
- average prompt length: 512
- generation length: 128

Best throughput-proxy policy for all three dtypes:

- `N=64`, `mu=64`
- attention on CPU: `A_g=0`
- FFN on GPU: `F_g=1`
- all FFN weights resident on GPU: `r_w=1.0`
- KV cache on CPU under this policy: `r_c=0.0`

Best throughput-proxy rows:

| dtype | layer latency ms | proxy tokens/s | estimated GPU memory used GiB | bottleneck |
|---|---:|---:|---:|---|
| bf16 | 1.9431 | 32937.83 | 86.99 | gpu-compute-or-memory |
| int8 | 0.9715 | 65875.66 | 43.50 | gpu-compute-or-memory |
| int4 | 0.4858 | 131751.32 | 21.75 | gpu-compute-or-memory |

BF16 best policy by batch:

| N | mu | A_g | F_g | r_w | r_c | latency ms | proxy tokens/s | GPU memory GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 1 | 0.25 | 0.0 | 0.4856 | 2059.33 | 27.93 |
| 4 | 4 | 0 | 1 | 0.75 | 0.0 | 1.3278 | 3012.48 | 67.30 |
| 8 | 8 | 0 | 1 | 1.00 | 0.0 | 1.7480 | 4576.74 | 86.99 |
| 16 | 16 | 0 | 1 | 1.00 | 0.0 | 1.9230 | 8320.14 | 86.99 |
| 32 | 32 | 0 | 1 | 1.00 | 0.0 | 1.9425 | 16473.63 | 86.99 |
| 64 | 64 | 0 | 1 | 1.00 | 0.0 | 1.9431 | 32937.83 | 86.99 |

## Conclusion

The new 96 GB single-GPU environment materially changes the HRM conclusion for Mixtral-8x7B. Under BF16, full expert residency is now plausible if KV/cache/overhead reservation stays around 8 GB or lower. If effective available memory drops below about 87 GiB because of KV cache, runtime overhead, or fragmentation, Mixtral returns to the expert-transfer-bound regime.

For the WREC/offload story, this means the new machine is good for making Mixtral run, but it is less clean as the main evidence for expert-loading bottlenecks unless we deliberately evaluate lower effective memory budgets, larger KV reservations, quantization variants, or scale-out models.

