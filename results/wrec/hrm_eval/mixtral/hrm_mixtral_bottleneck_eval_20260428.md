# Mixtral-8x7B-Instruct HRM Bottleneck Evaluation

Date: 2026-04-28 UTC

## Goal

Use the existing `config.json` for `models/Mixtral-8x7B-Instruct-v0.1` to run a config-only MoE-Lightning-style HRM bottleneck evaluation with the measured local hardware constants.

This does not require full model weights. It estimates structure-level memory pressure, transfer pressure, and policy bottlenecks from model shape, workload, and measured hardware constants.

## Inputs

Model config:

- `models/Mixtral-8x7B-Instruct-v0.1/config.json`

Measured hardware profile:

- `results/wrec/hrm_hardware_profile_20260428.json`

Recommended HRM constants:

```text
b_c  = 31.13 GB/s
p_c  = 0.93 TFLOP/s
b_g  = 893.93 GB/s
b_cg = 6.50 GB/s
p_g  = 72.36 TFLOP/s
```

Model structure inferred from config:

```text
num_hidden_layers      = 32
hidden_size            = 4096
num_local_experts      = 8
num_experts_per_tok    = 2
intermediate_size      = 14336
num_attention_heads    = 32
num_key_value_heads    = 8
```

Estimated BF16 footprint:

```text
total params           = 46.70B
active params/token    = 12.88B
total weight           = 86.99 GiB
dense non-expert weight= 2.99 GiB
expert FFN weight      = 84.00 GiB
```

## Commands

Two-GPU aggregate-capacity approximation:

```bash
conda run -n vllm_moe python scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py \
  --model-path models/Mixtral-8x7B-Instruct-v0.1 \
  --model-name mixtral-8x7b-instruct \
  --hardware-profile results/wrec/hrm_hardware_profile_20260428.json \
  --gpu-memory-gb 47.9765625 \
  --output-json results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_calibrated_20260428.json \
  --output-csv results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_calibrated_20260428.csv
```

Single-GPU capacity approximation:

```bash
conda run -n vllm_moe python scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py \
  --model-path models/Mixtral-8x7B-Instruct-v0.1 \
  --model-name mixtral-8x7b-instruct \
  --hardware-profile results/wrec/hrm_hardware_profile_20260428.json \
  --gpu-memory-gb 23.988 \
  --output-json results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_1gpu_calibrated_20260428.json \
  --output-csv results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_1gpu_calibrated_20260428.csv
```

## Two-GPU Result

Output:

- `results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_calibrated_20260428.json`
- `results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_2gpu_calibrated_20260428.csv`

Summary:

```text
total policies              = 756
feasible policies           = 442
infeasible-gpu-memory       = 314
cpu-compute-or-memory       = 126
cpu-gpu-transfer            = 304
gpu-compute-or-memory       = 12
```

Best throughput policy under the current proxy:

```text
N                            = 1
mu                           = 1
A_g                          = 0
F_g                          = 1
r_w                          = 0.25
r_c                          = 0.0
layer_latency_ms             = 0.7883
bottleneck                   = gpu-compute-or-memory
gpu_memory_used_gb           = 27.93
cpu_memory_used_gb           = 63.07
```

For WREC-relevant policies where FFN stays on GPU and expert weights are only partially resident:

```text
feasible F_g = 1 policies    = 316
cpu-gpu-transfer bottleneck  = 304
gpu-compute-or-memory        = 12
```

Fastest feasible `F_g=1` policy for `N=4`:

```text
mu                           = 1
A_g                          = 1
r_w                          = 0.5
r_c                          = 1.0
layer_latency_ms             = 79.6305
comm_cpu_to_gpu_ms           = 79.6305
T_gpu_ms                     = 3.1637
cross_transfer_gb_per_layer  = 0.4819
bottleneck                   = cpu-gpu-transfer
```

This is the important WREC signal: once batch size grows beyond the trivial `N=1` case, partial expert residency makes CPU-GPU expert transfer dominate.

## Single-GPU Result

Output:

- `results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_1gpu_calibrated_20260428.json`
- `results/wrec/hrm_policy_eval_mixtral_8x7b_instruct_1gpu_calibrated_20260428.csv`

Summary:

```text
total policies              = 756
feasible policies           = 252
infeasible-gpu-memory       = 504
cpu-compute-or-memory       = 126
cpu-gpu-transfer            = 126
gpu-compute-or-memory       = 0
```

Best throughput policy under the current proxy:

```text
N                            = 64
mu                           = 64
A_g                          = 1
F_g                          = 0
r_w                          = 0.0
r_c                          = 0.0
layer_latency_ms             = 90.5888
bottleneck                   = cpu-compute-or-memory
```

Fastest feasible `F_g=1` policy for `N=1`:

```text
A_g                          = 1
r_w                          = 0.0
r_c                          = 1.0
layer_latency_ms             = 108.4331
comm_cpu_to_gpu_ms           = 108.4331
T_gpu_ms                     = 0.7909
cross_transfer_gb_per_layer  = 0.6563
bottleneck                   = cpu-gpu-transfer
```

Single-GPU capacity is too tight to keep a meaningful fraction of Mixtral experts resident. If FFN runs on GPU, expert transfer dominates; if FFN runs on CPU, CPU compute/memory dominates.

## Conclusion

Mixtral is much more suitable than Qwen1.5-MoE-A2.7B for exposing the WREC target bottleneck.

Clear conclusions:

1. Mixtral BF16 total weight is about 86.99 GiB, so all-resident execution is impossible on both one RTX 3090 Ti and the current two-card aggregate 47.98 GiB capacity.
2. Under the two-GPU aggregate-capacity approximation, most feasible `F_g=1` policies are `cpu-gpu-transfer` bound. This directly matches the expert cache/offload bottleneck WREC wants to optimize.
3. Under single-GPU capacity, Mixtral becomes even more constrained: `F_g=1` policies are transfer-bound, while `F_g=0` policies become CPU compute/memory-bound.
4. Therefore Mixtral should be kept as the main bottleneck-evaluation model for Phase 0/Phase 2 planning, even before full weights are downloaded. The current config-only HRM result is enough to justify prioritizing Mixtral for expert cache experiments.

Limitations:

- This is a config-only HRM estimate, not actual inference profiling.
- The two-GPU case is modeled as aggregate GPU memory capacity and does not model tensor-parallel/NVLink/PCIe peer-transfer details.
- Routing is estimated with uniform top-k expected unique experts because real route traces are not available without executing the model.
