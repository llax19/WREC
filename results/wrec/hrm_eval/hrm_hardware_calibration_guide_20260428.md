# HRM Hardware Calibration Guide

Date: 2026-04-28

## Goal

Replace the default HRM constants with measured hardware values:

```text
b_g  -> gpu_bandwidth_gbps
b_c  -> cpu_bandwidth_gbps
b_cg -> cpu_gpu_bandwidth_gbps
p_g  -> gpu_tflops
p_c  -> cpu_tflops
```

These constants are used by:

- `scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py`

## What Each Value Means

### `b_g`: GPU local memory bandwidth

Meaning:

- Effective GPU HBM/GDDR bandwidth for local GPU memory accesses.

How to measure:

- Allocate two large GPU tensors.
- Repeatedly run device-to-device copy.
- Use CUDA events for timing.
- Count bytes as read + write, so effective bandwidth is:

```text
b_g = 2 * tensor_bytes / copy_time
```

Script section:

- `gpu_bandwidth`

### `b_c`: CPU memory bandwidth

Meaning:

- Effective CPU DRAM bandwidth.

How to measure:

- Run STREAM-like NumPy copy and triad benchmarks.
- Copy counts one read + one write.
- Triad counts two reads + one write.
- Use the larger median of copy/triad as `b_c`.

Script section:

- `cpu_bandwidth`

### `b_cg`: CPU-to-GPU transfer bandwidth

Meaning:

- Effective host-to-device transfer bandwidth.

How to measure:

- Allocate CPU tensor and GPU tensor.
- Time CPU -> GPU copies using CUDA events.
- Prefer pinned CPU memory result if available.

Script section:

- `cpu_gpu_bandwidth`

### `p_g`: GPU compute throughput

Meaning:

- Effective GPU matrix multiplication throughput.

How to measure:

- Run large CUDA matmul.
- Use:

```text
p_g = 2 * n^3 / matmul_time
```

Recommended dtype:

- Use `fp16` for RTX 3090 Ti tensor-core throughput.
- Use `tf32` if the target kernels use TF32.

Script section:

- `gpu_flops`

### `p_c`: CPU compute throughput

Meaning:

- Effective CPU matrix multiplication throughput.

How to measure:

- Run NumPy FP32 matmul.
- Use:

```text
p_c = 2 * n^3 / matmul_time
```

Script section:

- `cpu_flops`

## Script

Added:

- `scripts/runtime/benchmark_hrm_hardware.py`

Run:

```bash
python scripts/runtime/benchmark_hrm_hardware.py \
  --output results/wrec/hrm_hardware_profile_20260428.json \
  --repeats 7 \
  --cpu-bytes-mib 512 \
  --gpu-bytes-mib 512 \
  --cpu-matmul-size 1536 \
  --gpu-matmul-size 4096 \
  --gpu-flops-dtype fp16
```

The script writes:

```json
{
  "recommended_hrm_constants": {
    "cpu_bandwidth_gbps": "...",
    "cpu_tflops": "...",
    "gpu_bandwidth_gbps": "...",
    "cpu_gpu_bandwidth_gbps": "...",
    "gpu_tflops": "..."
  }
}
```

If PyTorch/CUDA is unavailable, GPU-related sections are marked skipped instead of failing the whole benchmark.

## Current Full Measurement

The full measurement was run in the `vllm_moe` conda environment, where PyTorch and CUDA are available.

Command:

```bash
conda run -n vllm_moe python scripts/runtime/benchmark_hrm_hardware.py \
  --output results/wrec/hrm_hardware_profile_20260428.json \
  --repeats 7 \
  --cpu-bytes-mib 512 \
  --gpu-bytes-mib 512 \
  --cpu-matmul-size 1536 \
  --gpu-matmul-size 4096 \
  --gpu-flops-dtype fp16
```

Output:

- `results/wrec/hrm_hardware_profile_20260428.json`

Recommended HRM constants:

```text
b_c  = 31.13 GB/s
p_c  = 0.93 TFLOP/s
b_g  = 893.93 GB/s
b_cg = 6.50 GB/s
p_g  = 72.36 TFLOP/s
```

Measurement details:

```text
CPU copy median bandwidth     = 31.13 GB/s
CPU triad median bandwidth    = 10.12 GB/s
CPU matmul median throughput  = 0.93 TFLOP/s
GPU local copy bandwidth      = 893.93 GB/s
CPU -> GPU pinned H2D         = 6.50 GB/s
CPU -> GPU pageable H2D       = 6.29 GB/s
GPU fp16 matmul throughput    = 72.36 TFLOP/s
```

## How To Use The Measured Values

`evaluate_moe_lightning_hrm_policy.py` now accepts:

```bash
--hardware-profile results/wrec/hrm_hardware_profile_20260428.json
```

Example with the full profile:

```bash
conda run -n vllm_moe python scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py \
  --model-path qwen1.5-MoE-A2.7B \
  --model-name qwen1.5-moe-a2.7b \
  --hardware-profile results/wrec/hrm_hardware_profile_20260428.json \
  --output-json results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_calibrated_20260428.json \
  --output-csv results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_calibrated_20260428.csv
```

Generated:

- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_calibrated_20260428.json`
- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_calibrated_20260428.csv`
- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_calibrated_20260428.json`
- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_calibrated_20260428.csv`

## Calibrated HRM Result

With the measured hardware profile, the current 2-GPU capacity budget still selects an all-GPU policy:

```text
total policies: 756
feasible policies: 756
bottleneck counts:
  cpu-compute-or-memory: 149
  cpu-gpu-transfer: 555
  gpu-compute-or-memory: 52
best throughput policy:
  N = 64
  mu = 64
  A_g = 1
  F_g = 1
  r_w = 1.0
  r_c = 1.0
  layer_latency_ms = 1.5636
  bottleneck = gpu-compute-or-memory
```

With a single RTX 3090 Ti capacity budget, full residency is infeasible for many policies, and the calibrated HRM selects a partial expert-residency policy:

```text
total policies: 756
feasible policies: 587
infeasible-gpu-memory policies: 169
bottleneck counts:
  cpu-compute-or-memory: 134
  cpu-gpu-transfer: 432
  gpu-compute-or-memory: 21
best throughput policy:
  N = 8
  mu = 8
  A_g = 1
  F_g = 1
  r_w = 0.5
  r_c = 1.0
  layer_latency_ms = 0.6124
  bottleneck = gpu-compute-or-memory
```

Conclusion:

- The calibrated `b_cg` is only 6.50 GB/s, so CPU-GPU transfer is more expensive than the previous default assumption.
- The calibrated `p_g` is 72.36 TFLOP/s, so GPU execution is more attractive than the previous default assumption.
- For Qwen1.5-MoE-A2.7B, the 2-GPU setup still does not create the WREC target bottleneck. The single-GPU constrained setup does create meaningful cache/offload tradeoffs and is the better Phase 0 stress setting for this model.
