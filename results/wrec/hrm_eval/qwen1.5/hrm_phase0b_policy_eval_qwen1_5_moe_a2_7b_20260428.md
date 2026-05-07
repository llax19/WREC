# Phase 0B HRM Policy Evaluation: Qwen1.5-MoE-A2.7B

Date: 2026-04-28

## What Was Implemented

Implemented a MoE-Lightning-style HRM policy evaluator:

- `scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py`

This script evaluates the policy tuple described in MoE-Lightning Section 4.2:

```text
P = (N, mu, A_g, F_g, r_w, r_c)
```

where:

```text
N: batch size
mu: micro-batch size
A_g: whether attention runs on GPU
F_g: whether MoE FFN runs on GPU
r_w: ratio of FFN weights resident on GPU
r_c: ratio of KV cache resident on GPU
```

For each policy, it estimates per-layer decode latency using:

```text
T(M, H, W, P) = max(comm_cpu_to_gpu, T_cpu, T_gpu)
```

with:

```text
T_gpu = T_gpu_attn + T_gpu_ffn
T_cpu = T_cpu_attn + T_cpu_ffn
T_x = max(comm_x, comp_x)
```

## Important Scope Note

This is a formula-level reproduction of the HRM policy-evaluation structure, not the original MoE-Lightning implementation. The paper does not provide source code or every kernel-specific FLOP/byte term, so the script uses explicit formulas documented below. Hardware calibration is still required before making final throughput claims.

## Commands

Current 2-GPU capacity budget:

```bash
python scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py \
  --model-path qwen1.5-MoE-A2.7B \
  --model-name qwen1.5-moe-a2.7b \
  --output-json results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_20260428.json \
  --output-csv results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_20260428.csv
```

Single-GPU constrained budget:

```bash
python scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py \
  --model-path qwen1.5-MoE-A2.7B \
  --model-name qwen1.5-moe-a2.7b \
  --gpu-memory-gb 23.988 \
  --output-json results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_20260428.json \
  --output-csv results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_20260428.csv
```

## Assumptions

Default hardware:

```text
GPU memory: 47.977GiB for current 2-GPU run; 23.988GiB for constrained single-GPU run
CPU memory: host MemTotal from /proc/meminfo
GPU bandwidth: 900GB/s
CPU bandwidth: 100GB/s
CPU-GPU bandwidth: 16GB/s
GPU compute: 40TFLOP/s
CPU compute: 1.6TFLOP/s
```

Default workload:

```text
average prompt length: 512
generation length: 128
average decode context length: 512 + 128 / 2 = 576
```

Default policy grid:

```text
N: 1, 4, 8, 16, 32, 64
mu: 1, 4, 8, 16, 32, 64, with mu <= N
A_g: 0, 1
F_g: 0, 1
r_w: 0, 0.25, 0.5, 0.75, 1
r_c: 0, 0.25, 0.5, 0.75, 1
```

Attention model:

- Decode attention core only.
- QK, AV, and softmax FLOPs are included.
- QKVO projection is omitted, matching the paper's Section 3.3 case-study note.
- If `A_g = 0`, attention runs on CPU and query/output transfer is counted across CPU-GPU.
- If `A_g = 1`, attention runs on GPU and `(1-r_c)` KV cache bytes are transferred from CPU to GPU.

MoE FFN model:

- SwiGLU FFN FLOPs are estimated from active routed experts plus shared expert when present.
- Expected active routed experts use a uniform top-k approximation before real route traces exist.
- If `F_g = 1`, FFN runs on GPU and offloaded active FFN weights are counted in CPU-GPU transfer.
- If `F_g = 0`, FFN runs on CPU and hidden-state transfer back to GPU is counted.

## Current 2-GPU Result

Output files:

- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_20260428.json`
- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_20260428.csv`

Policy rows:

```text
total policies: 756
feasible policies: 756
```

Bottleneck counts:

```text
cpu-compute-or-memory: 127
cpu-gpu-transfer: 511
gpu-compute-or-memory: 118
```

Best throughput policy:

```text
N = 64
mu = 64
A_g = 1
F_g = 1
r_w = 1.0
r_c = 1.0
layer_latency_ms = 1.5531
throughput_proxy = 41208 tokens/s/layer
bottleneck = gpu-compute-or-memory
gpu_memory_used = 34.96GiB
cpu_memory_used = 0.00GiB
```

Interpretation:

- Under the current two-GPU capacity budget, Qwen can keep weights and KV cache on GPU for the tested workload.
- HRM therefore chooses the all-GPU policy.
- This confirms Qwen is not a strong resource-constrained WREC evidence model on the current full platform.

## Single-GPU Constrained Result

Output files:

- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_20260428.json`
- `results/wrec/hrm_policy_eval_qwen1_5_moe_a2_7b_single_gpu_20260428.csv`

Policy rows:

```text
total policies: 756
feasible policies: 587
infeasible GPU-memory policies: 169
```

Bottleneck counts:

```text
cpu-compute-or-memory: 124
cpu-gpu-transfer: 409
gpu-compute-or-memory: 54
infeasible-gpu-memory: 169
```

Best throughput policy:

```text
N = 16
mu = 16
A_g = 0
F_g = 1
r_w = 0.75
r_c = 0.0
layer_latency_ms = 0.8480
throughput_proxy = 18867 tokens/s/layer
bottleneck = gpu-compute-or-memory
gpu_memory_used = 22.54GiB
cpu_memory_used = 7.88GiB
```

Best feasible policy by batch size:

| N | mu | A_g | F_g | r_w | r_c | latency ms | throughput proxy | bottleneck |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0 | 1 | 0.25 | 0.0 | 0.1538 | 6501.9 | gpu-compute-or-memory |
| 4 | 4 | 0 | 1 | 0.50 | 0.0 | 0.3551 | 11264.4 | gpu-compute-or-memory |
| 8 | 8 | 0 | 1 | 0.50 | 0.0 | 0.5662 | 14128.8 | gpu-compute-or-memory |
| 16 | 16 | 0 | 1 | 0.75 | 0.0 | 0.8480 | 18867.5 | gpu-compute-or-memory |
| 32 | 32 | 1 | 0 | 0.00 | 0.0 | 9.9343 | 3221.2 | cpu-compute-or-memory |
| 64 | 64 | 1 | 0 | 0.00 | 0.5 | 10.9527 | 5843.3 | cpu-compute-or-memory |

Interpretation:

- Under a single 24GB GPU budget, all-GPU policies become infeasible.
- HRM prefers CPU attention (`A_g=0`) and GPU FFN (`F_g=1`) for the best constrained-throughput policy, matching the qualitative MoE-Lightning claim that CPU attention can be beneficial under memory pressure.
- The preferred `r_w=0.75` means most FFN weights should be resident on GPU, while the remaining fraction is offloaded.
- For larger batches (`N=32,64`), the best feasible policy changes, and CPU compute/memory becomes the bottleneck.

## Conclusion

Compared with the earlier pre-HRM screen, this Phase 0B evaluator now captures the main HRM policy axes:

```text
N, mu, A_g, F_g, r_w, r_c
```

and uses the MoE-Lightning-style latency equation:

```text
T = max(comm_cpu_to_gpu, T_cpu, T_gpu)
```

For Qwen1.5-MoE-A2.7B:

1. On the full 2x3090 Ti platform, HRM chooses an all-GPU policy, so Qwen remains a debug/toolchain model rather than a main WREC evidence model.
2. Under a single-GPU constrained budget, HRM exposes meaningful policy tradeoffs and selects CPU attention + GPU FFN + partial GPU FFN residency.
3. The next useful step is hardware calibration: replace default `b_g`, `b_c`, `b_cg`, `p_g`, and `p_c` with measured values.
