# Gap Analysis: Current Phase 0 Estimator vs MoE-Lightning HRM

Date: 2026-04-27

## Direct Answer

The current script:

- `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py`

is **not** a full implementation of the HRM used in MoE-Lightning.

It is only a **pre-HRM capacity/transfer screen**. It can answer:

- How large are routed experts and dense/shared parts?
- Can all experts fit under a GPU memory budget?
- If experts do not fit, how large could expert transfer pressure be under simple assumptions?

It cannot yet answer the full MoE-Lightning question:

> Given hardware, model, workload, and a candidate policy, what is the per-layer decode latency and which resource is the bottleneck under hierarchical CPU/GPU memory and compute roofs?

## What MoE-Lightning HRM Actually Models

From MoE-Lightning Section 4.2, the performance model takes:

```text
Hardware H:
  m_g, m_c: GPU and CPU memory
  b_g, b_c, b_cg: GPU, CPU, CPU-GPU bandwidth
  p_g, p_c: GPU and CPU FLOPS

Model M:
  l: number of layers
  h1, h2: model and intermediate hidden dimensions
  n_q, n_kv: query and key/value heads
  n_e, k: number of experts and top-k routing
  dt: data type

Workload W:
  s: average prompt length
  n: generation length

Policy P:
  N: batch size
  mu: micro-batch size
  A_g: whether attention runs on GPU
  F_g: whether MoE FFN runs on GPU
  r_w: ratio of weights stored on GPU
  r_c: ratio of KV cache stored on GPU
```

It estimates per-layer decode latency as:

```text
T(M, H, W, P) = max(comm_cpu_to_gpu, T_cpu, T_gpu)
```

where:

```text
T_gpu = T_gpu_attn + T_gpu_ffn
T_cpu = T_cpu_attn + T_cpu_ffn
T_x = max(comm_x, comp_x)
```

The key point is that HRM is not just checking whether weights fit. It searches for a policy that balances CPU compute, GPU compute, CPU memory bandwidth, GPU memory bandwidth, CPU-GPU transfer bandwidth, GPU memory capacity, CPU memory capacity, KV cache placement, and weight placement.

## What The Current Script Does

The current script reads a model config and estimates:

- routed expert parameter count.
- dense/shared/embedding parameter count.
- BF16/INT8/INT4 footprint.
- all-resident feasibility under GPU memory and KV reserve.
- expected unique experts per layer using a uniform-access approximation.
- estimated expert transfer bytes and transfer time.
- a rough transfer ratio:

```text
transfer_time / (transfer_time + estimated_compute_time)
```

This is useful for a first-pass screen, but it is not enough to claim HRM reproduction.

## Missing Pieces

The current script does not model:

1. Batch size `N`.
2. Micro-batch size `mu`.
3. CPU attention vs GPU attention.
4. CPU FFN vs GPU FFN.
5. GPU-resident weight ratio `r_w` as a searched policy variable.
6. GPU-resident KV cache ratio `r_c`.
7. CPU memory capacity constraint.
8. CPU memory bandwidth roof.
9. GPU HBM bandwidth roof.
10. CPU-GPU bandwidth roof as a simultaneous bottleneck with compute.
11. Per-layer `max(comm_cpu_to_gpu, T_cpu, T_gpu)` latency.
12. CGOPipe-style overlap between compute and transfers.
13. Workload prompt length / generation length effects on KV memory and attention cost.
14. Policy search or MILP-like optimization.

## Corrected Interpretation Of Qwen Result

The Qwen Phase 0 result should be interpreted as:

> Under a simplified pre-HRM capacity/transfer screen, Qwen1.5-MoE-A2.7B is all-resident feasible on the current 2x3090 Ti platform, so it is not a strong main WREC evidence model.

It should **not** be interpreted as:

> The full MoE-Lightning HRM has been reproduced for Qwen.

## Next Implementation Direction

To move closer to MoE-Lightning HRM, the next script should be a separate policy evaluator, not an extension hidden inside the current capacity screen.

Suggested script:

- `scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py`

Minimum inputs:

```text
model config
hardware profile: m_g, m_c, b_g, b_c, b_cg, p_g, p_c
workload profile: average prompt length, generation length
policy grid: N, mu, A_g, F_g, r_w, r_c
```

Minimum outputs:

```text
per-policy memory feasibility
per-policy T_cpu
per-policy T_gpu
per-policy comm_cpu_to_gpu
per-policy T = max(...)
bottleneck label
best feasible policy
```

This should become Phase 0B. The existing script should remain Phase 0A, a fast pre-HRM screen.
