# Phase 0 Pre-HRM Result: Qwen1.5-MoE-A2.7B

Date: 2026-04-27

## Workflow

Implemented Phase 0 preliminary estimator:

- `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py`

Important correction:

```text
This is not a full implementation of MoE-Lightning's HRM.
It is a pre-HRM capacity/transfer screen.
```

The full MoE-Lightning HRM models hierarchical roofs and searches policies over batch size, micro-batch size, CPU/GPU placement for attention and MoE FFN, GPU-resident weight ratio, GPU-resident KV cache ratio, and CPU/GPU memory constraints. The current Qwen result only estimates model footprint, all-resident feasibility, and rough expert transfer pressure under simplified assumptions.

Inputs:

- Model config: `qwen1.5-MoE-A2.7B/config.json`
- Environment inventory: `results/wrec/env_inventory_20260427.json`

Command:

```bash
python scripts/moe_affinity/estimate_moe_hrm_bottleneck.py \
  --model-path qwen1.5-MoE-A2.7B \
  --model-name qwen1.5-moe-a2.7b \
  --output-json results/wrec/hrm_bottleneck_matrix_qwen1_5_moe_a2_7b_20260427.json \
  --output-csv results/wrec/hrm_bottleneck_matrix_qwen1_5_moe_a2_7b_20260427.csv
```

Generated:

- `results/wrec/hrm_bottleneck_matrix_qwen1_5_moe_a2_7b_20260427.json`
- `results/wrec/hrm_bottleneck_matrix_qwen1_5_moe_a2_7b_20260427.csv`

## Assumptions

This is a config-based pre-HRM estimate, not measured profiling and not the complete MoE-Lightning HRM.

Default matrix:

- dtype: BF16, INT8, INT4
- GPU memory budget: one GPU `23.988GiB`, two GPUs `47.977GiB`
- KV cache budget: `4GiB`, `8GiB`, `16GiB`
- bandwidth: `8GB/s`, `16GB/s`, `32GB/s`
- resident expert fractions: `10%`, `25%`, `50%`, `100%`
- active decode tokens: `1`, `8`, `32`
- effective compute throughput: `40 TFLOP/s`

Before route traces exist, expected active experts use a uniform-access approximation.

## Model Profile

From config:

```text
model_type: qwen2_moe
num_hidden_layers: 24
num_experts: 60
num_experts_per_tok: 4
hidden_size: 2048
moe_intermediate_size: 1408
shared_expert_intermediate_size: 5632
```

Estimated parameters:

```text
expert params per expert: 8,650,752
total routed expert params: 12,457,082,880
dense + shared-expert + embedding params: 1,858,502,656
total params: 14,315,585,536
active params per token: 2,688,974,848
```

Estimated weight footprint:

| dtype | total GiB | routed expert GiB | non-routed/dense GiB |
| --- | ---: | ---: | ---: |
| BF16 | 26.66 | 23.20 | 3.46 |
| INT8 | 13.33 | 11.60 | 1.73 |
| INT4 | 6.67 | 5.80 | 0.87 |

## Key Results

BF16 all-resident feasibility:

| GPU budget | KV budget | all experts resident feasible | model total |
| ---: | ---: | --- | ---: |
| 23.988GiB | 4GiB | no | 26.66GiB |
| 23.988GiB | 8GiB | no | 26.66GiB |
| 23.988GiB | 16GiB | no | 26.66GiB |
| 47.977GiB | 4GiB | yes | 26.66GiB |
| 47.977GiB | 8GiB | yes | 26.66GiB |
| 47.977GiB | 16GiB | yes | 26.66GiB |

Matrix summary:

```text
all-resident-feasible rows: 504
expert-transfer-bound rows: 144
```

The expert-transfer-bound rows are the single-GPU constrained/offload cases. On the current two-GPU capacity budget, Qwen1.5-MoE-A2.7B can be treated as all-resident feasible at this HRM level.

Single-GPU BF16 forced-offload example:

```text
gpu_memory = 23.988GiB
kv_cache = 8GiB
bandwidth = 16GB/s
active_tokens = 1
```

| requested resident fraction | actual resident experts/layer | transfer ms/step | transfer ratio |
| ---: | ---: | ---: | ---: |
| 10% | 6 | 93.43 | 0.999 |
| 25% | 15 | 77.86 | 0.998 |
| 50% | 30 | 51.90 | 0.997 |
| 100% | 32 | 48.44 | 0.997 |

## Conclusion

Qwen1.5-MoE-A2.7B is useful for Phase 0 and later trace/simulator debugging, but it is not a strong main WREC model on the current two-GPU platform.

Reason:

- On one 24GB GPU, BF16 all-resident is not feasible, and forced offload would be expert-transfer-bound.
- On the actual two-GPU platform, the estimated BF16 model footprint is only `26.66GiB`, so the full model fits within the combined `47.977GiB` GPU memory budget even with up to `16GiB` reserved for KV cache.
- Therefore, Qwen1.5-MoE-A2.7B does not naturally create the intended resource-limited expert cache problem on this platform unless we artificially restrict to one GPU or impose a much smaller expert cache budget.

Phase 0 decision:

```text
Use Qwen1.5-MoE-A2.7B for toolchain validation.
Do not use it as the main WREC evidence model on 2x3090 Ti.
Continue to Mixtral/scale-out HRM when model metadata or weights are available.
```

## Gap To MoE-Lightning HRM

The current estimator does not yet model:

- Policy search over `(N, micro-batch size, A_g, F_g, r_w, r_c)`.
- CPU attention vs GPU attention placement.
- CPU FFN vs GPU FFN placement.
- Per-layer decode latency:

```text
T(M,H,W,P) = max(comm_cpu_to_gpu, T_cpu, T_gpu)
```

- Separate CPU/GPU compute roofs and CPU/GPU memory roofs.
- CPU DRAM bandwidth, GPU HBM bandwidth, and CPU-GPU bandwidth as simultaneous hierarchical roofs.
- KV cache placement ratio and KV transfer/CPU attention bottlenecks.
- Pipeline overlap and CGOPipe-style transfer scheduling.

Therefore, this result should be used only as a first-pass capacity screen for Qwen, not as a claim that the full HRM analysis has been reproduced.
