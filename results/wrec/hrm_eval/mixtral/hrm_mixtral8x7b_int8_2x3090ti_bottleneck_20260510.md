# Mixtral INT8 HRM bottleneck check 20260510

## Scope

- Model: `Mixtral-8x7B-Instruct-v0.1`.
- Quantization assumption: INT8 weights, `1 byte/param`.
- Local hardware used for interpretation: `2 x NVIDIA GeForce RTX 3090 Ti`, about `47.98 GiB` aggregate GPU memory.
- HRM constants: `gpu_bandwidth_gbps=1451.13`, `cpu_gpu_bandwidth_gbps=41.37`, `gpu_tflops=382.45`; CPU bandwidth uses evaluator default `100 GB/s`, CPU compute uses `1.6 TFLOPS`.
- Workload: average prompt length `512`, generation length `128`.

## Outputs

- Policy evaluator JSON: `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int8_2x3090ti_20260510.json`.
- Policy evaluator CSV: `results/wrec/hrm_eval/mixtral/hrm_policy_eval_mixtral8x7b_int8_2x3090ti_20260510.csv`.
- Capacity/transfer screen JSON: `results/wrec/hrm_eval/mixtral/hrm_bottleneck_matrix_mixtral8x7b_int8_2x3090ti_20260510.json`.
- Capacity/transfer screen CSV: `results/wrec/hrm_eval/mixtral/hrm_bottleneck_matrix_mixtral8x7b_int8_2x3090ti_20260510.csv`.

## Key numbers

- INT8 total weights: `43.50 GiB`.
- Dense non-expert weights: `1.50 GiB`.
- Expert FFN weights: `42.00 GiB`.
- Per expert size: `168 MiB` in INT8.
- HRM policy search rows: `756/756` memory-feasible under aggregate `47.98 GiB`.
- Bottleneck counts in policy evaluator: `cpu-gpu-transfer=464`, `gpu-compute-or-memory=166`, `cpu-compute-or-memory=126`.

## Main finding

Under an aggregate 2-GPU memory model, INT8 Mixtral can barely fit all weights on GPU: best throughput policy keeps FFN weights fully on GPU (`F_g=1`, `r_w=1.0`) and leaves attention/KV on CPU (`A_g=0`, `r_c=0.0`). This gives a proxy layer latency of `0.972 ms` at `N=64`, with `T_gpu=0.972 ms`, `T_cpu=0.760 ms`, and CPU-GPU communication only `0.013 ms` per layer.

The practical bottleneck depends on KV/cache budget:

- If only `4 GiB` is reserved for KV/cache, all INT8 weights can be resident. The bottleneck shifts to GPU-side FFN compute/memory, not PCIe transfer.
- With `8 GiB` or more KV/cache budget, all weights no longer fit in the HRM capacity screen. Even one missing expert per layer makes CPU-GPU transfer dominate: at bandwidth `41.37 GB/s`, missing one INT8 expert per layer costs about `136 ms` per decode step at `64` active tokens, versus about `4.31 ms` estimated compute.
- On one 24 GiB 3090 Ti, every screened INT8 configuration remains expert-transfer-bound.

## Conclusion

The local INT8 Mixtral bottleneck is a capacity cliff. If the runtime can shard and keep almost all INT8 expert weights resident across both 3090 Ti GPUs, the remaining bottleneck is GPU-side FFN execution. Once KV/cache/runtime overhead pushes even a fraction of experts off GPU, the bottleneck becomes PCIe CPU-GPU expert transfer by a large margin. For real serving, the main risk is not raw INT8 weight size alone, but whether vLLM quantization, tensor parallel placement, KV cache, CUDA buffers, and fragmentation leave enough per-GPU headroom to avoid expert offload.

