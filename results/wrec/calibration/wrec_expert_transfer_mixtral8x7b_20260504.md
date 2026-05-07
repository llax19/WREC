# WREC Expert Transfer Microbenchmark

## Setup

- Device: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- CUDA device: `cuda:0`
- Model path: `/root/workspace/models/Mixtral-8x7B-Instruct-v0.1`
- Dtype for expert-size inference: `float16`
- Expert bytes: `352321536`
- Expert size: `336.00 MiB`
- Repeats / warmups: `100` / `5`

## Simulator Reference

- Assumed bandwidth: `41.370000 GB/s`
- Assumed transfer time: `8.516353 ms/expert`

## Measurements

| host memory | available | median ms | mean ms | p90 ms | median GB/s | mean GB/s | ratio vs assumed bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|
| pageable | yes | 23.455377 | 23.849332 | 24.816227 | 15.020929 | 14.791594 | 0.3631 |
| pinned | yes | 8.467264 | 8.468275 | 8.485136 | 41.609846 | 41.605076 | 1.0058 |

## Conclusion

- This benchmark measures host-to-GPU copy latency for exactly one inferred Mixtral expert worth of bytes.
- The result calibrates the replay simulator's `expert_bytes / bandwidth` transfer-stall proxy.
- It is still a transfer microbenchmark, not an end-to-end serving latency measurement.
