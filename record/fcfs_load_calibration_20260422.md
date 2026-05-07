# FCFS 负载校准记录

## 目的

在正式运行 `FCFS` 三档基线实验之前，先用
[eval_requests.jsonl](/root/workspace/data/prompts/eval_requests.jsonl)
做一轮较小规模校准，确认 `low / medium / high` 的到达控制参数确实能在当前单卡环境下拉开吞吐与时延差异。

## 本轮使用的配置

- 策略：`FCFS`
- 数据集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 重复次数：每档 `1` 次
- GPU：`GPU 1`
- 服务参数：
  - `tensor_parallel_size=1`
  - `gpu_memory_utilization=0.90`
  - `cpu_offload_gb=8`
  - `max_model_len=512`
  - `--enforce-eager`

负载定义：

- `low`: `arrival_interval=8.0`, `max_concurrency=1`
- `medium`: `arrival_interval=1.5`, `max_concurrency=2`
- `high`: `arrival_interval=0.0`, `max_concurrency=6`

## 结果文件

- `low`
  - [fcfs_low_r1_20260422T130043Z_fcfs_low_summary.json](/root/workspace/results/fcfs_low_r1_20260422T130043Z_fcfs_low_summary.json)
  - [fcfs_low_r1_20260422T130043Z_gpu_summary.json](/root/workspace/logs/processed/fcfs_low_r1_20260422T130043Z_gpu_summary.json)
  - [fcfs_low_r1_20260422T130043Z_stage1_run.md](/root/workspace/logs/runs/fcfs_low_r1_20260422T130043Z_stage1_run.md)
- `medium`
  - [fcfs_medium_r1_20260422T130635Z_fcfs_medium_summary.json](/root/workspace/results/fcfs_medium_r1_20260422T130635Z_fcfs_medium_summary.json)
  - [fcfs_medium_r1_20260422T130635Z_gpu_summary.json](/root/workspace/logs/processed/fcfs_medium_r1_20260422T130635Z_gpu_summary.json)
  - [fcfs_medium_r1_20260422T130635Z_stage1_run.md](/root/workspace/logs/runs/fcfs_medium_r1_20260422T130635Z_stage1_run.md)
- `high`
  - [fcfs_high_r1_20260422T131050Z_fcfs_high_summary.json](/root/workspace/results/fcfs_high_r1_20260422T131050Z_fcfs_high_summary.json)
  - [fcfs_high_r1_20260422T131050Z_gpu_summary.json](/root/workspace/logs/processed/fcfs_high_r1_20260422T131050Z_gpu_summary.json)
  - [fcfs_high_r1_20260422T131050Z_stage1_run.md](/root/workspace/logs/runs/fcfs_high_r1_20260422T131050Z_stage1_run.md)

## 指标摘要

- `low`
  - `throughput_tokens_per_s = 4.2189`
  - `avg_ttft = 1.3236 s`
  - `p95_ttft = 1.5956 s`
  - `avg_tpot = 0.1955 s`
  - `p99_tpot = 0.2161 s`
  - GPU peak memory: `22275 MiB`
- `medium`
  - `throughput_tokens_per_s = 6.0015`
  - `avg_ttft = 1.7905 s`
  - `p95_ttft = 2.5364 s`
  - `avg_tpot = 0.2728 s`
  - `p99_tpot = 0.3090 s`
  - GPU peak memory: `22277 MiB`
- `high`
  - `throughput_tokens_per_s = 9.3556`
  - `avg_ttft = 2.5581 s`
  - `p95_ttft = 3.1952 s`
  - `avg_tpot = 0.4991 s`
  - `p99_tpot = 0.5950 s`
  - GPU peak memory: `22279 MiB`

## 结论

本轮校准说明当前三档负载定义是有效的：

1. `throughput_tokens_per_s` 随负载提高而持续上升。
2. `avg_ttft / p95_ttft / avg_tpot / p99_tpot` 也随负载提高而持续恶化。
3. 三档之间的时延和吞吐趋势清晰，可用于后续正式 `FCFS` 基线实验。
4. GPU 峰值显存三档之间差异很小，说明当前主要区分的是请求排队和服务竞争，而不是显存峰值继续显著增加。

## 下一步

下一步直接进入正式 `FCFS` 基线：

- 使用完整 `240` 条请求
- `low / medium / high` 三档
- 每档至少重复 `3` 次
