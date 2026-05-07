# TP=2 双卡三档负载校准记录

## 目的

在完成双卡参数摸底并选出当前最优候选配置后，继续做一轮正式的双卡三档负载校准，确认：

1. `low / medium / high` 三档在双卡 `TP=2` 配置下是否仍然具有清晰区分度。
2. 当前双卡推荐参数是否已经能够稳定承接正式 `FCFS` 基线实验。
3. 双卡三档结果相对于现有单卡基线是否已经形成稳定优势。

## 本轮使用的配置

- 模型：`/root/workspace/qwen1.5-MoE-A2.7B`
- 数据集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 重复次数：每档 `1` 次
- 设备：`GPU 0,1`
- `CUDA_VISIBLE_DEVICES=0,1`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.70`
- `cpu_offload_gb=4`
- `max_model_len=1024`
- `disable_custom_all_reduce=True`
- `enforce_eager=False`

负载定义：

- `low`: `arrival_interval=8.0`, `max_concurrency=1`
- `medium`: `arrival_interval=1.5`, `max_concurrency=2`
- `high`: `arrival_interval=0.0`, `max_concurrency=6`

## 使用脚本

本轮使用新增脚本：

- [run_tp2_load_levels.sh](/root/workspace/scripts/run_tp2_load_levels.sh)

它会复用：

- [run_stage1_dual_gpu.sh](/root/workspace/scripts/run_stage1_dual_gpu.sh)

## 结果文件

- `low`
  - [tp2_fcfs_low_r1_20260422T142118Z_fcfs_low_summary.json](/root/workspace/results/tp2_fcfs_low_r1_20260422T142118Z_fcfs_low_summary.json)
  - [tp2_fcfs_low_r1_20260422T142118Z_gpu_summary.json](/root/workspace/logs/processed/tp2_fcfs_low_r1_20260422T142118Z_gpu_summary.json)
  - [tp2_fcfs_low_r1_20260422T142118Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_fcfs_low_r1_20260422T142118Z_dual_gpu_run.md)
- `medium`
  - [tp2_fcfs_medium_r1_20260422T142548Z_fcfs_medium_summary.json](/root/workspace/results/tp2_fcfs_medium_r1_20260422T142548Z_fcfs_medium_summary.json)
  - [tp2_fcfs_medium_r1_20260422T142548Z_gpu_summary.json](/root/workspace/logs/processed/tp2_fcfs_medium_r1_20260422T142548Z_gpu_summary.json)
  - [tp2_fcfs_medium_r1_20260422T142548Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_fcfs_medium_r1_20260422T142548Z_dual_gpu_run.md)
- `high`
  - [tp2_fcfs_high_r1_20260422T142857Z_fcfs_high_summary.json](/root/workspace/results/tp2_fcfs_high_r1_20260422T142857Z_fcfs_high_summary.json)
  - [tp2_fcfs_high_r1_20260422T142857Z_gpu_summary.json](/root/workspace/logs/processed/tp2_fcfs_high_r1_20260422T142857Z_gpu_summary.json)
  - [tp2_fcfs_high_r1_20260422T142857Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_fcfs_high_r1_20260422T142857Z_dual_gpu_run.md)

## 双卡三档指标摘要

- `low`
  - `throughput_tokens_per_s = 6.8073`
  - `avg_ttft = 0.6448 s`
  - `p95_ttft = 0.7200 s`
  - `avg_tpot = 0.1049 s`
  - `p99_tpot = 0.1159 s`
  - `wall_time = 189.94 s`
- `medium`
  - `throughput_tokens_per_s = 11.6206`
  - `avg_ttft = 0.8758 s`
  - `p95_ttft = 0.9757 s`
  - `avg_tpot = 0.1415 s`
  - `p99_tpot = 0.1650 s`
  - `wall_time = 109.72 s`
- `high`
  - `throughput_tokens_per_s = 15.6244`
  - `avg_ttft = 1.4612 s`
  - `p95_ttft = 2.2554 s`
  - `avg_tpot = 0.3013 s`
  - `p99_tpot = 0.3578 s`
  - `wall_time = 82.12 s`

## GPU 摘要

三档下双卡显存都非常稳定，且两张卡分布均衡：

- `low`
  - `GPU 0 peak = 21949 MiB`
  - `GPU 1 peak = 21935 MiB`
- `medium`
  - `GPU 0 peak = 21949 MiB`
  - `GPU 1 peak = 21935 MiB`
- `high`
  - `GPU 0 peak = 21949 MiB`
  - `GPU 1 peak = 21935 MiB`

## 服务日志关键信息

三档服务日志的核心信号一致：

- `Total CPU offloaded parameters: 4.09`
- `Available KV cache memory: 6.42 GiB`
- `init engine (profile, create kv cache, warmup model)` 约为 `39.7~39.9 s`

这说明当前双卡推荐参数在三档负载下都表现稳定，没有出现新的启动或 KV cache 风险。

## 与单卡三档校准的对比

单卡参考结果来自此前的：

- [fcfs_load_calibration_20260422.md](/root/workspace/record/fcfs_load_calibration_20260422.md)

对比摘要如下：

- `low`
  - 单卡：`throughput = 4.2189`, `avg_ttft = 1.3236 s`, `avg_tpot = 0.1955 s`
  - 双卡：`throughput = 6.8073`, `avg_ttft = 0.6448 s`, `avg_tpot = 0.1049 s`
- `medium`
  - 单卡：`throughput = 6.0015`, `avg_ttft = 1.7905 s`, `avg_tpot = 0.2728 s`
  - 双卡：`throughput = 11.6206`, `avg_ttft = 0.8758 s`, `avg_tpot = 0.1415 s`
- `high`
  - 单卡：`throughput = 9.3556`, `avg_ttft = 2.5581 s`, `avg_tpot = 0.4991 s`
  - 双卡：`throughput = 15.6244`, `avg_ttft = 1.4612 s`, `avg_tpot = 0.3013 s`

## 结论

本轮校准可以明确得出以下结论：

1. 当前双卡推荐参数下，`low / medium / high` 三档区分度清晰：
   - 吞吐随负载提升而持续上升
   - `TTFT / TPOT` 也随负载提升而持续恶化
2. 双卡三档结果已经整体稳定优于单卡三档基线。
3. 双卡显存稳定、分布均衡，说明当前服务配置已经足够进入正式实验阶段。
4. 因此后续正式双卡 `FCFS` 全量实验可以直接沿用当前这组参数继续展开。

## 下一步

下一步建议直接进入正式双卡 `FCFS` 基线实验：

- 固定当前双卡配置
- 使用完整 `240` 条请求
- `low / medium / high` 三档
- 每档至少重复 `3` 次
