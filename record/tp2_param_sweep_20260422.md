# TP=2 双卡参数摸底记录

## 目的

在双卡 `TP=2` 已经恢复可用后，继续做一轮较小规模参数摸底，判断当前性能瓶颈主要来自哪里，并为后续正式双卡实验选出更合适的服务参数。

本轮重点比较两个方向：

1. 降低 `cpu_offload_gb`
2. 打开 `enforce_eager`

## 摸底方法

- 数据集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 设备：`GPU 0,1`
- `CUDA_VISIBLE_DEVICES=0,1`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.70`
- `max_model_len=1024`
- `disable_custom_all_reduce=True`

对比方式分两步：

1. 先在 `medium` 下比较不同参数组合
2. 选出更优组合后，再在 `high` 下验证

## 新增脚本

为了让后续双卡参数摸底可以重复运行，新增：

- [run_tp2_param_sweep.sh](/root/workspace/scripts/run_tp2_param_sweep.sh)

这个脚本会复用 [run_stage1_dual_gpu.sh](/root/workspace/scripts/run_stage1_dual_gpu.sh)，并按预设 case 依次运行：

- `baseline_graph`
- `offload4_graph`
- `offload8_eager`
- `offload4_eager`

本轮实际使用了其中两组进行 `medium` 对比，并对优胜组补跑了 `high`。

## 对比结果

### 1. 参考基线：`offload8_graph`

这里复用前一轮双卡探针结果作为参考基线。

- `medium`
  - [tp2_fcfs_medium_probe_20260422T134941Z_fcfs_medium_summary.json](/root/workspace/results/tp2_fcfs_medium_probe_20260422T134941Z_fcfs_medium_summary.json)
  - `throughput_tokens_per_s = 6.0784`
  - `avg_ttft = 1.7222 s`
  - `avg_tpot = 0.2704 s`
  - 启动阶段 `init engine = 75.34 s`
- `high`
  - [tp2_fcfs_high_probe_20260422T135531Z_fcfs_high_summary.json](/root/workspace/results/tp2_fcfs_high_probe_20260422T135531Z_fcfs_high_summary.json)
  - `throughput_tokens_per_s = 8.1765`
  - `avg_ttft = 2.7578 s`
  - `avg_tpot = 0.5781 s`
  - 启动阶段 `init engine = 70.69 s`

### 2. `offload4_graph`

结果文件：

- `medium`
  - [tp2_offload4_graph_medium_20260422T140716Z_fcfs_medium_summary.json](/root/workspace/results/tp2_offload4_graph_medium_20260422T140716Z_fcfs_medium_summary.json)
  - [tp2_offload4_graph_medium_20260422T140716Z_gpu_summary.json](/root/workspace/logs/processed/tp2_offload4_graph_medium_20260422T140716Z_gpu_summary.json)
  - [tp2_offload4_graph_medium_20260422T140716Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_offload4_graph_medium_20260422T140716Z_dual_gpu_run.md)
- `high`
  - [tp2_offload4_graph_high_20260422T141514Z_fcfs_high_summary.json](/root/workspace/results/tp2_offload4_graph_high_20260422T141514Z_fcfs_high_summary.json)
  - [tp2_offload4_graph_high_20260422T141514Z_gpu_summary.json](/root/workspace/logs/processed/tp2_offload4_graph_high_20260422T141514Z_gpu_summary.json)
  - [tp2_offload4_graph_high_20260422T141514Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_offload4_graph_high_20260422T141514Z_dual_gpu_run.md)

指标摘要：

- `medium`
  - `throughput_tokens_per_s = 11.6342`
  - `avg_ttft = 0.8750 s`
  - `p95_ttft = 0.9749 s`
  - `avg_tpot = 0.1413 s`
  - `p99_tpot = 0.1649 s`
  - 启动阶段 `init engine = 44.09 s`
- `high`
  - `throughput_tokens_per_s = 15.8497`
  - `avg_ttft = 1.3648 s`
  - `p95_ttft = 1.6829 s`
  - `avg_tpot = 0.2990 s`
  - `p99_tpot = 0.3506 s`
  - 启动阶段 `init engine = 39.80 s`

双卡显存摘要：

- `medium`
  - `GPU 0 peak = 21949 MiB`
  - `GPU 1 peak = 21935 MiB`
- `high`
  - `GPU 0 peak = 21949 MiB`
  - `GPU 1 peak = 21935 MiB`

日志关键信息：

- `Total CPU offloaded parameters: 4.09`
- `Available KV cache memory: 6.42 GiB`

### 3. `offload8_eager`

结果文件：

- [tp2_offload8_eager_medium_20260422T141028Z_fcfs_medium_summary.json](/root/workspace/results/tp2_offload8_eager_medium_20260422T141028Z_fcfs_medium_summary.json)
- [tp2_offload8_eager_medium_20260422T141028Z_gpu_summary.json](/root/workspace/logs/processed/tp2_offload8_eager_medium_20260422T141028Z_gpu_summary.json)
- [tp2_offload8_eager_medium_20260422T141028Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_offload8_eager_medium_20260422T141028Z_dual_gpu_run.md)

指标摘要：

- `throughput_tokens_per_s = 6.0956`
- `avg_ttft = 1.7689 s`
- `p95_ttft = 2.7177 s`
- `avg_tpot = 0.2663 s`
- `p99_tpot = 0.3120 s`
- 启动阶段 `init engine = 7.09 s`

双卡显存摘要：

- `GPU 0 peak = 17663 MiB`
- `GPU 1 peak = 17649 MiB`

日志关键信息：

- `Total CPU offloaded parameters: 8.01`
- `Available KV cache memory: 10.54 GiB`

## 结果对比与判断

### `medium` 下的结论

- `offload8_graph`: `6.0784 tok/s`
- `offload8_eager`: `6.0956 tok/s`
- `offload4_graph`: `11.6342 tok/s`

判断：

1. `offload8_eager` 和原始 `offload8_graph` 在请求期性能上几乎没有本质差别。
2. `enforce_eager` 显著改善了启动时间，但没有显著改善请求期吞吐和时延。
3. `offload4_graph` 的吞吐和时延都明显优于另外两组，说明当前双卡主瓶颈更像是 `CPU offload` 带来的数据搬运成本。

### `high` 下的结论

- 旧双卡基线 `offload8_graph`: `8.1765 tok/s`
- 新候选 `offload4_graph`: `15.8497 tok/s`
- 单卡高负载参考：`9.3556 tok/s`

判断：

1. 把 `cpu_offload_gb` 从 `8` 降到 `4` 后，双卡高负载吞吐几乎翻倍。
2. 新双卡高负载结果明显优于旧双卡基线，也明显优于当前单卡高负载结果。
3. 这说明双卡本身并不是没有优势，而是之前被过重的 offload 开销掩盖了。

## 当前最优候选

基于当前这轮摸底，最值得进入下一阶段正式实验的双卡配置是：

- `CUDA_VISIBLE_DEVICES=0,1`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.70`
- `cpu_offload_gb=4`
- `max_model_len=1024`
- `disable_custom_all_reduce=True`
- `enforce_eager=False`

## 当前结论

可以明确记录三点：

1. 双卡正式可用，而且已经找到一组明显优于旧双卡基线的参数。
2. 当前双卡性能问题的主因不是 CUDA graph，而是 `cpu_offload_gb=8` 过大导致的请求期搬运开销。
3. 后续正式双卡 `FCFS` 全量实验，应优先基于 `offload4_graph` 继续展开。

## 下一步

下一步建议直接进入双卡正式基线实验准备：

- 固定 `offload4_graph` 作为双卡默认配置
- 重新做双卡 `low / medium / high` 三档校准
- 然后再跑正式 `FCFS` 全量实验
