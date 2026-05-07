# TP=2 双卡探针记录

## 目的

在确认 `GPU 0` 恢复可用后，先做一轮较小规模的双卡探针，回答三个问题：

1. `Qwen1.5-MoE-A2.7B` 是否能在当前机器上稳定以 `tensor_parallel_size=2` 启动。
2. 双卡显存是否能稳定、均衡地分布在 `GPU 0/1`。
3. 双卡 `TP=2` 在当前请求驱动下，是否已经明显优于单卡基线。

## 本轮使用的配置

- 模型：`/root/workspace/qwen1.5-MoE-A2.7B`
- 数据集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 物理 GPU：`0,1`
- `CUDA_VISIBLE_DEVICES=0,1`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.70`
- `cpu_offload_gb=8`
- `max_model_len=1024`
- `disable_custom_all_reduce=True`
- `enforce_eager=False`

补充说明：

- 本轮没有继续使用单卡阶段的 `--enforce-eager + max_model_len=512` 生存参数。
- 选择 `disable_custom_all_reduce=True`，是因为当前机器的 P2P 检测没有通过，显式关闭后日志更干净，行为也更稳定。

## 先做的 smoke 验证

先运行了：

```bash
bash /root/workspace/scripts/run_vllm_smoke_tp2.sh
```

观察到的关键现象：

- 双卡 `NCCL` 初始化成功。
- 权重可以正常加载。
- 日志中给出了正数的 KV cache 空间：
  - `Available KV cache memory: 10.13 GiB`
- 日志中给出了较高的理论并发上限：
  - `Maximum concurrency for 1,024 tokens per request: 108.06x`
- 最终成功返回了一条生成结果，没有出现 `No available memory for the cache blocks`。

这说明当前双卡 `TP=2` 路径已经具备继续做正式实验的基本条件。

## 新增脚本

为了把双卡实验也纳入与单卡相同的日志链路，新增了：

- [run_stage1_dual_gpu.sh](/root/workspace/scripts/run_stage1_dual_gpu.sh)

这个脚本会自动完成：

- 启动双卡 `vLLM` 服务
- 同时监控 `GPU 0/1` 显存
- 等待服务 ready
- 发送请求并记录请求级日志
- 汇总双卡 GPU summary
- 生成本轮运行说明文件

## 已完成的双卡探针

### 1. `FCFS + medium`

结果文件：

- [tp2_fcfs_medium_probe_20260422T134941Z_fcfs_medium_summary.json](/root/workspace/results/tp2_fcfs_medium_probe_20260422T134941Z_fcfs_medium_summary.json)
- [tp2_fcfs_medium_probe_20260422T134941Z_gpu_summary.json](/root/workspace/logs/processed/tp2_fcfs_medium_probe_20260422T134941Z_gpu_summary.json)
- [tp2_fcfs_medium_probe_20260422T134941Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_fcfs_medium_probe_20260422T134941Z_dual_gpu_run.md)
- [tp2_fcfs_medium_probe_20260422T134941Z_vllm_server.log](/root/workspace/logs/server/tp2_fcfs_medium_probe_20260422T134941Z_vllm_server.log)

指标摘要：

- `throughput_tokens_per_s = 6.0784`
- `avg_ttft = 1.7222 s`
- `p95_ttft = 1.9857 s`
- `avg_tpot = 0.2704 s`
- `p99_tpot = 0.3076 s`

双卡显存摘要：

- `GPU 0 peak = 21987 MiB`
- `GPU 1 peak = 21973 MiB`

### 2. `FCFS + high`

结果文件：

- [tp2_fcfs_high_probe_20260422T135531Z_fcfs_high_summary.json](/root/workspace/results/tp2_fcfs_high_probe_20260422T135531Z_fcfs_high_summary.json)
- [tp2_fcfs_high_probe_20260422T135531Z_gpu_summary.json](/root/workspace/logs/processed/tp2_fcfs_high_probe_20260422T135531Z_gpu_summary.json)
- [tp2_fcfs_high_probe_20260422T135531Z_dual_gpu_run.md](/root/workspace/logs/runs/tp2_fcfs_high_probe_20260422T135531Z_dual_gpu_run.md)
- [tp2_fcfs_high_probe_20260422T135531Z_vllm_server.log](/root/workspace/logs/server/tp2_fcfs_high_probe_20260422T135531Z_vllm_server.log)

指标摘要：

- `throughput_tokens_per_s = 8.1765`
- `avg_ttft = 2.7578 s`
- `p95_ttft = 3.2731 s`
- `avg_tpot = 0.5781 s`
- `p99_tpot = 0.6913 s`

双卡显存摘要：

- `GPU 0 peak = 21963 MiB`
- `GPU 1 peak = 21949 MiB`

## 与单卡探针的直接对比

当前能直接对齐的单卡结果是同样 `24` 条请求下的单卡校准结果。

### `medium`

- 单卡：
  - `throughput_tokens_per_s = 6.0015`
  - `avg_ttft = 1.7905 s`
  - `avg_tpot = 0.2728 s`
- 双卡：
  - `throughput_tokens_per_s = 6.0784`
  - `avg_ttft = 1.7222 s`
  - `avg_tpot = 0.2704 s`

判断：

- 双卡 `medium` 只有小幅改善，说明中等负载下还没有把双卡潜力完全打出来。

### `high`

- 单卡：
  - `throughput_tokens_per_s = 9.3556`
  - `avg_ttft = 2.5581 s`
  - `avg_tpot = 0.4991 s`
- 双卡：
  - `throughput_tokens_per_s = 8.1765`
  - `avg_ttft = 2.7578 s`
  - `avg_tpot = 0.5781 s`

判断：

- 在当前 `high` 探针下，双卡结果反而弱于单卡。

## 初步原因判断

结合结果文件和服务日志，当前最值得怀疑的不是显存，而是通信与执行路径开销：

1. 双卡显存非常均衡，峰值都接近 `21.95~21.99 GiB`，说明模型切分本身是正常的。
2. `Available KV cache memory` 约为 `10.34 GiB`，也说明双卡没有遭遇单卡早期那种 KV cache 负数问题。
3. 但当前机器的自定义 all-reduce 已经显式关闭，说明 `TP=2` 下通信路径并不理想。
4. 同时仍然保留了 `cpu_offload_gb=8`，这会引入额外的数据搬运成本。
5. 因此当前双卡虽然“能跑稳”，但未必已经进入“比单卡更快”的甜蜜区。

## 当前结论

可以先明确三个结论：

1. 双卡 `TP=2` 现在已经跑通了，不再是不可用状态。
2. 现有双卡实验链路已经具备继续扩展的条件，可以直接承接下一轮正式双卡实验。
3. 但当前参数下，双卡收益并没有自然出现，所以不能直接默认“双卡结果一定优于单卡”。

## 建议的下一步

下一步不要直接开全量三档正式实验，而是先做一轮双卡参数摸底：

- 比较 `cpu_offload_gb=8` 与更小 offload 的影响
- 比较 `enforce_eager=False` 与 `enforce_eager=True` 的影响
- 固定 `medium/high`，先看双卡吞吐能不能稳定超过单卡

只有在双卡参数摸底后，再进入正式 `FCFS` 全量实验，才更稳妥。
