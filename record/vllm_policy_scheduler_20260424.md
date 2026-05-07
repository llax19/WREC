# vLLM Policy Scheduler 记录

## 目的

把当前客户端代理版 `Length-only` 推进为服务端内嵌调度实现，避免实验结果只反映客户端请求重排。

## 修改内容

在当前 `vllm_moe` conda 环境的 vLLM 0.19.1 中增加了一个轻量 policy 层：

- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/policy.py`
  - 新增 `RequestFeatures`
  - 新增 `FCFSPolicy`
  - 新增 `PriorityPolicy`
  - 新增 `LengthOnlyPolicy`
- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/request_queue.py`
  - 保留原 FCFS 队列
  - 将 priority/length-only 统一为 policy heap queue
  - 新增 `SchedulingPolicy.LENGTH_ONLY`
- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py`
  - 在 scheduler 初始化时加载 policy plugin
  - waiting queue 与 skipped waiting queue 同时存在时，按 policy key 比较队首
- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/config/scheduler.py`
  - `--scheduling-policy` 增加 `length_only` 选项

## 策略定义

当前 `LengthOnlyPolicy` 的服务端排序键为：

```text
(
  prompt_tokens + max_tokens,
  prompt_tokens,
  arrival_time,
  request_id
)
```

排序键越小越早进入调度。

## 实验脚本配套

- `run_stage1_single_gpu.sh` / `run_stage1_dual_gpu.sh`
  - 新增 `SCHEDULING_POLICY`
  - 新增 `CLIENT_DISPATCH_STRATEGY`
  - 启动 vLLM 时传入 `--scheduling-policy`
- `log_vllm_requests.py`
  - 解耦结果标签 `--strategy` 和客户端发送顺序 `--dispatch-strategy`
  - 服务端实验应使用 `--strategy length_only --dispatch-strategy fcfs`
- `run_tp2_length_only_server_formal.sh`
  - 新增服务端内嵌 Length-only 正式实验入口
- `run_tp2_length_only_formal.sh`
  - 保留为客户端代理预实验入口

## 验证

- Python 编译通过：
  - `policy.py`
  - `request_queue.py`
  - `scheduler.py`
  - `config/scheduler.py`
  - `scripts/log_vllm_requests.py`
- Shell 语法检查通过：
  - `run_stage1_single_gpu.sh`
  - `run_stage1_dual_gpu.sh`
  - `run_tp2_load_levels.sh`
  - `run_tp2_fcfs_formal.sh`
  - `run_tp2_length_only_formal.sh`
  - `run_tp2_length_only_server_formal.sh`
- `vllm serve --help=SchedulerConfig` 已显示：

```text
--scheduling-policy {fcfs,length_only,priority}
```

## 后续注意

最终论文实验应使用 `run_tp2_length_only_server_formal.sh`，而不是客户端代理脚本。

## 2026-04-24 小规模验证

### 验证目的

确认服务端内嵌 `Length-only` 能真实进入 vLLM 调度路径，并和同一 patched vLLM 环境下的
`FCFS` 做一个最小 sanity 对比。

### 运行设置

- 模型：`/root/workspace/qwen1.5-MoE-A2.7B`
- 设备：`GPU 0,1`
- `tensor_parallel_size=2`
- `gpu_memory_utilization=0.70`
- `cpu_offload_gb=4`
- `max_model_len=1024`
- 请求集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 重复次数：每组 `1` 次

### 运行组合

1. `medium`
   - `FCFS`: `server_scheduling_policy=fcfs`, `dispatch_strategy=fcfs`
   - `Length-only`: `server_scheduling_policy=length_only`, `dispatch_strategy=fcfs`
2. `high`
   - `FCFS`: `server_scheduling_policy=fcfs`, `dispatch_strategy=fcfs`
   - `Length-only`: `server_scheduling_policy=length_only`, `dispatch_strategy=fcfs`

服务端日志确认 `Length-only` 运行时 vLLM 启动参数包含：

```text
'scheduling_policy': 'length_only'
```

### 结果文件

- 可复现 patch 留痕包：
  - `/root/workspace/patches/vllm_policy_scheduler_20260424/`
- 聚合结果：
  - `/root/workspace/results/tp2_server_policy_smoke_20260424_aggregate.json`
- `medium / length_only`：
  - `/root/workspace/results/tp2_length_only_server_medium_r1_20260424T081127Z_length_only_medium_summary.json`
  - `/root/workspace/logs/raw/tp2_length_only_server_medium_r1_20260424T081127Z_length_only_medium.jsonl`
- `medium / fcfs`：
  - `/root/workspace/results/tp2_fcfs_servercheck_medium_r1_20260424T081521Z_fcfs_medium_summary.json`
  - `/root/workspace/logs/raw/tp2_fcfs_servercheck_medium_r1_20260424T081521Z_fcfs_medium.jsonl`
- `high / length_only`：
  - `/root/workspace/results/tp2_length_only_server_high_r1_20260424T081851Z_length_only_high_summary.json`
  - `/root/workspace/logs/raw/tp2_length_only_server_high_r1_20260424T081851Z_length_only_high.jsonl`
- `high / fcfs`：
  - `/root/workspace/results/tp2_fcfs_servercheck_high_r1_20260424T082136Z_fcfs_high_summary.json`
  - `/root/workspace/logs/raw/tp2_fcfs_servercheck_high_r1_20260424T082136Z_fcfs_high.jsonl`

### 初步观察

`medium` 档下两组几乎完全重合，说明当前 patched vLLM 没有破坏 FCFS，但
`arrival_interval=1.5`、`max_concurrency=2` 的排队压力不足以明显放大调度差异。

`high` 档下出现了可观测差异：

| 指标 | FCFS | Length-only | 相对变化 |
| --- | ---: | ---: | ---: |
| throughput tokens/s | 15.5266 | 15.7067 | +1.16% |
| avg TTFT | 1.4695 | 1.3751 | -6.42% |
| P95 TTFT | 2.1298 | 1.7681 | -16.98% |
| P99 TTFT | 2.1521 | 1.7686 | -17.82% |
| avg TPOT | 0.3037 | 0.3007 | -1.01% |
| P95 TPOT | 0.3598 | 0.3571 | -0.75% |
| P99 TPOT | 0.3623 | 0.3588 | -0.97% |

### 结论边界

这只是 `24` 条请求、每组 `1` 次的 smoke 验证，不能作为论文正式结论。
它目前只说明：

1. 服务端 `--scheduling-policy length_only` 能正常启动并完成请求。
2. 客户端保持 `dispatch_strategy=fcfs`，结果不再依赖客户端代理重排。
3. 在 high 档小样本下，服务端 Length-only 已经出现改善尾部 TTFT 的迹象。

下一步应跑 `high` 档多次重复，确认该改善是否稳定；确认后再跑
`low / medium / high` 三档正式实验。

## 2026-04-24 负载档位重新校准

### 背景

小规模 smoke 中，旧 `medium` 档：

```text
arrival_interval=1.5
max_concurrency=2
```

在 FCFS 与服务端 Length-only 之间几乎完全重合。这说明该档位实际排队压力偏低，
更像 low/medium 之间的轻负载，难以放大服务端调度策略差异。

### 校准方法

本轮只使用服务端 `FCFS` 做负载校准，避免策略差异干扰负载定义。

- 请求集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 策略：`server_scheduling_policy=fcfs`
- 客户端：`dispatch_strategy=fcfs`
- 对照端点：
  - 旧 `medium`: `arrival_interval=1.5`, `max_concurrency=2`
  - 旧 `high`: `arrival_interval=0.0`, `max_concurrency=6`
- 新增候选：
  - `arrival_interval=1.0`, `max_concurrency=3`
  - `arrival_interval=0.75`, `max_concurrency=3`
  - `arrival_interval=0.5`, `max_concurrency=4`
  - `arrival_interval=0.25`, `max_concurrency=4`

聚合结果：

- `/root/workspace/results/tp2_fcfs_load_calibration_20260424.json`

### 校准结果

| label | interval | concurrency | throughput | avg TTFT | P95 TTFT | P99 TTFT | avg TPOT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_medium_i150_c2 | 1.5 | 2 | 11.636 | 0.875 | 0.975 | 1.011 | 0.141 | 0.165 |
| calib_i100_c3 | 1.0 | 3 | 12.109 | 0.968 | 1.148 | 1.332 | 0.204 | 0.238 |
| calib_i075_c3 | 0.75 | 3 | 12.104 | 1.006 | 1.359 | 1.382 | 0.204 | 0.237 |
| calib_i050_c4 | 0.5 | 4 | 15.140 | 1.128 | 1.455 | 1.562 | 0.214 | 0.254 |
| calib_i025_c4 | 0.25 | 4 | 15.181 | 1.189 | 1.577 | 1.898 | 0.212 | 0.252 |
| old_high_i000_c6 | 0.0 | 6 | 15.527 | 1.469 | 2.130 | 2.152 | 0.304 | 0.362 |

### 推荐新档位

建议后续正式实验采用：

```text
low:
  arrival_interval=8.0
  max_concurrency=1

medium:
  arrival_interval=0.5
  max_concurrency=4

high:
  arrival_interval=0.0
  max_concurrency=6
```

理由：

1. 旧 `medium=1.5/c2` 太轻，P95/P99 TTFT 与轻负载差异不足。
2. `0.5/c4` 已经有明显排队压力，P99 TTFT 从旧 medium 的 `1.011s` 提升到
   `1.562s`，但仍低于 high 的 `2.152s`。
3. `0.25/c4` 更接近 medium-high，P99 TTFT 已到 `1.898s`，与 high 的间隔较小。
4. 新 medium 采用 `0.5/c4` 能给调度策略留下选择空间，同时不至于完全贴近瓶颈。

后续如时间允许，可以把 `0.25/c4` 作为额外 stress-medium 档，但不建议替代主
`medium`。

## 2026-04-24 新 medium 下服务端 Length-only 复测

### 目的

负载校准后，使用新的 `medium=0.5/c4` 再跑一次服务端 Length-only，确认该档位
是否能比旧 `medium=1.5/c2` 更容易暴露策略差异。

### 配置

- 请求集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 服务端策略：`--scheduling-policy length_only`
- 客户端发请求顺序：`dispatch_strategy=fcfs`
- 负载档位：`arrival_interval=0.5`, `max_concurrency=4`

本轮输出：

- 摘要：`/root/workspace/results/tp2_length_only_server_medium_r1_20260424T092009Z_length_only_medium_summary.json`
- 原始请求日志：`/root/workspace/logs/raw/tp2_length_only_server_medium_r1_20260424T092009Z_length_only_medium.jsonl`
- GPU 摘要：`/root/workspace/logs/processed/tp2_length_only_server_medium_r1_20260424T092009Z_gpu_summary.json`
- 对比汇总：`/root/workspace/results/tp2_server_policy_new_medium_smoke_20260424_aggregate.json`

### 与 FCFS 校准点对比

FCFS 使用负载校准中的同档结果：

`/root/workspace/results/tp2_fcfs_loadcalib_calib_i050_c4_20260424T084745Z_fcfs_calib_i050_c4_summary.json`

| strategy | throughput | avg TTFT | P95 TTFT | P99 TTFT | avg TPOT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCFS | 15.140 | 1.128 | 1.455 | 1.562 | 0.214 | 0.254 |
| Length-only | 15.144 | 1.128 | 1.455 | 1.562 | 0.214 | 0.254 |
| Length-only vs FCFS | +0.026% | -0.028% | -0.040% | -0.045% | -0.027% | -0.029% |

### 判断

新 `medium=0.5/c4` 相比旧 medium 确实提高了排队压力，适合作为正式三档中的
中负载点；但在 `24` 条请求的小样本 smoke 中，服务端 Length-only 与 FCFS 仍然
几乎完全重合。

这个结果不应解释为 Length-only 无效。更合理的解释是：当前 medium 样本量偏小、
请求到达模式较确定，实际等待队列里可供调度策略交换顺序的窗口仍然有限。此前
`high=0.0/c6` 已经出现过 Length-only 改善尾部 TTFT 的迹象，因此下一步应优先：

1. 在 `high=0.0/c6` 下跑服务端 FCFS 与 Length-only 多次重复，确认尾部改善是否稳定。
2. 在时间允许时，把 `medium=0.5/c4` 的请求数从 `24` 提高到 `48` 或 `72`，观察更长队列下策略差异是否出现。

## 2026-04-24 high 档服务端策略三轮重复

### 目的

旧 smoke 中 `high=0.0/c6` 已经出现过服务端 Length-only 改善尾部 TTFT 的迹象。
本轮固定请求数为 `24`，对 FCFS 与服务端 Length-only 各跑三次，验证该现象是否稳定。

### 配置

- 请求集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `24` 条
- 负载档位：`arrival_interval=0.0`, `max_concurrency=6`
- 客户端发请求顺序：两组均为 `dispatch_strategy=fcfs`
- 对比变量：
  - FCFS：`--scheduling-policy fcfs`
  - Length-only：`--scheduling-policy length_only`

聚合结果：

- `/root/workspace/results/tp2_server_policy_high_r3_20260424_aggregate.json`

单轮摘要：

- FCFS:
  - `/root/workspace/results/tp2_fcfs_high_r1_20260424T092542Z_fcfs_high_summary.json`
  - `/root/workspace/results/tp2_fcfs_high_r2_20260424T092824Z_fcfs_high_summary.json`
  - `/root/workspace/results/tp2_fcfs_high_r3_20260424T093105Z_fcfs_high_summary.json`
- Length-only:
  - `/root/workspace/results/tp2_length_only_server_high_r1_20260424T093356Z_length_only_high_summary.json`
  - `/root/workspace/results/tp2_length_only_server_high_r2_20260424T093644Z_length_only_high_summary.json`
  - `/root/workspace/results/tp2_length_only_server_high_r3_20260424T093925Z_length_only_high_summary.json`

### 三轮均值

| strategy | throughput | avg TTFT | P95 TTFT | P99 TTFT | avg TPOT | P99 TPOT | output tokens | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCFS | 15.641 | 1.455 | 2.029 | 2.173 | 0.302 | 0.356 | 1275.3 | 81.539 |
| Length-only | 15.971 | 1.454 | 1.749 | 1.942 | 0.306 | 0.360 | 1332.0 | 83.387 |
| Length-only vs FCFS | +2.111% | -0.107% | -13.785% | -10.623% | +1.525% | +1.168% | +4.443% | +2.265% |

### 判断

服务端 Length-only 在 high 档三轮中稳定降低尾部 TTFT：

- P95 TTFT 从 `2.029s` 降到 `1.749s`，约 `-13.8%`。
- P99 TTFT 从 `2.173s` 降到 `1.942s`，约 `-10.6%`。
- avg TTFT 基本不变，说明收益主要集中在尾部请求。

需要注意的是，Length-only 三轮的平均输出 token 数也更高：

- FCFS：`1275.3`
- Length-only：`1332.0`

因此 throughput、wall time、TPOT 的对比不能直接解释为纯调度收益；它们同时受到
生成长度差异影响。更适合优先引用的是 TTFT 尾部指标，因为这轮实验的核心问题是
服务端等待队列调度是否能降低排队等待。

当前结论：`high=0.0/c6` 下，服务端 Length-only 的尾部 TTFT 改善比单轮 smoke 更
稳定，值得作为后续 LTR 策略的 server-side 对照基线。下一步建议扩大 medium 的
请求数到 `48` 或 `72`，判断中负载下是否只是样本太短导致策略窗口不足。

## 2026-04-24 medium 档 72 请求扩样复测

### 目的

24 请求下，新 `medium=0.5/c4` 中 FCFS 与服务端 Length-only 仍然几乎完全重合。
本轮把请求数扩大到 `72`，检验 medium 下策略不显著是否只是由于样本太短、等待队列
窗口不足。

### 配置

- 请求集：`data/prompts/eval_requests.jsonl`
- 请求数：前 `72` 条
- 负载档位：`arrival_interval=0.5`, `max_concurrency=4`
- 客户端发请求顺序：两组均为 `dispatch_strategy=fcfs`
- 对比变量：
  - FCFS：`--scheduling-policy fcfs`
  - Length-only：`--scheduling-policy length_only`

输出文件：

- 聚合结果：`/root/workspace/results/tp2_server_policy_medium72_20260424_aggregate.json`
- FCFS 摘要：`/root/workspace/results/tp2_fcfs_medium_r1_20260424T104819Z_fcfs_medium_summary.json`
- Length-only 摘要：`/root/workspace/results/tp2_length_only_server_medium_r1_20260424T105351Z_length_only_medium_summary.json`
- FCFS 原始日志：`/root/workspace/logs/raw/tp2_fcfs_medium_r1_20260424T104819Z_fcfs_medium.jsonl`
- Length-only 原始日志：`/root/workspace/logs/raw/tp2_length_only_server_medium_r1_20260424T105351Z_length_only_medium.jsonl`

raw log 检查：

- FCFS：`dispatch_strategy=fcfs`, `server_scheduling_policy=fcfs`, rows=`72`
- Length-only：`dispatch_strategy=fcfs`, `server_scheduling_policy=length_only`, rows=`72`

### 结果

| strategy | throughput | avg TTFT | P95 TTFT | P99 TTFT | avg TPOT | P99 TPOT | output tokens | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCFS | 15.527 | 1.144 | 1.565 | 1.750 | 0.223 | 0.253 | 3804 | 244.993 |
| Length-only | 15.528 | 1.143 | 1.568 | 1.748 | 0.223 | 0.253 | 3804 | 244.976 |
| Length-only vs FCFS | +0.007% | -0.045% | +0.229% | -0.081% | -0.008% | -0.019% | 0.000% | -0.007% |

### 判断

扩大到 `72` 条请求后，medium 档仍然没有出现可解释的策略差异。由于两组
`total_output_tokens` 完全相同，这里的贴合比 24 请求 smoke 更有说服力：

- `medium=0.5/c4` 能形成比旧 medium 更明显的尾部压力，P99 TTFT 约 `1.75s`。
- 但该压力仍不足以让 Length-only 在服务端等待队列中产生明显收益。
- raw log 的前序完成请求基本保持原顺序，说明该负载下服务端可重排窗口有限。

因此，正式实验不应把 medium 作为 Length-only/LTR 的主要收益来源。medium 可以保留
为“中负载下策略基本不伤害性能”的控制点；主要结论应放在 high 档，那里已经观察到
稳定的尾部 TTFT 改善。

下一步建议进入 LTR 策略实现，而不是继续在 medium 上细抠。现有结论已经足够说明：

1. medium：策略差异不明显，可作为 neutral/control load。
2. high：Length-only 已有稳定尾部收益，是 LTR 应该超越的 server-side baseline。

## 2026-04-24 LTR 服务端调度入口实现

### 抄袭风险判断

复现 LTR 论文中的调度思想本身不构成抄袭风险，前提是：

1. 论文中明确引用原工作，说明 LTR/ranking scheduler 思想来自该论文。
2. 代码独立实现，不复制对方仓库或论文附录中的源码。
3. 实验方法写清楚哪些部分是复现 baseline，哪些部分是本文自己的 joint 排序策略。
4. 如果最终 joint 中包含 LTR 分数，应表述为“在 LTR 长度排序信号基础上结合本文的
   MoE/专家局部性信号”，而不是把 LTR 本身包装成原创贡献。

更稳妥的定位是：把 `LTR` 写成一个被比较或被组合的 baseline/component；本文贡献是
joint 排序目标、MoE 特征、风险控制或工程集成，而不是 LTR 这个通用思想。

### 实现范围

本轮实现的是 LTR 调度入口，而不是训练好的 predictor：

- vLLM 增加 `--scheduling-policy ltr`
- `LTRPolicy` 从 `VLLM_LTR_SCORE_FILE` 读取外部分数
- 排序键为：

```text
(priority, ltr_score, estimated_total_tokens, prompt_tokens, arrival_time, request_id)
```

其中 `ltr_score` 越小越先调度。缺失 score 时退化为
`estimated_total_tokens = prompt_tokens + max_tokens`，避免线上实验因为少量缺失分数失败。

这对应论文 §4.3 的核心调度思想：predictor 给请求打排序分数，服务端按
`(priority, score)` 形成 running batch。当前暂未实现论文中的 starvation count /
priority quantum 动态提升机制，因为这需要在 scheduler iteration 中维护每个请求的
跨轮状态，侵入性高于单纯 waiting queue policy。后续如果 high 档 LTR 出现长请求饥饿，
再补这一层更合适。

### 修改文件

- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/policy.py`
  - 新增 `LTRPolicy`
  - 支持 JSON/JSONL score 文件
  - 支持 `cmpl-` / `chatcmpl-` 前缀兼容
- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/request_queue.py`
  - 新增 `SchedulingPolicy.LTR`
  - 新增 `LTRRequestQueue`
- `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/config/scheduler.py`
  - CLI 允许 `--scheduling-policy ltr`
- `/root/workspace/scripts/log_vllm_requests.py`
  - `strategy` 增加 `ltr`
  - completion 请求透传 `request_id` 和 `X-Request-Id`
- `/root/workspace/scripts/build_ltr_score_file.py`
  - 从请求 manifest 生成 LTR score 文件，用于链路验证或接 predictor 输出
- `/root/workspace/scripts/run_tp2_ltr_server_formal.sh`
  - TP=2 服务端 LTR 正式实验入口
- `/root/workspace/patches/vllm_policy_scheduler_20260424/`
  - 同步更新可复用 patch overlay

### 验证

已完成：

- `python -m py_compile`：
  - vLLM `policy.py`
  - vLLM `request_queue.py`
  - vLLM `scheduler.py`
  - vLLM `config/scheduler.py`
  - `scripts/log_vllm_requests.py`
  - `scripts/build_ltr_score_file.py`
- `bash -n`：
  - `scripts/run_tp2_ltr_server_formal.sh`
  - `scripts/run_stage1_dual_gpu.sh`
  - `scripts/run_tp2_load_levels.sh`
- CLI 检查：

```text
--scheduling-policy {fcfs,length_only,ltr,priority}
```

- 队列级测试：

```text
['cmpl-eval-short', 'cmpl-eval-mid', 'cmpl-eval-missing', 'cmpl-eval-long']
```

说明 `ltr` 能按外部 score 文件排序，并且能识别 OpenAI completions 内部的
`cmpl-` request id 前缀。

### 下一步

1. 接入真正 predictor 输出，生成完整 `VLLM_LTR_SCORE_FILE`。
2. 如果短期内还没有训练 predictor，可以先用 `target_max_new_tokens` 或历史观测长度
   生成 score 文件做 scheduler plumbing smoke，但记录中必须标注“不是训练 LTR”。
3. 在 high 档跑 `FCFS / Length-only / LTR` 对比，确认 LTR 是否能超过当前 Length-only
   server-side baseline。

## 2026-04-24 LTR-lite predictor 第一版

### 目的

先实现一个低成本、可解释、不会干扰服务端性能的 predictor，用来给服务端 `ltr`
policy 提供 `request_id -> score`。这个版本是离线 predictor，不在 vLLM 请求路径上
运行，因此不会增加在线调度开销。

### 实现

新增脚本：

- `/root/workspace/scripts/train_ltr_lite_predictor.py`

模型选择：

- 依赖：只使用 `numpy`，不依赖 `sklearn/scipy`
- 模型：ridge regression
- 标签：已有 FCFS raw logs 中的 `output_tokens`
- 标签聚合：同一 request 多次运行取平均输出长度
- score 约定：预测越短，score 越小，越先调度

特征：

- 数值特征：
  - `target_max_new_tokens`
  - `prompt_tokens`
  - `prompt_chars`
  - `prompt_words`
  - `prompt_lines`
  - `prompt_digits`
  - `prompt_punctuation`
  - `prompt_ascii_ratio`
  - `prompt_cjk_chars`
- 类别/稀疏特征：
  - `group`
  - `lang`
  - `domain`
  - `task_family`
  - `topic`
  - `source`
  - `tags`

训练命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe
python scripts/train_ltr_lite_predictor.py \
  --request-file data/prompts/eval_requests.jsonl \
  --raw-log-glob 'logs/raw/tp2_fcfs_*_fcfs_*.jsonl' \
  --output-score-file logs/processed/ltr_lite_scores_eval_20260424.jsonl \
  --model-output results/ltr_lite_model_20260424.json \
  --report-output results/ltr_lite_report_20260424.json \
  --tokenizer-path /root/workspace/qwen1.5-MoE-A2.7B \
  --label-strategy fcfs \
  --alpha 5.0 \
  --validation-fraction 0.2
```

输出：

- score 文件：`/root/workspace/logs/processed/ltr_lite_scores_eval_20260424.jsonl`
- 模型文件：`/root/workspace/results/ltr_lite_model_20260424.json`
- 报告文件：`/root/workspace/results/ltr_lite_report_20260424.json`

### 数据规模

- manifest 请求数：`240`
- 有标签请求数：`240`
- 训练请求数：`192`
- 验证请求数：`48`
- 每个请求标签样本数：
  - min: `9`
  - mean: `10.7`
  - max: `24`

### 验证指标

验证集：

| score | Kendall tau-b | Spearman |
| --- | ---: | ---: |
| LTR-lite ridge | 0.629 | 0.728 |
| target_max_new_tokens | 0.689 | 0.752 |
| estimated_total_tokens | 0.525 | 0.647 |

解释：

- LTR-lite 明显优于当前服务端 `Length-only` 使用的
  `estimated_total_tokens = prompt_tokens + max_tokens` 排序基线。
- 但它没有超过更强的单特征 `target_max_new_tokens` baseline，说明当前特征模型仍然会
  学到一些噪声。
- 因此，当前 LTR-lite 可以用于 server-side `ltr` plumbing 和 preliminary 对比，但不应
  作为最终论文里的强 LTR predictor 结论。

可解释性 top coefficients：

- `target_max_new_tokens`: `+25.20`
- `group=long`: `+23.60`
- `group=short`: `-17.81`
- `prompt_chars`: `+8.33`
- `tag=planning`: `+5.01`
- `prompt_cjk_chars`: `+4.95`
- `prompt_punctuation`: `+4.47`

这些权重符合直觉：更大的 decode budget、long group、更长 prompt 往往对应更大的
输出长度 score。

### 链路验证

已验证：

- `python -m py_compile scripts/train_ltr_lite_predictor.py`
- score JSONL 共 `240` 行，均包含 `request_id` 和 `score`
- `VLLM_LTR_SCORE_FILE=/root/workspace/logs/processed/ltr_lite_scores_eval_20260424.jsonl`
  时，vLLM `LTRPolicy` 能读取 score 并按分数排序

队列级前序排序示例：

```text
['eval-0008', 'eval-0007', 'eval-0012', 'eval-0024',
 'eval-0015', 'eval-0018', 'eval-0014', 'eval-0013',
 'eval-0004', 'eval-0016', 'eval-0006', 'eval-0023']
```

### 下一步判断

短期可以跑一轮 high 档 `LTR-lite` server-side smoke，目标不是证明最终 LTR 有效，而是
确认：

1. `request_id -> score -> vLLM ltr queue` 端到端链路正确。
2. LTR-lite 相比当前 Length-only 是否在 high 档有额外收益。

如果 LTR-lite 不超过 Length-only，不应继续在这个线性模型上硬调；更合理的下一步是：

- 用 `target_max_new_tokens` 作为更强的非学习长度排序 baseline；
- 或进入 neural / pairwise / ListMLE predictor。

