# 批判指导实验部分：压缩上下文

来源会话：`.codex/sessions/2026/05/03/rollout-2026-05-03T12-07-39-019dedbc-b098-7dd3-8454-8b9acfb462be.jsonl`  
会话名称：批判指导实验部分  
压缩日期：2026-05-04

## 读取结论

本会话已围绕 WREC 实验“是否过于 toy”完成批判性审阅、方法模块化改造和 policy overhead 补测。当前应把 WREC 定位为“硬件校准的 trace-driven expert cache replay 方法”，不能声称已经完成生产级 MoE serving 系统。

## 核心判断

- 当前 WREC 不是没有价值的 toy demo，但证据链仍主要停留在离线 route-trace replay。
- 最强主结果是 prefill-stage fixed total-budget replay：WREC-H2 / recent0 WREC-H2 显著优于 LRU 和 static-hot，并具有 zero prefetch waste。
- 论文必须明确主指标是 transfer stall proxy，不是端到端 TTFT、TPOT 或真实 serving latency。
- decode-only replay 是负结果：prefill-trained WREC-H2 在 decode 阶段弱于 LRU，说明方法具有 phase-specific 假设。
- 后续补强重点不应继续堆同类表格，而应增强方法模块、硬件校准和 runtime shadow 落点。

## 答辩视角的主要质疑

- WREC 是可复用方法，还是一组实验脚本堆出的结果？
- 核心方法代码是否有清晰的 `policy / online state / trace prior` 边界？
- 换 workload、模型或 serving trace 时，WREC 是否仍能复用？
- simulator 中的 stall proxy 是否有真实硬件依据？
- WREC 在线决策开销是否会抵消减少 expert miss 带来的收益？
- 真实 vLLM / MoE runtime 中是否已有可生效的 expert weight cache 控制链路？

## 已完成的补强

### WREC 方法模块化

已从 `moe_affinity` 实验脚本中抽出独立包：

```text
workspace/scripts/wrec/
  __init__.py
  policy.py
  online_state.py
  trace_prior.py
```

模块含义：

- `policy.py`：WREC score、admission、eviction、constrained target set、prefetch replan。
- `online_state.py`：维护 request-local、recent history、token-layer expert history。
- `trace_prior.py`：由 train router trace 构建 train prior、window score、cross-layer transition。

已更新主要调用脚本：

- `workspace/scripts/moe_affinity/simulate_expert_cache_total_budget.py`
- `workspace/scripts/moe_affinity/run_wrec_phase5_ablation.py`

验证结果：

- Python 编译检查通过。
- Mixtral Dolly 25% total-budget smoke 通过。
- `wrec_h2` / `wrec_c` miss rate 保持 `0.16333409910228316`，说明模块化未改变核心行为。

### 命名调整

- `state.py` 改为 `online_state.py`，避免泛化命名。
- `stats.py` 改为 `trace_prior.py`，明确其职责是从训练 trace 构建 WREC 决策先验。

### Policy overhead 测量

新增脚本：

- `workspace/scripts/wrec/benchmark_policy_overhead.py`

输出结果：

- `workspace/results/wrec/calibration/wrec_policy_overhead_mixtral8x7b_dolly_n256_slots64_20260503.json`
- `workspace/results/wrec/calibration/wrec_policy_overhead_mixtral8x7b_dolly_n256_slots64_20260503.md`

配置：

- Eval trace: Mixtral Dolly eval n256。
- Train trace: Mixtral Dolly train n512。
- Policy: WREC-H2 no-recent。
- Total slots: `64`。
- Weights: recent `0`，request `1024`，cross-layer `1024`。
- Bandwidth: `41.37 GB/s`。

结果：

- Expert refs: `905408`。
- Router events: `452704`。
- Demand misses: `147884`。
- Online loop total: `13.857839 s`。
- Mean overhead: `15.306 us/expert ref`，`30.611 us/router event`。
- History update: `0.565 us/expert ref`。
- Admission decision: `88.290 us/miss`。
- Mixtral expert bytes: `352321536`。
- Estimated expert transfer time: `8.516353 ms`。
- Policy loop / expert transfer ratio: `0.001797`。

结论：当前 Python 实现的 WREC 在线决策开销比单次 expert transfer 低两个数量级以上，可支撑“policy overhead 不会主导当前 transfer-bound replay proxy”的论断。但这仍不是端到端 serving latency 证明。

## 关键信号解释

`request-local` 信号表示当前 request 内某一层到目前为止偏向哪些 expert。它维护当前 request 内每层 expert 计数，并以 `request_weight * request_prob` 加入 WREC score。该信号捕获 request 内部的局部路由偏好，不等同于全局 static-hot。

`cross-layer` 信号表示同一个 token 在上一层选择的 expert 对当前层 expert 的条件预测。训练阶段统计 `P(current_layer_expert | previous_layer_expert)`，在线阶段用当前 token 上一层已知 expert 查询转移表，并以 `cross_layer_weight * cross_prob` 加入 score。该信号捕获 token 级层间路由相关性。

二者互补：

- `request-local` 抓 request 级别偏好画像。
- `cross-layer` 抓 token 级别层间路径预测。
- 当前消融显示 request-local 与 cross-layer 是 WREC-H2 的主要有效项。
- recent 信号在 total-budget prefill replay 中不是稳定正贡献。

## Simulator 不是纯 toy 的论证逻辑

WREC replay simulator 的核心链路是：

```text
expert cache miss
-> CPU/host memory 到 GPU 的 expert weight transfer
-> transfer stall
-> WREC 通过减少 miss 降低 stall proxy
```

因此需要证明：

- 输入来自真实 router trace，而不是随机合成访问序列。
- expert size 来自 Mixtral expert 权重规模。
- bandwidth 使用当前机器实测 `41.37 GB/s`。
- stall proxy 由 `miss bytes / measured bandwidth` 得出。
- bandwidth sensitivity 中带宽降低时 absolute saved stall 增大，符合 transfer-bound 系统模型。
- policy overhead 明显小于 expert transfer time，不会抵消 miss reduction 收益。

更稳妥的论文表述是：

```text
本文基于硬件校准的 trace-driven replay simulator 评估 WREC 对 prefill 阶段 expert cache miss 与 transfer stall proxy 的影响。
```

不应表述为：

```text
WREC 已在真实 vLLM serving runtime 中降低端到端推理延迟。
```

## 后续优先级

最高优先级：

1. 继续稳定 `workspace/scripts/wrec/` 包边界，必要时补 `cache_manager.py` 与 `metrics.py`。
2. 补 expert-size copy microbenchmark，进一步确认 `41.37 GB/s` 与单 expert transfer time 的对应关系。
3. 做最小 runtime shadow mode：真实 runtime 产生 routed expert events，WREC 在线消费 events，输出 would-admit / would-evict / cache-state metrics，并与离线 replay 对齐。

暂不优先：

- 继续深挖 WREC-C prefetch。
- 将 decode 写成主文正结果。
- 继续堆更多同类 simulator 表格。
- 声称生产级 vLLM integration。

## 可直接沿用的论文边界

本文提出一种面向资源受限 MoE 推理的 expert cache scheduling 方法 WREC，并在硬件校准的 route-trace replay simulator 中验证其对 prefill 阶段 transfer stall proxy 的降低效果。实验表明，在不改变模型 routing 的前提下，WREC 能稳定优于 LRU 和 static-hot baseline；同时，decode replay 的负结果表明 prefill 与 decode 需要 phase-specific cache policy。

