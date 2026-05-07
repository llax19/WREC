# WREC 论文收束摘要

## 论文焦点

第一版论文应明确写成一篇 **prefill 阶段 MoE expert cache scheduling** 论文，而不是同时覆盖 prefill 与 decode 的统一 serving 系统论文。当前主问题不是 router 学习，也不是 runtime 工程化，而是：

```text
在 transfer-bound、partial-residency 的 MoE 设定下，
是否可以仅通过在线 expert cache scheduling，
在不改变 routing 的前提下稳定降低 prefill-stage stall proxy？
```

当前推荐的主方法配置为：

```text
request/cross-layer WREC-H2
window_size = 4
history_size = 8
recent_weight = 0
request_weight = 1024
cross_layer_weight = 1024
prefetch_queue = 0
```

这个版本应被写成 **由消融实验驱动的简化版 WREC-H2**，而不是新的方法家族。默认 `WREC-H2` 仍应保留在附录中，作为 refinement 之前的版本。

## 叙事主线

论文主线建议固定为：

```text
HRM bottleneck justification
-> Mixtral prefill route traces
-> locality / hotness stability analysis
-> fair online baselines + Belady oracle
-> fixed total-budget WREC result
-> ablation and sensitivity
-> negative findings and claim boundary
```

这条主线的优点是变量少、证据链闭环，而且负结果也能自然放入同一叙事中。

## 核心证据

### 1. 为什么选 Mixtral，为什么这个 bottleneck 值得研究

建议引用：

- `results/wrec/hrm_eval/mixtral/hrm_mixtral_bottleneck_eval_20260428.md`
- `results/wrec/hrm_eval/new_env/hrm_mixtral_new_env_20260501.md`
- `results/wrec/calibration/wrec_expert_transfer_mixtral8x7b_20260504.md`
- `figures/wrec/phase8_20260503/wrec_hrm_transfer_bound_heatmap_mixtral8x7b_20260503.svg`

主文应提炼出三个判断：

1. Mixtral-8x7B 在 BF16 下的 expert weights 约为 `84.00 GiB`，总权重约为 `86.99 GiB`。
2. 在受限显存、KV reservation 和 measured `41.37 GB/s` CPU-GPU bandwidth 下，Mixtral 会进入 expert-transfer-bound regime。单 expert 大小为 `352321536 bytes`，pinned host memory 到 GPU 的 100 次实测中位 copy time 为 `8.467264 ms`，与 simulator 使用的 `8.516353 ms/expert` 基本一致。
3. 因而本文关注的问题不是 “small MoE all-resident inference”，而是 “partial residency 下的 online expert cache scheduling”。

注意边界：该校准结果只证明 transfer-stall proxy 的硬件量级合理，不等价于端到端 serving latency 结果。

### 2. 为什么 prefill trace 值得优化

建议引用：

- `results/wrec/workload_stats/mixtral8x7b_dolly_train_n512_mem48_workload_stats_20260502.md`
- `figures/wrec/phase8_20260503/wrec_locality_summary_mixtral8x7b_dolly_train_20260503.svg`

主文应强调以下几个数字：

- train trace: `29,123` input tokens
- router events: `931,936`
- expert refs: `1,863,872`
- same-layer reuse distance `p50/p90/p99 = 5 / 27 / 7097`
- window-4 unique experts `p50 = 5`
- train/eval top-2 和 top-4 overlap mean 均为 `0.9375`

这允许你稳妥地写出：prefill trace 中既存在明显的短窗口 locality，也存在跨 split 的热度稳定性。

### 3. 为什么这个问题并不 trivial

建议引用：

- `results/wrec/cache_sim/mixtral8x7b_dolly_eval_n256_mem48_fair_baselines_bw41p37_20260502.md`

这一部分主文应完成两件事：

1. 说明公平对比只使用在线或可由离线统计得到的 baseline：
   `on_demand`, `lru`, `static_hot`
2. 说明 `Belady oracle` 只是理论上界，而不是 deployable policy

这里最重要的信息不是哪个 baseline 最弱，而是：**Belady 与 online baselines 之间仍存在明显 gap**。这说明 expert cache scheduling 既值得优化，也不是一个简单的 popularity caching 问题。

### 4. 主结果应如何领起全文

主文主表应使用：

- `results/wrec/wrec_total_budget/wrec_total_budget_recent0_mixtral8x7b_dolly_n256_20260503.md`

建议突出这组结果：

| total slots | gain vs LRU | waste |
|---:|---:|---:|
| 32 | 71.80% | 0 |
| 64 | 76.43% | 0 |
| 96 | 80.37% | 0 |
| 128 | 81.40% | 0 |
| 192 | 75.27% | 0 |

这组结果支持四个稳定表述：

- WREC 在全部测试预算上优于 LRU
- 收益不是单点现象
- 收益不依赖 prefetch waste
- 与 Belady 相比仍有清晰的剩余 gap

### 5. 消融实验真正说明了什么

建议引用：

- `results/wrec/phase5/wrec_phase5_findings_20260503.md`
- `results/wrec/phase5/wrec_ablation_mixtral8x7b_dolly_20260503.md`
- `results/wrec/phase5/wrec_sensitivity_mixtral8x7b_dolly_20260503.md`

这里的重点不是“所有项都正向贡献”，而是把真正有效的机制讲清楚：

1. request-local 和 cross-layer 是主要有效信号
2. recent term 在当前 total-budget prefill replay 中不是稳定正贡献
3. `train-window-only` 接近 LRU，说明仅靠 train prior 不足以支撑主收益
4. bandwidth 降低时，absolute saved stall 明显增大，符合 transfer-bound story
5. window/history sensitivity 总体平坦，说明结果不是单个超参偶然对齐

这恰好支持将主方法收缩为 request/cross-layer WREC-H2。

### 6. runtime shadow prototype 说明了什么

建议引用：

- `results/wrec/runtime_shadow/wrec_shadow_mixtral8x7b_dolly_eval_n256_slots64_20260504.md`
- `results/wrec/runtime_shadow/wrec_shadow_alignment_mixtral8x7b_dolly_eval_n256_slots64_20260504.md`

这一部分只能作为系统落地点的补强，而不是主结果。可以写成：

> 为验证 WREC 的策略接口是否具备在线运行形式，本文实现了一个 runtime shadow prototype。该原型按 routed expert event stream 逐条消费事件，在线维护 request-local、cross-layer history 与 shadow resident set，并输出 would-admit、would-bypass 和 would-evict 决策。

关键数字：

- expert refs: `905408`
- router events: `452704`
- shadow miss rate: `0.16333409910228316`
- 与 total-budget replay 的 miss rate 完全一致
- alignment validation: `overall_pass = True`，expert refs、hit/miss counts、hit/miss rate、transfer bytes 和 stall metrics 的 delta 均为 `0`
- shadow overhead: `32.380 us/router event`
- decision overhead: `91.532 us/miss`

应明确边界：

- shadow prototype 不控制真实 expert loading；
- 不测端到端 serving latency；
- 它证明的是 WREC policy/state 可以按 runtime event contract 在线运行，并与离线 replay 对齐。

## 负结果应如何安放

当前必须保留的两个负结果是：

### decode 迁移失败

建议引用：

- `results/wrec/decode_replay/wrec_total_budget_decode_only_mixtral8x7b_dolly_eval_n64_new16_20260502.md`

可以写成：

> prefill 训练得到的 WREC prior 不能稳定迁移到 decode-only replay，因此 decode 更适合作为 phase-specific extension，而不是并入第一版主线方法。

### WREC-C 未带来额外净收益

建议引用：

- `results/wrec/wrec_total_budget/wrec_c_total_budget_mixtral8x7b_dolly_n256_20260503.md`
- `results/wrec/wrec_total_budget/wrec_c_prefetch_q1_ov10_i16_total_budget_mixtral8x7b_dolly_n256_20260503.md`

应明确写：

- no-prefetch WREC-C 与 WREC-H2 等价
- constrained-prefetch WREC-C 更差，并带来明显 waste

这说明 constrained planning 在当前版本下并没有增加净收益，不能包装成正结果。

## scale-out 在论文中的位置

scale-out 不应再作为主文必须项，但现在已经足够作为一个有内容的附录：

- `results/wrec/hrm_scaleout_moe_models_20260503.md`
- `figures/wrec/scaleout_20260503/scaleout_resident_fraction_20260503.svg`
- `figures/wrec/scaleout_20260503/scaleout_expert_transfer_ratio_20260503.svg`

附录可以安全地写：

- DeepSeek-MoE-16B 在 `48 GB` 下可以常驻
- Mixtral-8x22B 与 DBRX 在 `96 GB` 下仍高度 transfer-bound
- Mixtral-8x7B 处在两者之间，因此它既不太小，也不极端大，适合作为主实验模型

## 可写结论

第一版论文可以写的结论：

1. HRM 证明 Mixtral 在受限显存/带宽配置下会进入 expert-transfer-bound regime。
2. Mixtral prefill route trace 存在可利用的短窗口 locality。
3. Belady oracle 证明 expert cache scheduling 存在明显可利用上界。
4. 在不改变 routing 的前提下，WREC 降低了 prefill-stage stall proxy 和 transfer pressure。
5. WREC 捕获了部分 oracle gap，但与理想上界仍有差距。
6. WREC 已有 runtime shadow prototype，证明其在线策略接口可以逐 event 运行并与 replay 对齐。

第一版论文不应写：

1. WREC 改善模型质量。
2. WREC 改善 TTFT 或真实 serving latency。
3. WREC 已经是完整 runtime / vLLM integration。
4. WREC 已经解决 decode-phase scheduling。
5. WREC-C 优于 WREC-H2。
6. route-window oracle prefetch 是可部署 online baseline。

## 图表放置建议

主文建议保留五类核心图/表：

1. HRM transfer-bound heatmap
2. expert locality summary
3. fair baseline + Belady oracle gap
4. recent0 WREC total-budget main result
5. ablation + bandwidth sensitivity

appendix 建议保留：

- default H2 vs recent0 H2
- WREC-C negative result
- decode negative result
- scale-out appendix

## 这份文档的用途

这份 summary 的作用不是存档，而是固定论文口径。后续写 abstract、intro、method、experiments 时，应默认遵守这里的主线、claim boundary 和附录放置策略。
