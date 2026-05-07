# WREC 论文写作素材包

## 使用方式

这份文档面向论文正文写作，内容应可以直接拷入草稿后做轻微润色。内部决策、实验路线和阶段判断不应写在这里。

## 方法收束段落

我们最终采用 WREC-H2 的一个简化版本作为在线策略。原始 WREC-H2 将 train-window hotness、online recent-history、current-request frequency 和 cross-layer transition 四类信号组合为一个自适应 admission score。Phase 5 消融结果表明，在固定 total-budget 的 prefill replay 中，recent-history 项不是稳定的正向贡献，而 request-local 与 cross-layer 信号始终承担了主要收益。因此，我们去掉不稳定的 recent 项，保留一个 request/cross-layer WREC-H2，其配置为 `window_size=4`、`history_size=8`、`recent_weight=0`、`request_weight=1024` 和 `cross_layer_weight=1024`。这一变化应被理解为由消融实验驱动的策略简化，而不是新的方法家族。

## 主结果段落

在固定全局 expert-cache budget 下，简化后的 request/cross-layer WREC-H2 在 Mixtral prefill traces 上稳定优于在线 LRU。对于 `32`、`64`、`96`、`128` 和 `192` 个 total expert-cache slots，WREC-H2 相对 LRU 的 stall proxy 改善分别为 `71.80%`、`76.43%`、`80.37%`、`81.40%` 和 `75.27%`，且全程没有引入 prefetch waste。这一收益明显大于 `LRU` 与 `static_hot` 之间的差距，说明改进并不能简单归因于静态热度统计。同时，Belady 在所有预算上都显著更强，说明该 workload 中仍然存在当前在线启发式尚未完全捕获的可利用 locality。

## 消融段落

Phase 5 消融结果表明，WREC-H2 的主要收益来自 request-local 与 cross-layer 信息，而不是一般性的 recency 机制。去掉 cross-layer 项后，平均 gain vs LRU 从 `77.80%` 降至 `68.54%`；相反，去掉 recent-history 项后，平均 gain 反而略升至 `79.40%`。与此同时，`train-window-only` 变体几乎退化到 LRU，早期的 `WREC-H` recent-only 策略也明显弱于 WREC-H2。这些结果共同支持我们采用简化后的最终策略：保留 request 和 cross-layer 信号，去掉不稳定的 recent 项。

## 负结果段落

我们保留两个负结果，因为它们有助于明确本文贡献的边界。第一，prefill-trained WREC prior 不能稳定迁移到 decode-only replay，因此当前方法应被理解为 prefill-stage expert cache scheduler，而不是统一的 inference-time policy。第二，constrained WREC-C 在 no-prefetch 版本下并不优于 WREC-H2，而在 constrained prefetch 开启后还会因为额外 waste 和 cache pollution 而进一步变差。这两个负结果说明，decode 与 constrained prefetch 更适合作为独立扩展，而不是并入第一版主线方法。

## 局限性段落

本文在 trace-driven simulator 中验证 WREC，而不是在完整 serving runtime 中直接验证。主结果仅覆盖 prefill-stage expert cache scheduling，decode 仅作为一个负向迁移结果出现，而不是一个已经解决的扩展。当前 transfer-aware score 仍建立在 homogeneous expert size 的前提上，因此尚不能隔离现实系统中可能出现的所有 heterogeneous transfer cost。最后，scale-out appendix 基于 config-only 的结构分析，应被理解为动机扩展，而不是替代更大模型上的 route-trace 证据。

## 有效性威胁段落

当前 simulator 使用的是 measured 但仍经过简化的 bandwidth 与 copy-cost 抽象，因此 runtime overlap、fragmentation 以及 policy execution overhead 仍未被完全建模。`route-window prefetch` 使用了 future evaluation information，因此只能被视为 oracle-style stress baseline，而不是公平的 deployable policy。DBRX 的 scale-out 点依赖于一个公开 converted config proxy，而不是官方 Databricks Hugging Face repository；这一点足以支持粗粒度的结构对比，但不足以支撑精确实现层面的结论。这些限制不会推翻本文的 prefill-stage 主结果，但会约束结论的外推范围。

## 结论边界

### 安全表述

1. WREC 通过 trace-driven expert cache/offload simulation 得到验证。
2. 当前主要收益体现在 fixed expert-cache budgets 下的 prefill-stage stall proxy 和 transfer pressure 降低。
3. 该方法不修改 routing，也不改变模型输出。
4. 当前结果主要适用于 transfer-bound、partial-residency 的 MoE 场景。

### 避免表述

1. WREC 改善模型质量。
2. WREC 改善 TTFT 或真实端到端 serving latency。
3. WREC 已经是 production-ready 的 vLLM integration。
4. WREC 已经解决 decode-phase MoE caching。
5. WREC-C 优于 WREC-H2。

## 图注草稿

### HRM transfer-bound 热图

图 X：Mixtral-8x7B 在 BF16 和实测 `41.37 GB/s` CPU-GPU bandwidth 下的 HRM transfer-bound 热图。每个单元格表示在固定 GPU memory 与 KV reservation 预算下，被测试的 resident-fraction 和 active-token 组合中，有多大比例仍然属于 expert-transfer-bound。该图表明，当有效显存预算下降时，Mixtral 会稳定落入 transfer-bound regime，因此本文关注 expert cache scheduling，而不是 all-resident execution。

### Prefill 局部性摘要

图 X：Mixtral Dolly train trace 的 prefill expert locality summary。该 trace 包含 `29,123` 个 input tokens、`931,936` 个 router events 和 `1,863,872` 个 expert references。same-layer reuse distance 的 `p50=5`、`p90=27`，window-4 下仅使用 `5` 个 unique experts（`p50`），train/eval hotness overlap 的 top-2 和 top-4 mean 都达到 `0.938`，说明短窗口 locality 既强且稳定。

### 公平基线与 oracle gap

图/表 X：expert-cache budgets 下的公平 online baselines 与 oracle gap。`static_hot` 仅使用 train-trace 统计，`LRU` 是完全在线的 baseline，而 `Belady` 仅作为 oracle upper bound 使用。online baselines 与 Belady 之间的大 gap 说明 expert cache scheduling 问题并不 trivial，而 `LRU` 与 `static_hot` 在部分预算下差距很小，则说明静态热度本身不足以解释全部收益。

### WREC 主结果

图 X：固定全局 expert-cache budgets 下的 WREC total-budget 主结果。request/cross-layer WREC-H2 在所有测试预算上都相对 LRU 降低 stall proxy：当 total slots 为 `32/64/96/128/192` 时，收益分别为 `71.80%/76.43%/80.37%/81.40%/75.27%`，且 zero prefetch waste。最高预算下收益略有回落，说明当 resident set 接近完整 working set 时，cache scheduling 的决定性作用会减弱。

### Oracle-gap 捕获

图 X：不同 total expert-cache budgets 下的 oracle-gap capture。WREC 在全部预算上都能弥合 LRU 与 Belady 之间的大部分 gap，但仍明显弱于 oracle upper bound。这说明当前策略已经捕获了有意义的在线 locality，同时仍为未来的 constrained planning 或 phase-specific extension 留下空间。

### 消融实验

图 X：WREC 信号的 Phase 5 消融结果。去掉 cross-layer 信息会显著降低性能，而去掉 recent-history 项则会略微改善结果。`train-window-only` 几乎退化到 LRU，这说明离线 hotness 本身并不足够；当前收益主要来自 train statistics 与在线 request-local、cross-layer context 的结合。

### 带宽敏感性

图 X：WREC-H2 的 bandwidth sensitivity。相对 LRU 的比例收益总体稳定，但当 CPU-GPU bandwidth 降低时，absolute saved stall 会快速增大，这与 transfer-bound 的 HRM 叙事一致。因此，这张图应以 absolute saved stall 来解释，而不仅仅是百分比 gain。

### scale-out 附录

图 X：Mixtral-8x7B、Mixtral-8x22B、DBRX 和 DeepSeek-MoE-16B 的 config-only scale-out appendix。在相同 BF16、KV reservation 和 measured bandwidth 假设下，更大的 MoE 模型保留的 feasible resident fraction 更低，并且即使在 96 GB 下也几乎完全 transfer-bound。该分析扩展了 WREC 的问题动机，但并不替代主文中基于 Mixtral-8x7B route trace 的核心实验证据。

## 默认 H2 与 recent0 的附录说明

为完整起见，我们同时报告原始 WREC-H2 与 refinement 后的 `recent_weight=0` 版本。后者在 `32/64/96/128` total expert-cache slots 上更强，而原始版本仅在 `192` slots 上略优。因此，正文将 refined 版本作为低中预算 regime 的主线策略，而原始版本保留为 high-budget robustness 的 appendix 对照。
