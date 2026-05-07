# WREC decode robustness literature scan

日期：2026-05-03

## 背景

昨天的 decode-only replay 说明：用 prefill train trace 学到的 WREC 先验不能稳定迁移到 decode。

本地关键事实：

- decode small trace：`64` requests，`1024` decode tokens，`32768` router events，`0` failures。
- prefill/decode hotness shift 明显：per-layer TV shift p50 `0.0549`、p90 `0.0746`、max `0.1181`；top-2 overlap mean `0.5625`。
- decode-only replay 中 WREC-H2 在 slots `64/96/128/192` 下弱于 LRU，gain 为 `-5.68%/-8.48%/-8.97%/-7.33%`。
- Belady 仍明显优于在线策略，说明 decode 有 locality，但当前 WREC 信号没有抓住它。

## 资料结论

1. Prefill 与 decode 应拆成两个 workload 建模。

- DuoServe-MoE 明确指出 MoE serving 有 phase disparity：prefill 会在多 token 上更密集地激活 experts，而 decode 每步只激活少数 experts；统一 expert loading/caching policy 会导致 prefill peak-memory blowup 或 decode tail-latency inflation。
- SARATHI 和 vLLM chunked prefill 文档也把 prefill 视为 compute-bound、decode 视为更 memory-bound/ITL-sensitive 的阶段，并通过 chunked prefill 与 decode-priority 调度减少阶段干扰。

对 WREC 的含义：prefill train trace 不能直接作为 decode prior 的主来源。decode 至少需要独立的 train prior，最好还要按 decode step/context length 分桶。

2. Decode 阶段需要轻量预测/预取，而不是粗粒度 global hotness。

- ProMoE 的核心是用 intermediate results 预测后续 expert usage，把 expert fetch 从 critical path 挪出去；其 prefill 和 decode 都报告了加速。
- Fate 用 adjacent-layer gate inputs 做 cross-layer expert prefetch，强调预测准确性，且使用 shallow-favoring cache。
- MoE-Infinity 使用 request/sequence-level activation tracing，说明 batch=1 或低并发 decode 中存在可复用的 sparse activation pattern。
- MoE-SpeQ 更激进：用小 draft model 预测未来 tokens 的 expert sequence，再把 host-device I/O 与计算 overlap。

对 WREC 的含义：WREC-H2 目前的 cross-layer transition 是从 prefill train trace 聚合出来的，decode 迁移失败不意外。应改成 decode-train 上的 step-aware / layer-aware predictor，并把 prefetch 行为限制在高置信和可 overlap 的情况。

3. Admission/eviction 应更重视鲁棒性，而不是继续调线性权重。

- TinyLFU 的思想是先比较新对象与 victim 的 recent frequency，再决定是否 admit，用 admission filter 避免 cache pollution。
- ARC 动态平衡 recency/frequency，并具有 scan-resistant 特性，适合 workload shift。
- MoE offloading analysis 资料也指出 LRU 并非最优，LFU 与 speculative prefetch 在 MoE traces 上有潜力。
- RAC 对 LLM cache 的观察是：真实 workload 可能存在 long reuse distance 和 sparse local recurrence，单纯短窗口 recency/frequency 不够鲁棒。

对 WREC 的含义：decode 中每个 request token 少、局部历史短，WREC-H2 的 request frequency 容易放大噪声。需要加入 admission guard、ghost cache / shadow history、reuse-distance-aware 的保守机制。

## 建议算法修改

### WREC-D0：先做 decode-specific prior

- 新增 decode train trace，不再用 prefill train trace 训练 decode policy。
- 统计单位从全局 `(layer, expert)` 改为：
  - `P_decode(expert | layer, step_bucket)`
  - `P_decode(expert | layer, previous_layer_experts, step_bucket)`
  - `P_decode(expert | layer, request_category/domain, step_bucket)`，如果 category 可用。
- step bucket 建议：`1`、`2-4`、`5-8`、`9-16`、`17-32`，后续按 trace 扩展。

### WREC-D1：phase-aware ensemble，而不是单 prior

把 score 拆成 4 个可独立 ablation 的信号：

```text
D_score =
  a * decode_step_prior
  + b * decode_cross_layer_prior
  + c * online_recency_frequency
  + d * prefill_static_hot_fallback
```

其中 `prefill_static_hot_fallback` 只做 fallback，不再主导 admission/eviction。若 decode prior 置信度低，退回 conservative LRU/TinyLFU，而不是强行用 prefill prior。

### WREC-D2：TinyLFU-style admission guard

Demand miss 后不总是 admit。只有满足以下条件才 admit：

```text
score(new_expert) > score(victim) + margin
and freq_estimate(new_expert) >= freq_estimate(victim) or predicted_next_use(new_expert) is near
```

否则 bypass。这样可以防止 decode 中一次性专家污染 cache。

### WREC-D3：ARC/ghost-cache 自适应 recency-frequency 权重

维护不占 GPU 的 shadow history：

- recent ghost：刚被 evict、随后很快 miss 的 experts。
- frequent ghost：长期重复 miss 的 experts。

若 recent ghost 命中增加，提高 recency 权重；若 frequent ghost 命中增加，提高 frequency/prior 权重。目标是让 WREC 对 prefill/decode shift 自适应，而不是手工固定 `recent_weight/request_weight/cross_layer_weight`。

### WREC-D4：decode prefetch 只做 confidence + overlap constrained

Prefetch 触发条件应改成：

```text
expected_saved_stall
  = P_hit_future * transfer_stall_ms
    - P_false_positive * waste_cost_ms
    - cache_pollution_penalty

trigger if expected_saved_stall > 0
and prefetch_can_overlap_with_non_moe_or_previous_layer_compute
and queue_depth_budget_not_exceeded
```

低置信时只做 keep/admission，不主动 prefetch。

### WREC-D5：layer budget 改成 decode-aware

Fate 和 DuoServe-MoE 都暗示浅层与深层应不同处理。建议先做简单版本：

- shallow layers：保留更多 static/decode-hot experts，减少早期 miss 对整步 decode 的阻塞。
- middle/deep layers：使用 cross-layer predictor 和 admission guard。
- total budget 下用 train decode trace 估计每层 `oracle_gap` 或 `miss_stall_contribution`，按贡献分配 slots。

## 下一步实验顺序

1. 采集小规模 decode train trace，例如 `train n=128, max_new_tokens=16`，先别扩大到正式规模。
2. 用 decode train prior 重跑当前 decode-only replay，比较：
   - LRU
   - static-hot from prefill train
   - static-hot from decode train
   - WREC-H2 prefill-prior
   - WREC-D0 decode-prior
3. 若 WREC-D0 仍不能超过 LRU，再实现 TinyLFU admission guard。
4. 若 D0+D2 稳定超过 LRU，再做 confidence-constrained prefetch。
5. 最后再扩大 trace 和 budget sweep。

## 参考资料

- MoE-Infinity: https://arxiv.org/abs/2401.14361
- ProMoE: https://arxiv.org/abs/2410.22134
- Fate: https://arxiv.org/abs/2502.12224
- DuoServe-MoE: https://arxiv.org/abs/2509.07379
- MoE-SpeQ: https://arxiv.org/abs/2511.14102
- MoE-Lightning: https://arxiv.org/abs/2411.11217
- SARATHI: https://arxiv.org/abs/2308.16369
- TinyLFU: https://arxiv.org/abs/1512.00727
- ARC: https://www.usenix.org/conference/fast-03/presentation/arc-self-tuning-low-overhead-replacement-cache
- In-depth Analysis on Caching and Pre-fetching in MoE Offloading: https://arxiv.org/abs/2511.05814
- vLLM chunked prefill documentation: https://docs.vllm.ai/en/v0.4.2/models/performance.html
