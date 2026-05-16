# Qwen 真实 serving 中 decode 退化的结构性分析

## 1. 文档目的

本文档用于整理 `Qwen1.5-MoE-A2.7B + vLLM + WREC selective prefetch runtime` 在真实 serving 实验中出现的 decode 性能退化问题，并给出当前证据能够支持的结构性判断。本文档的用途是：

- 作为后续毕业论文撰写时关于 decode 行为、runtime 边界和 negative result 的项目依据。
- 作为后续是否继续投入 decode 方向实现工作的判断依据。
- 作为后续实验设计中“哪些结论可以写、哪些结论不能写”的约束依据。

本文档不讨论 WREC score 本身是否有信息量，重点只放在当前 runtime 结构能否把这些信息量转化为 decode 阶段的真实收益。

## 2. 需要回答的问题

本轮分析关注的问题是：

1. 当前 WREC runtime 在 decode 阶段为什么没有得到收益，反而显著慢于 baseline。
2. 这个问题主要是参数选择不当，还是当前 runtime 结构本身不适合 decode。
3. 如果当前结构本身不适合 decode，这个判断应当如何被记录和表述。

## 3. 相关实验与结果目录

### 3.1 baseline：`no_wrec`

- 结果目录：`results/wrec/runtime_qwen_real_table_b_full_n64_20260515T172203Z/`
- 负载：`data/prompts/wrec_dolly_prefill_eval_long_n64_maxnew16_20260514.jsonl`
- 请求数：`64`
- 关键指标：
  - `input_tokens_per_s=37.08`
  - `output_tokens_per_s=2.048`
  - `p95_ttft_ms=4099.25`
  - `p95_tpot_ms=261.48`
  - `p95_e2e_ms=8018.48`

这是当前 full-serving 口径下的对照基线。

### 3.2 WREC 主线原始版本：decode 固定 `active_slots=32`

- 结果目录：`results/wrec/runtime_qwen_real_table_b_full_n64_20260515T172203Z/`
- 方法：`finite_slot` runner 参数名，对应当前 `WREC 主线 selective prefetch`
- 说明：该轮实验中途停止，但保留了完整 baseline、WREC partial `server.log`、`sidecar_metrics.json` 与 `wrec_residency_stats.jsonl`

该轮 partial 结果已经足以说明 decode 方向的主要矛盾：

- `server.log` 中 WREC decode `Avg generation throughput` 大致只有 `0.3~0.5 tok/s`
- 同一阶段 baseline `no_wrec` 大致在 `1.6~2.5 tok/s`
- `wrec_residency_stats.jsonl` 中 decode 行满足：
  - `num_tokens=1`
  - `active_slots=32`
  - `sum(H)/sum(U)=0.5264`
  - `avg_H≈2.105`
  - `avg_M≈1.895`
  - `avg_prefetched_len≈31.17`
  - `sum_waste_transfer_bytes≈6.67 TB`

这说明原始版本的 decode 行为是：

- 候选集大，命中率并不低；
- 但无效预取极重，generation throughput 仍显著下降。

### 3.3 WREC 修复版：启用动态 `active_slots`

- 结果目录：`results/wrec/runtime_qwen_real_table_b_full_n64_alpha1_wrec_only_20260515T182131Z/`
- 配置文件：`configs/runtime/qwen_real_table_b_full_n64_maxnew16_alpha1_wrec_only.env.sh`
- 核心改动：设置 `WREC_EXPERT_RESIDENCY_ACTIVE_SLOT_ALPHA=1`
- 运行方式：只补跑 WREC 主线，不重跑 baseline
- 运行设备：由于物理 `GPU0` 被其他进程占用，本轮切到 `CUDA_VISIBLE_DEVICES=1`

该轮完整结果：

- `input_tokens_per_s=19.95`
- `output_tokens_per_s=1.101`
- `p95_ttft_ms=6687.04`
- `p95_tpot_ms=559.36`
- `p95_e2e_ms=15089.47`

decode 机制统计：

- decode 行主组合变成 `((1, 4), 23064)`
- `sum(H)/sum(U)=0.0774`
- `avg_H≈0.3095`
- `avg_M≈3.6905`
- `avg_prefetched_len≈3.98`
- `sum_sync_copy_bytes≈1.47 TB`
- `sum_waste_transfer_bytes≈1.46 TB`

这说明修复生效了，但端到端指标仍显著差于 baseline。

## 4. 当前代码中的相关结构

### 4.1 attention 与 MoE 的相对位置

`Qwen2MoeDecoderLayer.forward()` 的顺序是：

1. `self_attn`
2. `post_attention_layernorm`
3. `mlp`

对应文件：`external/vllm-0.19.0/vllm/model_executor/models/qwen2_moe.py`

具体位置：

- `Qwen2MoeDecoderLayer.forward()`：约第 `337-357` 行
- `self_attn` 调用：约第 `349-352` 行
- `mlp` 调用：约第 `355-356` 行

这说明对同一层而言，attention 到 MoE 之间只有一段很短的 layer-local 窗口。

### 4.2 prefetch 在哪里触发

attention 侧会调用：

- `get_wrec_expert_residency_manager().maybe_prefetch_for_attention(self.layer_name, int(query.shape[0]))`

对应文件：

- `external/vllm-0.19.0/vllm/model_executor/layers/attention/attention.py`
- 约第 `421-424` 行

这里传入的 `num_tokens` 是 `query.shape[0]`，也就是本次 attention forward 的 token 数。

### 4.3 MoE 在哪里消费 prefetch 并检查 miss

MoE forward 会调用：

- `wrec_residency_manager.begin_forward(self, router_logits)`

对应文件：

- `external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/layer.py`
- 约第 `1478-1486` 行

`begin_forward()` 内部会：

1. 先等待 prefetch event
2. 计算 `current_experts`
3. 计算 `missing_experts`
4. 对缺失 expert 走 `_onload_experts()`
5. 记录 `U/H/M/prefetch_hit_rate/sync_copy_bytes/waste_transfer_bytes`

对应文件：

- `external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- 约第 `142-180` 行

### 4.4 `active_slots` 的来源

`active_slots` 的计算逻辑在：

- `external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- 约第 `841-855` 行

规则为：

- 若 `active_slot_alpha <= 0`，则 `active_slots = slot_capacity`
- 否则：
  - `active_slots = ceil(alpha * top_k * num_tokens)`
  - 再做 `max(top_k, ...)`
  - 再做 `min(slot_capacity, ...)`

因此：

- 代码本身已经支持“按本次 forward token 数动态决定 active slots”
- 原始实验没有启用这一逻辑，只是因为 `active_slot_alpha=0`

### 4.5 shared-slot pool 的关键约束

当前 shared-slot pool 的 key 不包含 `layer_id`，只包含：

- `slot_capacity`
- 权重 shape
- dtype
- device

对应文件：

- `external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- `_shared_slot_pool_key()`，约第 `662-675` 行

因此同形状层会共享同一个 slot pool。

更关键的是，owner 切换时会执行：

- `_invalidate_layer_slots(previous_state)`
- `_invalidate_layer_slots(state)`

对应文件：

- `external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- `_claim_shared_slot_pool()` 与 `_invalidate_layer_slots()`，约第 `809-829` 行

该逻辑会清空：

- `resident_experts`
- `logical_to_slot`
- `slot_to_logical`
- `last_prefetched_experts`

这意味着 layer 间不是“共享池上共存”，而是“共享池上轮流占有”。

## 5. decode 阶段的本质矛盾

### 5.1 decode 的两个可选收益来源

对 expert offload 来说，decode 想获得收益，通常只能依赖两类机制：

1. **短窗口 overlap**
   在同一层 attention 之后、同一层 MoE 之前，把后续 MoE 需要的 expert 提前搬上 GPU。

2. **跨 token 的同层复用**
   token `t` 在 layer `l` 上搬运过的 expert，到 token `t+1` 再次经过同一 layer `l` 时仍然驻留，因此不必重复搬运。

当前结构对这两类机制都不友好。

### 5.2 当前结构对短窗口 overlap 不友好

原因很直接：

- prefetch 触发点在 attention 内
- 真正消费点在紧接着的同层 MoE
- 中间只有 `post_attention_layernorm` 和少量层内计算

在 prefill 时，由于 token 数大、attention 自身更重，这个窗口仍有一定意义。  
在 decode 单 token 时，这个窗口非常短，不足以稳定隐藏 expert copy。

换言之，当前结构给 decode 提供的 overlap 空间天然比 prefill 小很多。

### 5.3 当前结构对跨 token 的同层复用不友好

decode 的执行顺序不是：

- `layer 0` 连续处理多个 token

而是：

- token `t` 经过 `layer 0 -> layer 1 -> ... -> layer L-1`
- token `t+1` 再次经过 `layer 0 -> layer 1 -> ... -> layer L-1`

因此如果希望在 `layer l` 上形成跨 token 复用，必须保证：

- `layer l` 在 token `t` 用过的 resident experts
- 在 token `t+1` 回到 `layer l` 之前不被其他层清掉

但当前 shared-slot pool 的 owner 切换规则恰恰会在每次跨层时清空前一层 resident set。于是：

- `layer l` 在 token `t` 留下的 resident
- 还没等 token `t+1` 回到 `layer l`
- 就已经在 `layer l+1, l+2, ...` 的 owner 切换中被失效掉了

这使得 decode 很难积累“同层跨 token 命中”。

## 6. 两轮 WREC decode 结果揭示的 tradeoff

### 6.1 固定 `active_slots=32` 的版本

这个版本的 decode 行为可以概括为：

- 候选集很大
- recall 尚可
- waste 很高
- decode throughput 很差

证据：

- `sum(H)/sum(U)=0.5264`
- `avg_prefetched_len≈31.17`
- `sum_waste_transfer_bytes≈6.67 TB`
- `server.log` generation throughput 大约 `0.3~0.5 tok/s`

它说明：

- 当前 ranking 在大候选集下并不是完全无效
- 但共享 slot 切换 + 高频大规模预取会把 decode 拖垮

### 6.2 动态 `active_slots=4` 的版本

这个版本的 decode 行为可以概括为：

- 候选集收缩到了与 `top_k` 同量级
- waste 下降
- 但 recall 崩得更严重
- decode throughput 仍明显差于 baseline

证据：

- `sum(H)/sum(U)=0.0774`
- `avg_H≈0.3095`
- `avg_M≈3.6905`
- `avg_prefetched_len≈3.98`
- `output_tokens_per_s=1.101`
- `p95_tpot_ms=559.36`

它说明：

- 当前 ranking 并不具备“在 decode 的超小预算下精确命中本步 top-k expert”的能力
- 只把候选集从 `32` 缩到 `4`，不能解决 decode 问题

### 6.3 当前结构暴露出的核心 tradeoff

目前观察到的是一个很强的两难：

- 候选集大：recall 较高，但 waste 很大，decode 仍慢
- 候选集小：waste 下降，但 recall 过低，decode 仍慢

这个两难不是单纯靠调阈值就能解释的，因为它来自两类约束同时存在：

1. shared-slot 生命周期太短
2. score/ranking 对小预算 decode 预测不够 sharp

## 7. 当前证据能支持的判断

### 7.1 可以支持的判断

当前证据已经足够支持以下判断：

1. **当前 WREC selective prefetch runtime 更适合 prefill，不适合 decode。**
   prefill 至少存在更长的 layer-local overlap window，也允许更宽的候选集；decode 则同时要求更短路径、更高精度和更持久 residency。

2. **当前 decode 退化不能简单归因于“sidecar 太慢”。**
   ranking-only 之后 `online_loop_us_per_expert_ref` 已经很低，decode 退化的主要来源不在 sidecar Python 环路。

3. **当前 decode 退化不能简单归因于“active slots 公式错了”。**
   动态 `active_slots` 已经验证成功，但性能问题依旧存在。

4. **在“单共享 slot pool + 跨层 owner 失效 + 单层短窗口预取”结构下，decode expert offload 很难获得稳定收益。**
   这是目前最重要的结构性判断。

### 7.2 当前证据尚不能支持的判断

当前还不能据此得出以下结论：

1. “WREC score 对 decode 完全没有信息量。”
   当前只能说它不足以在现有结构下支持小预算 decode 命中，不等于完全无效。

2. “任何 decode prefetch 都不可能有效。”
   当前否定的是现有结构，不是否定所有可能的 decode prefetch 结构。

3. “只要做 per-layer 持久 residency 就一定能解决 decode。”
   当前没有直接实验支持这一点，只能把它列为未来结构方向。

## 8. 对论文叙事的建议边界

### 8.1 主线卖点应放在哪里

较稳妥的主线是：

- WREC score 利用 routed experts 的结构性信号，识别高价值 expert
- selective prefetch runtime 能在 prefill 主导场景下把一部分同步 copy 转化为 overlap
- 在当前结构下，这一机制的收益主要体现在 prefill 相关指标，而不是 decode

### 8.2 decode 部分建议如何表述

建议把 decode 写成“结构性边界”或“negative result”，而不是“尚未调优完成”。

合适的表达应接近：

- 当前 runtime 结构在 decode 阶段存在两类限制：预取窗口短、跨 token 同层 residency 不稳定。
- 因此 decode 阶段的主要挑战不是是否存在 ranking 信号，而是该信号缺少足够合适的 runtime 承载结构。
- 实验表明，扩大 decode 预取预算会带来较高 waste，缩小预算又会导致命中率显著下降；这表明当前结构下 decode selective prefetch 存在明确边界。

不建议写成：

- “decode 方向仍需进一步调参”
- “再做一些工程优化后预计可以解决”

因为目前证据更接近结构不匹配，而不是单点工程问题。

## 9. 对后续实现工作的建议

如果目标是尽快形成稳定论文结论，建议优先级如下：

1. 将当前 runtime 主线明确收缩为 **prefill-first selective prefetch**。
2. decode 部分保留为结构性负结果分析，不继续在现有 shared-slot 结构上做大量微调。
3. 只有在明确接受较大结构改动时，才继续考虑 decode 方案，例如：
   - per-layer 持久 slot pool
   - 分区式共享 pool，而非单 owner 共享 pool
   - 针对 decode 重写预测目标，使其直接服务跨 token residency，而不是服务下一层短窗口 prefetch

若不准备做上述结构级改动，则当前最合理的项目边界是：

- 将 WREC runtime 的论文结论限制在 prefill 主导收益与结构性 decode 边界
- 不再把“decode 一并获益”作为必须完成的目标

## 10. 总结

当前 decode 退化问题已经可以归纳为以下一句话：

> 当前 WREC runtime 在 decode 阶段同时缺少足够长的 overlap window 和足够稳定的跨 token same-layer residency，因此 shared-slot selective prefetch 结构很难把 ranking 信号转化为稳定的 decode 收益。

就目前证据而言，这不是一个适合再靠阈值、slot 数或 sidecar 常数项优化去解决的问题，而是一个应当被明确记录为 runtime 结构边界的问题。
