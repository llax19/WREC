# WREC Selective Prefetch Experiment

## 这组实验在做什么

这组实验想验证一件事：MoE expert 权重已经被 vLLM offload 到 CPU 后，WREC 能不能根据 router/sidecar 的预测，只把下一步可能会用到的 expert row 提前搬到 GPU 上。

换句话说，不是让每一层 MoE 的完整 expert 权重都常驻 GPU，而是只在 GPU 上保留一小组 shared slots。每次某个 MoE layer 要算之前，WREC 从 CPU backing 里把需要的 experts copy 到这些 slots，然后通过 `expert_map` 告诉 MoE kernel：logical expert id 现在对应哪个 slot。

目前这套实验验证的是链路能不能成立：

- vLLM 负责把完整 expert 权重 offload 到 CPU。
- WREC 复用 vLLM 的 CPU backing，不再自己多复制一份完整 expert 权重。
- sidecar 根据已观察到的 routed experts 给出“哪些 expert 值得提前放进 cache”的建议。
- attention forward 时触发 prefetch，利用 attention 和后续 MoE 之间的窗口做异步 H2D copy。
- MoE forward 前检查当前 batch 真正需要的 experts，已经 prefetched 的就是 hit，没在 slots 里的就同步补齐。

## 为什么要先修 UVA backing

之前 smoke 里反复出现：

```text
WREC expert residency falling back to an extra CPU copy
```

这说明 WREC 没拿到 vLLM offloader 已经创建好的 CPU backing，只能自己 clone 一份完整 MoE expert tensor。这样有两个问题：

- CPU 内存多一份完整 expert 权重，和“复用 vLLM offload backing”的实验设定不一致。
- 后面测到的内存和搬运开销会混入 WREC 自己的额外 backing，结果不好解释。

现在的改法是：UVA offloader 在权重加载和 post-load 处理完成后，把最终的 CPU storage 登记给 WREC。WREC 初始化 layer state 时优先拿这个 storage。之后无论 prefetch 还是 missing expert 补齐，都只从这份 backing 里按 expert row copy 到 GPU slots。

这轮 smoke 已经确认真实 serving 里能看到：

```text
WREC expert residency reusing vLLM CPU offload storage
```

并且 `wrec_residency_stats.jsonl` 里 `fallback_true=0`。

## 当前实现的执行流程

一次请求大致是这样走的：

1. 请求进入 vLLM，模型正常执行 attention 和 MoE。
2. 每次请求完成后，scheduler 拿到 vLLM 返回的 routed experts，并发给 WREC sidecar。
3. sidecar 根据历史、request-local 信息和 WREC policy，返回哪些 `(layer, expert)` 值得 admit。
4. WREC runtime 把这些建议存在本地 desired expert 表里。
5. 下一次同层 attention forward 开始时，WREC 查这个 layer 的 desired experts。
6. 如果该 layer 已经有 residency state，WREC 就把预测 experts 异步 copy 到 shared GPU slot pool。
7. 后续 MoE forward 真的开始时，WREC 根据 router logits 算出本次 batch 的真实 top-k unique experts。
8. 已经在 slots 里的算 hit，没在 slots 里的从同一份 UVA CPU backing 同步 copy 进来。
9. MoE kernel 通过 WREC 的 `expert_map` 使用 slot 中的专家权重。

目前 shared slot pool 是跨 layer 复用的。常规 Qwen MoE 层的 expert 权重 shape/dtype/device 相同，所以它们共用同一组 GPU slots。每个 layer 自己保留 logical expert 到 slot 的映射状态；当 shared pool 切到另一个 layer 时，旧 layer 的 resident 状态会失效。

## 这轮 smoke 说明了什么

这次跑的是一个很小的 short smoke：

- 模型：Qwen1.5-MoE-A2.7B
- 请求：2 条相同短请求
- 输出：每条只生成 1 token
- slots：32
- `max_num_batched_tokens=16`
- offload：UVA，只 offload `experts.w13_weight` 和 `experts.w2_weight`

结果：

- 2 条请求都完成。
- WREC 观察到 24 个 MoE layer。
- residency stats 一共 96 行。
- `fallback_true=0`，说明没有触发 slot overflow fallback。
- `hit_rows=24`，说明 attention prefetch 到 MoE forward 的链路确实产生过命中。
- `avg_prefetch_hit_rate=0.0456`，`max_prefetch_hit_rate=0.3333`。
- sidecar 收到 240 个 router events，960 个 expert refs。

最重要的结论是：这不是性能收益实验，而是集成 smoke。它证明了以下几件事：

- vLLM server 能跑通。
- WREC 能复用 UVA offload backing。
- WREC finite slots 能接到真实 MoE forward。
- sidecar 预测能进入 attention prefetch hook。
- MoE forward 能统计到 prefetch hit。

## 为什么 hit rate 现在很低

这轮 `avg_prefetch_hit_rate=0.0456` 很低，但这不意外。

第一，样本太小。只有 2 条短请求，每条只生成 1 token。WREC 的 request-local history 和 recent history 还没积累起来，sidecar 的判断基础很弱。

第二，第一批 forward 天然是冷启动。当前 residency state 是第一次 MoE forward 时创建的。在此之前，同层 attention hook 找不到 state，所以第一批同层 prefetch 不能发生。这会让前面很多行都是 `H=0`。

第三，当前平均值是按 stats row 简单平均。每个 layer、每次 forward 都一行，短请求里很多行没有足够预测窗口，直接拉低均值。后续正式分析应该至少同时看：

- `sum(H) / sum(U)`，也就是按 expert 需求加权的总体命中率。
- prefill 和 decode 分开统计。
- 第一条请求和后续请求分开统计。
- 每层命中率分布。
- missing experts 的同步 copy bytes。

所以这轮 hit rate 低，只能说明“当前极短 smoke 下策略收益很弱”，不能说明 selective prefetch 方案无效。

通过单次smoke暴露出来的问题：
1. 先解决 CPU backing 复用 UVA storage
2. 再解决 lazy init，提前注册 MoE layers
3. 再处理长 prefill request timeout
4. 最后跑 slots=8/16/24/32 主实验

## lazy init 修复方案

问题：旧实现只在第一次 `FusedMoE.forward_native` 里创建 residency state。由于同层 attention hook 发生在 MoE forward 之前，第一批 prefill 的 attention-window prefetch 会因为找不到 layer state 而直接跳过。

修复：在 UVA offloader `post_init()` 中，等 `experts.w13_weight` 和 `experts.w2_weight` 的 CPU offload storage 都登记完成后，根据参数路径定位到对应 `mlp.experts` 模块，并调用 `WrecExpertResidencyManager.register_layer()` 提前创建 state、expert_map 和 shared finite slots。

结论：部署后的本地 smoke 已确认，`register_layer()` 能在首次 MoE forward 前建立 `layer_index -> state` 映射，随后 `maybe_prefetch_for_attention()` 可以直接把 desired expert 预取进 slots。真实 serving 还需要再跑 short smoke，看日志中的 `WREC expert residency eagerly registered FusedMoE layer ...` 是否覆盖 24 个 MoE layers，并比较首批 stats row 的 `H/M`。

## lazy init smoke 结果

路径：`results/wrec/runtime_lazy_init_smoke_20260515T`

- 状态：success，2 条 short requests 完成。
- server log 中 `WREC expert residency eagerly registered FusedMoE layer ...` 出现 24 次，覆盖 Qwen 的全部 MoE layers。
- `WREC expert residency reusing vLLM CPU offload storage` 出现 48 次，对应每层 `w13/w2`。
- `falling back to an extra CPU copy` 为 0，`slot overflow` 为 0。
- `wrec_residency_stats.jsonl` 共 96 行，`sum(H)/sum(U)=166/908=0.1828`，row-average hit rate 为 0.1170，max row hit rate 为 0.6923。

结论：lazy init 的 runtime 集成问题已经解决。首批 stats row 仍有 `H=0`，但这时原因变成 sidecar 在第一轮 attention 前还没有 desired experts，而不是 layer state 尚未创建。后续若要继续降低首批冷启动，应做 sidecar warm prior 或从 trace/static-hot 初始化 desired experts。

## ranked expert prefetch 方案

旧 runtime sidecar 只根据 `would_admit/would_evict` 维护 desired experts；desired set 是 online shadow cache admission 的副产物，不是每层全 expert 排名。

新方案：

- sidecar 增加 `/rankings`，返回所有 layer 的 `ranked_experts_by_layer`。
- sidecar 每次 `/event` 后，返回当前 layer 和下一层的 ranked experts。
- vLLM sidecar client 启动时先拉一次 `/rankings`，让 runtime 在首批 forward 前有 static prior ranking。
- WREC attention hook 优先使用 `ranked_experts_by_layer[layer][:active_slots]`，如果没有 ranking 再 fallback 到旧 desired experts。

这解决的是“候选生成”问题：runtime 不再依赖 shadow admission 产生候选，而是按当前 active slots 从每层 expert 排名中截断。

## ranked prefetch smoke 结果

路径：`results/wrec/runtime_ranked_prefetch_smoke_20260515T`

- 状态：success，2 条 short requests 完成。
- `wrec_residency_stats.jsonl` 共 96 行，`sum(H)/sum(U)=601/908=0.6619`，row-average hit rate 为 0.4192，max row hit rate 为 1.0000。
- block-level：前两个 24-layer block 仍为 `H=0`；第三个 block `243/358=0.6788`；第四个 block `358/358=1.0000`。
- sidecar ranking 开销：`0.1021s` total，`425.43us/router event`。
- 没有 extra CPU copy fallback，没有 slot overflow。

边界：该 smoke 证明 ranked-expert prefetch 已改变 runtime 机制，但不能证明性能收益。2 条短请求下 TTFT 比 lazy-init smoke 更高，说明 `top-active_slots` 且 `active_slots=32` 的预取过于激进，H2D transfer 开销明显。后续需要加 score gate，避免低价值 experts 进入 desired set；数量约束仍交给 runtime 的 `active_slots`。

## ranking 过滤策略

绝对 score threshold 已接入 sidecar 和 runner：

- sidecar CLI：`--ranking-score-threshold`
- runner env：`SIDECAR_RANKING_SCORE_THRESHOLD`

当前回退为只使用绝对 threshold gate。`score=0` 在静态 base score 中近似表示预取收益与搬运成本打平：`p_use * expected_tokens * miss_stall_ms - transfer_ms = 0`。但 runtime total score 还会叠加 recent/request/cross-layer 分数并受权重影响，所以 `threshold=0` 不是跨配置的自然常数，只能作为一个保守实验点。

语义：

1. 对每层所有 experts 计算 score 并排序。
2. 可选绝对门槛：保留 `score >= ranking_score_threshold`。
3. runtime attention hook 再按 `active_slots` 截断。

设计边界：sidecar 只负责 score-based filtering，不负责按数量 cap 每层候选。每次能预取多少由 runtime 根据当前 `active_slots` 和真实 slot pool 决定。
