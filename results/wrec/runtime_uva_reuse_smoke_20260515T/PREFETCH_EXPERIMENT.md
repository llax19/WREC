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

## 目前还不能宣称什么

现在还不能宣称 WREC prefetch 已经提升了端到端延迟。

原因是：

- smoke 请求太短，不适合看吞吐或 TTFT 改善。
- 没有 stock/UVA-only baseline 的同配置对比。
- 成功那次 server 是前台 session 跑的，完整 server stdout 没落到 `server.log`。
- 首批 lazy init 仍会污染第一批 prefill/TTFT。
- sidecar policy 现在只是能给出建议，还没有针对 Qwen short/long serving 做系统调参。

当前能说的是：关键机制已经从“代码能编译”推进到“真实 vLLM/Qwen server 上能跑通，并且复用 UVA backing”。

## 下一步该怎么跑

下一步不要直接上大实验，应该做一个中等 smoke：

- `n=8` 或 `n=16`
- `max_new_tokens=8` 或 `16`
- prompt 尽量用真实 Dolly 子集，而不是 2 条完全相同短 prompt
- 保持 `slot_capacity=32`
- 对比至少三组：
  - stock UVA offload
  - WREC state-only
  - WREC finite-slot selective prefetch

每组都要保存：

- request-level latency log
- WREC residency stats
- sidecar metrics
- GPU memory samples
- 完整 server log

这一轮主要看：

- 是否仍然没有 extra CPU copy。
- 是否没有 fallback。
- `sum(H)/sum(U)` 是否比 short smoke 明显提高。
- missing copy bytes 是否下降。
- TTFT / latency 是否至少没有明显恶化。

只有这些成立，再进入长 prefill 或主实验。
