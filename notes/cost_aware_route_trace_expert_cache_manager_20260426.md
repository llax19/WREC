# Cost-aware route-trace-driven expert cache manager 阅读笔记与落地方案

日期：2026-04-26

## 1. 方向判断

当前工作区已有的主线是 request scheduling：`FCFS`、`Length-only`、`LTR-lite`、`MoE-affinity`。这些实验已经说明一件事：真实 route trace 确实包含 expert locality，但仅把 locality 用来重排 waiting queue，不一定能稳定转化为端到端 latency 收益。

因此更贴合论文题目“面向资源受限环境的混合专家大模型多维资源协同推理优化方法”的研究方向，应从“请求调度”转为“专家缓存、预取、offload 与选择性推测执行”的协同优化：

```text
route trace / gate signal / cache state
        -> cost-aware cache action
        -> lower miss, lower PCIe traffic, lower offload stall, bounded quality loss
```

这个方向的核心不是把 expert prediction accuracy 从 95% 做到 98%，而是让预测结果直接服务于系统代价：

```text
Cost = miss penalty + transfer cost + stall cost + quality penalty
```

输出也不应只是 expert id，而应是动作：

- `prefetch expert e`
- `keep expert e`
- `evict expert e`
- `skip prefetch`
- `speculatively execute cached alternative`

## 2. 论文阅读顺序

建议先按下面顺序阅读和复现思想：

1. MoE-Infinity：理解 request/sequence-level activation tracing，以及 trace 如何转成 prefetch/cache 依据。
2. Fate：理解 cross-layer gate signal 和 shallow-favoring cache。
3. ExpertFlow：理解 predictor 如何和 token scheduling、expert cache engine 耦合。
4. DALI：理解 offloading 下 CPU/GPU workload、PCIe stall、cache replacement 的系统建模。
5. Speculating Experts 与 Cache-Conditional Experts：理解预测错后不一定回退，以及 routing 可以被 cache state 影响。

## 3. 论文速览表

| 论文 | 主要问题 | 方法关键词 | 对本方向的启发 | 需要避开的坑 |
| --- | --- | --- | --- | --- |
| MoE-Infinity | MoE 参数 offload 后按需加载太慢，普通 LRU/LFU 不懂 expert sparsity | sequence-level EAM trace、activation-aware prefetch/cache | trace 不能只做全局频率，要保留 request/sequence 粒度；cache priority 应考虑 activation ratio 与 layer position | 更偏 trace matching 与启发式 cache，不是直接学习 cost-aware action |
| Fate | edge/offload 场景 expert prediction 不准会造成长 stall | cross-layer gate、prefill popularity order、shallow-favoring cache、quantization | 当前层 gate input 可以预测相邻层；浅层预测不稳定，适合优先常驻；预取数量应受 compute-overlap time budget 限制 | 仍以 expert prediction 为中心，cache/prefetch 规则较手工 |
| ExpertFlow | 预测、token rebatching、expert cache 分离会导致系统收益有限 | RPP、Token Scheduler、Expert Cache Engine、PLEC | predictor 应与 cache engine 同时设计；预测要早到足以驱动预取；misprediction 要有 runtime correction | 训练目标仍接近 routing path BCE，不是系统 cost；token rebatching 会引入 KV cache 复杂度 |
| DALI | 本地 PC offload 中 CPU/GPU workload 不均、预取不准、cache hit 低 | greedy CPU/GPU assignment、residual-based prefetch、workload-aware cache replacement | cache action 应考虑 expert workload，而不仅是是否会访问；系统指0标要包含 PCIe traffic、GPU idle、CPU/GPU load balance | 目标平台偏 CPU+GPU mixed compute；移植到 vLLM/Qwen 需要收缩实验边界 |
| Speculating Experts | decode 阶段 CPU-GPU expert transfer 成为瓶颈，按需加载无法重叠 | representation-based speculative prefetch/execution、selective estimator | misprediction 后不一定必须 fallback；可执行 cached alternative，用 quality penalty 约束 | 质量边界要实测，不能只报告 latency |
| Cache-Conditional Experts | memory-constrained device 上普通 routing 不考虑 cache locality | cache-aware reranking、cache prior、Top-J fidelity guard | routing/action 可以显式依赖 cache bitmap；router entropy/margin 可决定替代空间 | 会改变原模型路由，必须报告 perplexity/task accuracy delta |

## 4. 单篇阅读笔记

### 4.1 MoE-Infinity

资料：

- arXiv: https://arxiv.org/abs/2401.14361
- HTML: https://ar5iv.labs.arxiv.org/html/2401.14361v3
- Code: https://github.com/TorchMoE/MoE-Infinity

核心问题：

MoE 参数量主要集中在 experts，资源受限环境无法让全部专家常驻 GPU。若按需从 CPU/SSD 拉取专家，PCIe/存储传输会让 per-token latency 变得很高。普通 offloading 系统为 dense model 设计，倾向于全量预取或 LRU/LFU，无法利用 MoE 的稀疏激活与 temporal locality。

关键设计：

1. Sequence-level Expert Activation Matrix，简称 EAM。
   - 对每个 sequence 单独记录每层每个 expert 的激活次数。
   - 不把不同请求的 expert 频率全局混在一起，因为全局聚合会抹平 request-level locality。
   - 用 EAM Collection 表示若干典型 routing pattern。

2. Activation-aware prefetch。
   - 根据当前已经观测到的 EAM，匹配 EAMC 中相似模式，预测后续层会访问的专家。
   - prefetch priority 同时考虑预测激活比例和距离当前执行层的远近。

3. Activation-aware cache。
   - 优先保留当前 sequence 里 activation ratio 高的专家。
   - 也偏向保留浅层专家，因为浅层距离执行点太近，留给预取的时间更短。

对我们有用的部分：

- 现有 `scripts/moe_affinity/build_qwen_router_trace_signatures.py` 已经能得到 request-level sparse expert signature，但它是聚合版，下一步应扩展到 token/layer event trace。
- cache simulator 的 oracle baseline 可以从 MoE-Infinity 的 EAM 思路开始：`route-window oracle`、`EAM-nearest prefetch`、`activation-ratio cache`。
- 论文中“浅层更适合缓存、远层更适合预取”的分层思想，可以转成我们的 cost model：浅层 miss penalty 更高，因为 overlap window 更短。

局限：

MoE-Infinity 本质上是 trace-driven heuristic。它证明 trace 有用，但没有把输出定义成可学习的 cache action，也没有系统处理“预测错了但可以不回退”的情况。

### 4.2 Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate

资料：

- arXiv: https://arxiv.org/abs/2502.12224
- HTML: https://ar5iv.labs.arxiv.org/html/2502.12224v2

核心问题：

offload-based MoE 在 edge 环境中依赖 expert prediction。预测不准会导致专家加载延迟，GPU 等待 I/O。Fate 的目标是用低开销的 cross-layer signal 提高 expert prefetch 准确率。

关键设计：

1. Cross-layer gate prefetch。
   - 在第 `i` 个 block 的 gate 输入出现时，把该输入复制到 CPU，并并行送入第 `i+1` 个 block 的 gate。
   - 用相邻层 gate input 的相似性预测下一层 experts。
   - 预测过程放在 CPU，尽量不增加 GPU 推理开销。

2. Prefill 与 decoding 分开处理。
   - Prefill 阶段 token 多、专家多，按预测后的 expert popularity 降序传输，使高 workload experts 更早可用。
   - Decoding 阶段每步 token 少，重点是估算当前 compute window 能覆盖多少 expert transfer。

3. Shallow-favoring expert cache。
   - 浅层 gate 分布更均匀，预测更不稳定。
   - Fate 优先缓存浅层 experts；Qwen1.5-MoE 里示例边界是前几层。
   - 该策略把 expert hit rate 提高到约 99% 的量级。

对我们有用的部分：

- Predictor input 不应只放历史 expert ids，也应加入当前/上一层 gate input、router entropy、top-k margin。
- action 约束应包含 time budget：只有能在 compute window 内完成的 prefetch 才值得发起。
- cache allocation 可以按层非均匀：浅层更多 cache，深层更多 prefetch。

局限：

Fate 的目标仍偏 expert prediction 和 hand-crafted cache allocation。我们的创新点应当进一步把目标改成 action cost，学习“该不该预取、预取谁、是否允许 cached alternative”。

### 4.3 ExpertFlow

资料：

- arXiv: https://arxiv.org/abs/2410.17954
- HTML: https://arxiv.org/html/2410.17954v2

核心问题：

offload 缓解了 GPU memory，但会引入 routing-dependent expert loading。单独做 static cache、单独训练 predictor 或单独 token scheduling 都不足以解决整体瓶颈。

关键设计：

1. Routing Path Predictor。
   - 用小型 transformer-style predictor 一次性预测所有层的 expert activation。
   - 输出 batch/layer/expert 级 routing matrix，为早期 prefetch 和 cache planning 服务。

2. Token Scheduler。
   - 将相邻 batch 的 tokens 按 predicted routing path 重组。
   - 目标是减少每个 batch 触达的 unique experts，提高每个 expert 处理的 token 数。

3. Expert Cache Engine。
   - Predictive Locality-aware Expert Caching 根据预测的 routing pattern 分配 cache slots。
   - runtime correction 处理实际 route 与预测不一致的情况。

对我们有用的部分：

- 我们的系统结构可以借鉴三段式：`Action Predictor -> Cache Manager -> Runtime Correction`。
- 不必先追求复杂 token rebatching，可以先保留当前 request/batch 顺序，只在 expert cache 层落地。
- `Routing Path Dataset` 的做法值得复用：对每条请求收集输入、输出、每层每 token route，并划分 train/test。

局限：

ExpertFlow 仍以 routing path accuracy 为训练目标，代价函数间接。我们的论文可以强调：同样的预测错误，在不同 cache state、layer、transfer queue 下代价完全不同，因此应训练 action value 或 action rank，而不是只训练 expert id。

### 4.4 DALI

资料：

- arXiv: https://arxiv.org/abs/2602.03495
- HTML: https://arxiv.org/html/2602.03495v1

核心问题：

本地 PC MoE offloading 不只是“专家在不在 GPU”的问题，还包括 CPU/GPU compute 如何分工、哪些 experts 是高 workload、cache replacement 是否跟得上动态 workload。

关键设计：

1. Greedy Assignment。
   - 将 expert assignment 建模为 CPU/GPU 0-1 分配问题。
   - 用 runtime greedy 近似代替昂贵的最优求解。

2. Residual-Based Prefetching。
   - 用跨层 residual 信息预测 high-workload experts。
   - 重点不是预测所有专家，而是准确命中高 workload experts。

3. Workload-Aware Cache Replacement。
   - 根据历史 workload score 更新 GPU cache。
   - 替换策略关注“未来能减少多少工作量和 PCIe 代价”，而不是简单 recency。

对我们有用的部分：

- 我们的 action value 可以显式使用 workload score：

```text
value(prefetch e) =
  P(use e) * tokens_routed_to_e * miss_penalty(e, layer)
  - transfer_cost(e)
  - prefetch_waste_penalty
```

- cache replacement 不应只看 hit/miss，还应看一个 miss 影响多少 tokens、是否会造成 GPU idle。
- 评估必须加入 PCIe traffic、GPU idle time、prefetch waste 和 stall time。

局限：

DALI 更强调 CPU/GPU 混合计算，而我们当前 vLLM/Qwen 环境主要是 GPU 推理加 CPU offload。因此落地时先收缩为 expert cache/prefetch simulator，不急着实现 CPU/GPU compute assignment。

### 4.5 Speculating Experts Accelerates Inference for Mixture-of-Experts

资料：

- arXiv: https://arxiv.org/abs/2603.19289
- HTML: https://arxiv.org/html/2603.19289v1

核心问题：

在 memory-constrained decode 中，专家权重必须从 CPU 拉到 GPU，copy time 可以成为 TPOT 的主要瓶颈。按需加载时，GPU 会等待专家传输。

关键设计：

1. 利用当前已计算的内部表示推测未来 experts。
2. 将 expert transfer 与 attention/router/MLP compute 重叠。
3. 对部分模型，直接执行 speculated experts 而不是等待 router-selected true experts，质量仍能保持。
4. 对质量更敏感的模型，再加轻量 estimator 提高 hit rate。

对我们有用的部分：

- “预测错了是否必须 fallback”是重要创新口。
- 我们可以把动作空间扩展为：

```text
if true expert not cached:
  either wait and fetch true expert
  or execute cached alternative expert
```

- 这个动作必须加 quality guard：
  - router margin 小，说明 top experts 接近，可以更大胆替代；
  - cached alternative rank 高，风险更小；
  - Top-J critical experts 必须保留原 route；
  - 替代动作的离线校准指标是 delta NLL / perplexity / task accuracy。

局限：

Speculating Experts 给了方向，但 selective speculative execution 在论文里必须有质量边界。不能只说 latency 下降，需要说明哪些层、哪些 margin、哪些任务允许替代。

### 4.6 Mixture of Cache-Conditional Experts

资料：

- arXiv: https://arxiv.org/abs/2412.00099
- HTML: https://ar5iv.labs.arxiv.org/html/2412.00099v2
- OpenReview: https://openreview.net/forum?id=ul4W26KEKz

核心问题：

普通 MoE routing 完全按 router logits 选 experts，不考虑 cache state。在 memory-constrained device 上，这会造成频繁 cache miss。

关键设计：

1. Max-rank / cumulative threshold routing。
   - 如果非 top-1 的 cached expert 概率足够接近，也可以提升它，换取 cache hit。
   - threshold 根据 router 分布不确定性动态调整。

2. Cache prior reranking。
   - 给已经在 cache 中的 experts 增加 logit prior。
   - 用改动后的 logits 决定 ranking，但仍用原 logits 计算 expert weights，尽量保持模型输出稳定。

3. Top-J fidelity guard。
   - 最关键的 Top-J experts 始终保留，不让 cache-aware routing 破坏核心语义。

对我们有用的部分：

- cache bitmap 应进入 predictor input。
- selective speculative execution 可以看作更保守的 cache-conditional routing：不总是改 route，只在代价收益明显且质量风险低时改。
- `router entropy / margin / cumulative probability` 可以作为 quality penalty 的代理特征。

局限：

它会改变原模型 route，因此我们的实验必须报告 quality delta。否则系统收益会被质疑是用准确率换来的。

## 5. 拟定研究问题

建议将后续研究问题表述为：

在资源受限的 MoE 推理环境中，如何利用 route trace、gate signal 和当前 cache state，直接学习专家缓存与预取动作，在保证输出质量基本不变的前提下，降低 expert miss、CPU-GPU 迁移、offload stall 与 TPOT 尾延迟？

可以拆成三个子问题：

1. Route trace 中是否存在可被 cache manager 利用的短窗口 expert locality？
2. 相比 expert-id prediction，cost-aware action prediction 是否能更有效降低 stall/transfer/idle？
3. 在专家缺失时，选择性执行 cached alternative 是否能减少 stall，并将质量损失控制在可接受范围内？

## 6. 系统设计草案

### 6.1 总体架构

```text
Trace Collector
  -> Token/Layer Route Event Dataset
  -> Cache/Offload Replay Environment
  -> Cost-aware Action Oracle
  -> Action Predictor
  -> Expert Cache Manager
  -> Simulator metrics / lightweight inference harness / optional vLLM integration
```

### 6.2 Trace collection

当前已有：

- `scripts/moe_affinity/build_qwen_router_trace_signatures.py`
  - 能收集 Qwen1.5-MoE-A2.7B prefill router logits。
  - 当前输出是 request-level sparse signature。

下一步应扩展为 token/layer event trace：

```json
{
  "request_id": "eval-0001",
  "step": 0,
  "phase": "prefill",
  "layer": 12,
  "token_pos": 37,
  "topk_experts": [16, 27, 52, 3],
  "topk_probs": [0.31, 0.24, 0.19, 0.08],
  "selected_experts": [16, 27, 52, 3],
  "router_entropy": 2.13,
  "router_margin": 0.07,
  "domain": "iot",
  "task_family": "argumentation"
}
```

若后续能侵入推理循环，再增加 runtime fields：

```json
{
  "cache_bitmap": "...",
  "resident_experts": [[12, 16], [12, 27]],
  "prefetch_queue": [[13, 5], [13, 9]],
  "transfer_time_ms": 2.4,
  "stall_time_ms": 1.8,
  "gpu_idle_ms": 1.2,
  "pcie_bytes": 17203200
}
```

短期内 runtime fields 可以由 simulator 生成，不必一开始接 vLLM。

### 6.3 Replay environment

先做 CPU-only simulator，目标是比现有 `simulate_moe_affinity_trace_replay.py` 更接近 expert cache/offload：

状态：

- 每层 GPU cache capacity。
- 每层 resident expert set。
- 正在传输的 prefetch queue。
- 当前执行 layer。
- 过去 `N` tokens 的 expert trace。
- 当前 token/router 的 entropy、margin、top-k logits。

动作：

- `keep(layer, expert)`
- `evict(layer, expert)`
- `prefetch(layer, expert)`
- `skip_prefetch`
- `execute_cached_alternative(layer, true_expert, alt_expert)`
- `fetch_on_demand(layer, expert)`

代价：

```text
miss_cost = I(true_expert not resident) * load_time(layer, expert)
transfer_cost = bytes(layer, expert) / bandwidth
stall_cost = max(0, demand_time - ready_time)
idle_cost = gpu_idle_ms
waste_cost = prefetched_but_unused_bytes
quality_penalty = delta_nll_or_task_loss(alt_expert)
```

总目标：

```text
minimize:
  stall_cost
  + alpha * transfer_cost
  + beta * idle_cost
  + gamma * waste_cost
  + lambda * quality_penalty
```

### 6.4 Baselines

必须保留的 baselines：

- Load-on-demand：miss 时再加载。
- LRU：每层独立 LRU expert cache。
- LFU/static-hot：用训练集 expert frequency 固定缓存热门专家。
- MoE-Infinity-style：route window / EAM-nearest activation-aware prefetch。
- Fate-style：shallow-favoring cache + cross-layer gate prefetch proxy。
- DALI-style：workload-aware cache replacement。
- Cache-Conditional：cache prior / cached alternative without learned cost。
- Oracle：
  - Belady-style future-known eviction。
  - route-window oracle prefetch。
  - quality-unconstrained cached alternative 上界。

我们的目标方法：

- Cost-aware action predictor。
- Cost-aware action predictor + selective speculative execution。

### 6.5 Predictor input

最小可落地版：

```text
past N tokens:
  layer-wise expert id histogram
  layer-wise recent bitset
  last token selected experts

current router signal:
  top-k probs/logits
  entropy
  top1-top2 margin
  cumulative probability threshold size

request signal:
  prompt/domain/task_family
  input length / generated length
  prompt embedding or lightweight hash bucket

cache state:
  per-layer cache bitmap
  cache age / frequency / workload score
  prefetch queue occupancy

system signal:
  expert size
  measured transfer bandwidth
  compute overlap window
  current batch size / active tokens
```

### 6.6 Predictor output

不要直接输出完整 routing path。建议输出 action scores：

```text
score(prefetch layer e)
score(keep layer e)
score(evict layer e)
score(skip)
score(spec_alt true_e -> cached_e)
```

为了简化实现，可以第一阶段只做 per-layer top-M candidate：

1. 由 router top-k、recent trace、cache candidates 生成候选 expert。
2. Predictor 对候选动作打分。
3. Cache manager 按预算选 action。

### 6.7 训练目标

推荐两阶段训练：

第一阶段：oracle imitation。

- 在 offline trace 上用未来窗口计算最优/近似最优 action。
- 标签不是 `expert id`，而是 `action` 或 `action utility`。
- 适合用 GBDT/LightGBM、logistic/ridge ranker 或小 MLP，先保证可解释和可复现。

第二阶段：cost regression / contextual bandit。

- 对每个候选 action 估计 `expected_cost_reduction`。
- 训练目标：

```text
target(action) = cost_without_action - cost_with_action
```

- 推理时选择正收益最大的动作，若收益不足则 `skip_prefetch`。

### 6.8 Selective speculative execution

触发条件：

```text
true expert not in cache
and cached alternative exists
and saved_stall_cost > lambda * predicted_quality_penalty
and router_margin <= margin_threshold
and alternative_rank <= rank_threshold
and true_expert not in protected Top-J
```

质量风险估计：

- `router_margin` 越小，替代风险越低。
- `alternative_rank` 越靠前，替代风险越低。
- shallow layers 与关键 tasks 可设置更高 `lambda`。
- 离线校准每个层/专家替换后的 delta NLL、perplexity、MMLU/GSM8K accuracy delta。

论文中要把它写成“选择性推测执行”，而不是无条件替换 expert。

## 7. 当前 workspace 的最小落地路线

### Phase 1：扩展 trace 数据

目标：把现有 request-level signature 扩展为 token/layer event trace。

改造点：

- 新增脚本：`scripts/moe_affinity/build_qwen_router_event_trace.py`
- 复用现有 `build_qwen_router_trace_signatures.py` 的模型加载逻辑。
- 输出 `logs/processed/qwen_router_event_trace_*.jsonl`。

验证：

- 对 24 条 eval 请求输出 event trace。
- 统计每层 entropy、margin、top-k overlap、expert frequency。
- 继续生成旧版 signature，保证与现有 replay 兼容。

### Phase 2：实现 cache/offload simulator

目标：把“同 batch expert locality”升级为“expert cache action cost”。

新增脚本：

- `scripts/moe_affinity/simulate_expert_cache_actions.py`

输入：

- token/layer event trace。
- expert cache capacity。
- expert size。
- PCIe bandwidth。
- per-layer compute time / overlap window。

输出：

- cache hit/miss。
- prefetch hit rate。
- prefetch waste bytes。
- PCIe bytes。
- stall time。
- GPU idle proxy。
- estimated TPOT。

先实现 baselines：

- on-demand
- LRU
- LFU/static-hot
- route-window oracle
- MoE-Infinity-style activation ratio
- shallow-favoring cache

### Phase 3：做 cost-aware action policy

目标：证明 action policy 比 expert-id prediction 或 LRU 更能降低系统代价。

最小实现：

- 不直接训练深度模型。
- 先用 ridge/GBDT/ranker 对候选 action 打分。
- 特征来自 trace、router entropy/margin、cache bitmap、workload score。
- 标签来自 Phase 2 oracle 的 `cost_reduction`。

比较：

- top-k expert prediction -> prefetch
- cost-aware prefetch action
- cost-aware prefetch + eviction
- cost-aware + selective speculative execution

### Phase 4：质量评估

目标：给 speculative alternative 加质量边界。

可做两档：

1. 离线 proxy：
   - 对 router logits 计算 alternative rank/margin。
   - 建立质量风险估计表。

2. 小规模真实 forward：
   - 在 Hugging Face Qwen2-MoE forward 中对少量 token/layer 替换 expert 输出。
   - 统计 delta logits、delta NLL、perplexity。
   - 若侵入模型 forward 太慢，先把 selective speculation 作为 simulator-only optional contribution。

### Phase 5：端到端验证

优先级从高到低：

1. simulator main result：最稳，能覆盖 cache miss、PCIe traffic、stall proxy。
2. lightweight HF harness：控制部分 experts resident/offloaded，测局部 latency。
3. vLLM integration：若时间足够，再接成真正 runtime cache manager。

论文主结果可以先以 simulator + 小规模真实 forward 质量校准为核心，不必把完整 expert cache engine 强行塞进 vLLM。

## 8. 评价指标

中间指标：

- cache hit rate。
- expert miss count。
- prefetch precision/recall。
- eviction regret。
- route prediction accuracy。

主指标：

- TPOT proxy / measured TPOT。
- p95/p99 TPOT 或 per-step latency。
- stall time。
- GPU idle time。
- PCIe traffic bytes。
- prefetch waste bytes。
- quality delta：
  - delta NLL / perplexity。
  - MMLU/GSM8K/自建任务 accuracy。
  - exact match 或 output consistency。

关键 ablation：

- without gate signal。
- without cache state。
- without cost-aware objective，只预测 expert id。
- without selective speculative execution。
- different cache budgets。
- different PCIe bandwidth assumptions。
- different workload domains。

## 9. 与论文题目的对应关系

论文标题关键词可以这样落地：

- “资源受限环境”：GPU 显存不能容纳全部 experts，需要 CPU-GPU offload。
- “混合专家大模型”：Qwen1.5-MoE-A2.7B、OLMoE trace，后续可补 Mixtral/DeepSeek-MoE 的公开或小规模 trace。
- “多维资源协同”：GPU expert cache、CPU pinned memory、PCIe bandwidth、GPU compute overlap、质量风险共同进入 cost model。
- “推理优化方法”：不是训练新 MoE，而是在 inference runtime 做 cache/prefetch/evict/speculate action。

## 10. 预期贡献表述

可以把最终贡献组织成三点：

1. 提出 route-trace-driven expert cache action formulation：把 MoE expert prediction 从 expert-id classification 改写为 cost-aware cache action selection。
2. 设计多信号专家缓存管理器：融合历史 route trace、cross-layer gate signal、router uncertainty 与 cache bitmap，联合决定 prefetch、keep、evict 和 skip。
3. 引入 selective speculative execution：在 miss 代价高且质量风险低时执行 cached alternative expert，减少 offload stall，并通过 router margin / Top-J guard / quality calibration 控制输出偏差。

## 11. 最近一步建议

最应该马上做的是 Phase 1 和 Phase 2：

1. 写 `build_qwen_router_event_trace.py`，把 Qwen router trace 从 request-level signature 升级为 token/layer event trace。
2. 写 `simulate_expert_cache_actions.py`，先不用训练 predictor，跑出 on-demand、LRU、static-hot、route-window oracle、shallow-favoring 的 cache/offload 代价。
3. 如果 oracle 和 route-window policy 相对 LRU 有明显收益，再进入 action predictor；否则说明当前模型/workload 的 expert locality 不足，需要换 workload 或扩大 trace。

这一步和当前 workspace 最贴合，因为已有：

- Qwen1.5-MoE-A2.7B 本地模型。
- eval request manifest。
- request-level router trace 采集脚本。
- OLMoE public trace 采集脚本。
- affinity replay simulator，可作为新 cache simulator 的骨架。


only human edit:
感悟：
1. 需要选择更多的模型，当前模型有可能由于参数量较小，导致瓶颈不在expert cache的加载上面，因此可能需要做多组实验，使用参数量更大的模型来改变瓶颈。
2. 是否可以考虑使用MoE-Lightning论文中提出来的HRM模型来评估推理瓶颈，需要调研确认。
3. 目前方向中的内容太多了，完全就是AI根据这么多篇论文的成果杂糅出来的一个方向。我们需要做减法，根据论文题目：面向资源受限环境的混合专家大模型多维资源协同推理优化方法，找到一个足够精准的切入点去进行接下去的实验，而不是要考虑这么多的指标、这么多的方法

## 12. 方案修订版：后续以本节为准

前面第 5 到第 11 节的方案过宽，后续不再按“cache action + speculative alternative + quality guard”的完整版本推进。新的主线明确收缩为：

> 面向资源受限 MoE 推理的专家权重迁移瓶颈识别与 workload-aware expert cache/prefetch 优化。

也就是说，论文第一版只解决一个问题：

> 当 GPU 显存无法常驻全部 experts，且 CPU-GPU expert weight transfer 成为 decode 阶段瓶颈时，如何利用 route trace 和 expert workload 信息降低 expert miss stall 与 PCIe 迁移开销？

### 12.1 明确保留的内容

保留以下内容作为主线：

1. HRM/roofline-style 瓶颈判定。
   - 使用 MoE-Lightning 中 HRM 的思想判断某个模型和硬件配置下是否真的 expert-loading-bound。
   - 先证明瓶颈存在，再做 cache/prefetch 策略。

2. token/layer 级 expert access trace。
   - 记录每个 token、每层实际选择的 experts。
   - 记录每个 expert 的 routed token 数，作为 workload score。

3. expert cache/offload simulator。
   - 模拟 GPU expert cache capacity。
   - 模拟 CPU -> GPU expert weight transfer。
   - 模拟 on-demand load、prefetch、eviction 和 stall。

4. workload-aware expert cache replacement + prefetch。
   - cache replacement 不只看 LRU/LFU，而是看未来窗口内 expert 是否会被访问、会服务多少 tokens、miss 会造成多少 stall。
   - prefetch 不追求预测所有 expert，只优先预取高 workload、短期会访问且能被 compute overlap 掩盖的 experts。

### 12.2 明确删除或后移的内容

以下内容不进入第一版主实验：

1. 删除 selective speculative execution。
   - 不执行 cached alternative expert。
   - 不改变原模型 routing。
   - 不做 delta NLL、perplexity、MMLU/GSM8K quality delta 作为主指标。
   - 理由：它会把问题从“资源优化”扩展成“改路由但保质量”，复杂度过高。

2. 删除 cache-conditional routing。
   - 不给 cached expert 加 logit prior。
   - 不调整 router top-k。
   - 理由：第一版必须保证模型语义路径不变，避免质量评估吞掉系统实验重点。

3. 后移 learned action predictor。
   - 第一阶段不训练深度 predictor。
   - 先做规则策略和 oracle 上界。
   - 只有当 route-window oracle 相对 LRU 的 stall proxy 降低至少 15%，才进入轻量 predictor。

4. 后移 cross-layer gate predictor。
   - Fate-style cross-layer gate 只作为 related work 和后续增强。
   - 第一版不实现跨层 gate 预测。

5. 删除 CPU/GPU compute assignment。
   - 不做 DALI 式 CPU/GPU 混合计算分配。
   - 本文只研究 expert weights 在 CPU/GPU 间迁移和 GPU expert cache 管理。

6. request scheduling 降级为动机实验。
   - 之前的 request sorting 结果只用于说明：单纯重排请求不足以稳定转化为 latency 收益。
   - 不再把 request scheduling 当主贡献。

### 12.3 固定模型选择

后续模型分工固定如下：

1. Qwen1.5-MoE-A2.7B：工具链 debug 模型。
   - 用于验证 trace 采集脚本、event schema、simulator 输入输出是否正确。
   - 不作为主实验模型。
   - 原因：模型太小，expert loading 未必是主要瓶颈。

2. Mixtral-8x7B：主实验模型。
   - 用于 HRM 瓶颈分析、expert cache/offload simulator 和主要实验表格。
   - 原因：模型规模足够大，experts 无法轻易全部常驻，能够形成更真实的 expert weight transfer 压力。

3. Mixtral-8x22B / DBRX / DeepSeek-MoE：扩展规模验证。
   - 不要求第一阶段真实端到端跑通。
   - 用 HRM 和 simulator 做 scale-out 分析。
   - 作用是证明方法对更大 MoE 更有意义。

4. OLMoE public trace：公开 trace 辅助验证。
   - 用于补充 route locality 和 access pattern 分析。
   - 不替代 Mixtral-8x7B 主实验。

### 12.4 固定实验阶段

#### Phase 0：瓶颈判定，必须先做

新增脚本：

- `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py`

输入：

- 模型层数。
- 每层 expert 数。
- 每 token 激活 expert 数。
- 单个 expert 参数量或权重字节数。
- GPU 可用显存。
- KV cache 预算。
- CPU-GPU 带宽。
- batch size。
- decode active token 数。
- quantization bytes。

输出：

- all-experts-resident 是否可行。
- per-step expert transfer bytes。
- estimated transfer time。
- estimated compute time。
- expert-transfer ratio。
- bottleneck type：`compute-bound`、`memory-capacity-bound`、`expert-transfer-bound`。

判定标准固定为：

```text
如果 all experts 可以在保留 KV cache 后常驻 GPU，则该模型不作为主实验模型。
如果 expert transfer time / decode step time < 30%，则该配置不作为主实验配置。
只有 expert transfer time / decode step time >= 30%，才进入 cache/prefetch 主实验。
```

#### Phase 1：event trace 采集

新增脚本：

- `scripts/moe_affinity/build_moe_router_event_trace.py`

输出 schema 固定为：

```json
{
  "request_id": "eval-0001",
  "phase": "prefill_or_decode",
  "step": 0,
  "layer": 12,
  "token_pos": 37,
  "selected_experts": [16, 27],
  "expert_probs": [0.41, 0.33],
  "num_routed_tokens": 1
}
```

第一版 simulator 必须至少使用：

- `layer`
- `step`
- `selected_experts`
- 每个 expert 在窗口内的 routed token 数

router entropy、margin、prompt domain 暂不进入主实验。

#### Phase 2：expert cache/offload simulator

新增脚本：

- `scripts/moe_affinity/simulate_expert_cache_offload.py`

必须实现的策略：

1. `on_demand`
   - miss 后才加载 expert。

2. `lru`
   - 每层独立 LRU cache。

3. `static_hot`
   - 根据训练 trace 的 expert frequency 固定缓存热门 experts。

4. `belady_oracle`
   - 已知未来访问，用于上界。

5. `route_window_prefetch`
   - 使用未来短窗口或预测窗口预取即将访问的 experts。

6. `workload_aware`
   - 本文方法。

#### Phase 3：本文方法

本文方法固定命名为：

```text
Workload-aware Route-window Expert Cache，简称 WREC。
```

核心打分函数固定为：

```text
score(layer, expert) =
  P_window_use(layer, expert)
  * expected_routed_tokens(layer, expert)
  * miss_stall_ms(layer, expert)
  - transfer_ms(layer, expert)
  - cache_contention_penalty(layer, expert)
```

执行规则：

1. prefetch：
   - 只预取 `score > 0` 的 expert。
   - 如果 prefetch queue 超预算，按 score 从高到低保留。

2. eviction：
   - 当 cache 满时，驱逐未来窗口 score 最低的 resident expert。
   - 若某 expert 在未来窗口内不会被访问，优先驱逐。

3. keep：
   - 高 workload、短期会复用、transfer cost 高的 expert 优先保留。

4. on-demand fallback：
   - 预测失败时仍加载真实 expert，不改变 routing。

### 12.5 固定评价指标

主指标只保留 4 个：

1. stall proxy ms/token。
2. CPU-GPU expert transfer bytes/token。
3. workload-weighted cache miss rate。
4. prefetch waste bytes/token。

辅助指标：

- ordinary cache hit rate。
- evictions per 1k tokens。
- oracle gap。

删除主指标中的：

- quality delta。
- MMLU/GSM8K accuracy。
- output consistency。
- TTFT。
- request scheduling displacement。

理由：第一版不改变 routing，理论上模型输出不变，因此质量指标不是主问题；TTFT 更偏 request scheduling，不是 expert cache/offload 的核心指标。

### 12.6 固定成功标准

主实验只有同时满足以下条件，才说明该方向成立：

```text
1. HRM 显示目标配置 expert transfer time / decode step time >= 30%。
2. Belady oracle 相对 LRU 的 stall proxy 降低 >= 20%。
3. WREC 相对 LRU 的 stall proxy 降低 >= 10%。
4. WREC 的 prefetch waste bytes/token 不超过 route_window_prefetch 的 1.2 倍。
```

如果第 1 条不满足，说明模型或硬件配置不适合作为主实验。

如果第 2 条不满足，说明 trace locality 对 expert cache 没有足够上界收益，应换 workload 或换模型。

如果第 3 条不满足，说明 WREC 方法无效，应先改策略，不进入 predictor。

### 12.7 新的论文贡献表述

贡献改为以下三点：

1. 提出面向 MoE expert offload 的 HRM-guided bottleneck selection 方法，先判定什么模型和资源配置真正受 expert weight transfer 限制。
2. 构建 token/layer 级 expert access trace 与 expert cache/offload replay simulator，用统一代价模型评估 miss stall、PCIe traffic 和 prefetch waste。
3. 提出 WREC：一种 workload-aware route-window expert cache/prefetch 策略，在不改变模型 routing 和输出质量的前提下，降低资源受限 MoE 推理中的 expert loading stall。

### 12.8 下一步只做这三件事

接下来不再扩展 speculative execution、gate predictor 或 request scheduler。只做：

1. 写 `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py`。
2. 写 `scripts/moe_affinity/build_moe_router_event_trace.py`。
3. 写 `scripts/moe_affinity/simulate_expert_cache_offload.py`，实现 LRU、static-hot、Belady、route-window 和 WREC。
