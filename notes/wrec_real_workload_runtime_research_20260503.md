# WREC 真实 Workload 泛化与 Runtime 落地调研

日期：2026-05-03

## 1. 调研结论摘要

当前 WREC 主线的主要薄弱点不是算法本身缺少消融，而是两类外部可信度问题：

1. `workload` 是否过于依赖 `Dolly` 这一类指令数据。
2. `runtime` 是否只停留在 trace-driven simulator，而没有说明真实系统中如何落地。

这两个问题都可以继续推进，但实现难度差异很大。

- 真实 workload 泛化：可以在现有代码框架上稳步扩展，优先级高，收益直接。
- runtime 落地：必须先区分“可观测集成”和“可执行集成”。前者可以较快完成，后者会触及 MoE 权重驻留与异步搬运机制，工程成本远高于调度器 patch。

因此，建议采用如下路线：

1. 先做真实 workload 泛化，补足第二类 prompt 分布与更真实的上下文结构。
2. runtime 先做到 `shadow mode + routed experts capture`，形成“真实系统观测链路”。
3. 若仍有时间，再决定是否进入真正的 `expert weight cache manager` 原型。

## 2. 真实 Workload 泛化应当如何定义

如果只是把现有 `Dolly eval n256` 换成另一批 prompt，这只能算“数据换样本”，还不能充分说明泛化。对当前课题而言，更合理的“真实 workload 泛化”应至少覆盖三个层面：

### 2.1 Prompt 来源更接近真实使用

当前主 workload 是 `databricks/databricks-dolly-15k`。它是公开指令数据集，易于复现，也已被当前脚本完全接入；本地统计显示它只有一个 `train` split，字段包含 `instruction`、`context`、`response` 与 `category`，共约 `15k` 行，适合做受控主实验，但不等价于真实在线对话分布。  
来源：

- [Dolly 15k 数据页](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

更接近真实线上分布的候选是 `lmsys/lmsys-chat-1m`。该数据集明确标注为“大规模真实世界 LLM 对话”，包含 `1,000,000` 条 conversations、`154` 种语言，并带有 redaction 与 moderation 信息；但它有单独许可协议，且包含不安全内容，因此只能作为受控扩展 workload，而不应无条件混入主实验。  
来源：

- [LMSYS-Chat-1M 数据页](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

### 2.2 Context 结构更接近真实 prefill

当前 WREC 主线聚焦 `prefill-stage expert cache scheduling`。因此，真实 workload 的关键不只是 prompt 文本来自哪里，更是 prompt 是否保留了真实多轮上下文结构。

这里建议分两级推进：

- `G1 单轮真实化`：只取真实数据中的首轮用户输入，保证和当前单轮 prompt 设定兼容。
- `G2 多轮真实化`：将对话历史按 chat template 展平成单个 prefill 输入，再采集 router trace。

如果直接跳到 `G2`，实验解释会更强，但 trace 成本和 prompt 长度方差也会显著上升。论文第一版更稳妥的顺序是先完成 `G1`，再视时间进入 `G2`。

### 2.3 流量组织更接近真实 serving

当前主结果本质上是 trace replay，不是线上请求流。即使补了 LMSYS prompt，如果仍按静态文件顺序 replay，也只能说明“prompt 分布泛化”，不能说明“流量模式泛化”。

因此，真实 workload 扩展至少应补一个弱形式的 traffic realism：

- 保留原始长度分布，不做完全等长采样。
- 按 conversation/source 字段做 block sampling，而不是纯随机打散。
- 在 runtime 侧复用已有 `ARRIVAL_INTERVAL / MAX_CONCURRENCY` 档位，对第二 workload 做一轮最小 serving 压力测试。

## 3. 真实 Workload 泛化的推荐实施方案

### 3.1 第一阶段：先完成 Dolly 内部切分，不把它误写成真实 workload

虽然你现在问的是“真实 workload 泛化”，但从论文证据链看，第一步仍然建议先补 `Dolly category split`。原因很简单：本地 `prompt_stats_dolly_20260427.json` 已经显示 eval split 包含 `8` 个类别，且各类样本数足够做小规模比较。这一步成本低，可以先回答“WREC 是否只对某一种任务族有效”。

这一步的定位必须写清楚：

- 它是 `intra-workload robustness`，不是“真实 workload”。
- 它用于筛查哪些类别对 locality 更敏感，帮助后续选择第二 workload 的采样重点。

建议输出：

- `results/wrec/generalization/wrec_dolly_category_split_mixtral8x7b_YYYYMMDD.csv`
- `results/wrec/generalization/wrec_dolly_category_split_mixtral8x7b_YYYYMMDD.md`

### 3.2 第二阶段：引入 LMSYS 单轮真实对话 workload

这是最值得做的“真实 workload”补充。

建议新增脚本：

- `scripts/data_prep/build_wrec_lmsys_manifests.py`

建议处理流程如下：

1. 先确认许可协议是否已接受，并在记录中写明数据使用边界。
2. 只抽取 `redacted=true` 或已完成匿名化的样本。
3. 先限定语言范围，建议从 `English + Chinese` 或仅 `English` 开始，避免多语种 tokenizer 分布把结论搅乱。
4. 先构建 `single-turn prefill` manifest：
   - 每条样本保留 `request_id`
   - `prompt`
   - `source=lmsys-chat-1m`
   - `conversation_id`
   - `language`
   - `turn_index`
   - `target_max_new_tokens`
5. 按当前 WREC 规则切成 `debug/train/eval`，并保证 train/eval 不共享 conversation。

这里最重要的不是数量，而是 split 的干净性。对话数据如果按 message 随机切分，很容易让 train/eval 共享同一 conversation 上下文，导致热度先验泄漏。

推荐初始规模：

- `debug`: `64`
- `train`: `512`
- `eval`: `256`

之所以建议先从 `512/256` 开始，而不是更大，是因为第二 workload 的价值首先在“分布迁移”，不是在“单 workload 大样本重复”。

### 3.3 第三阶段：多轮上下文 prefill workload

当单轮 LMSYS trace 跑通后，再进入更真实的 `multi-turn prefill`。

建议构造方式：

- 以 conversation 为单位，选择某个 assistant 响应前的全部历史作为 prompt。
- 使用和目标 serving 一致的 chat template 展平。
- 限制最大输入长度，避免极端长上下文占满 eval。

建议新增字段：

- `num_turns`
- `flattened_history_tokens`
- `conversation_source_model` 或类似上游模型标识

这一步的研究价值很高，因为它比 Dolly 更接近真实在线 prefill，但代价也更高：

- prompt 长度更长，trace 成本更高；
- route locality 可能因长历史噪声而下降；
- train prior 与 eval 的稳定性会更难维持。

因此，多轮真实 workload 更适合作为扩展实验或附录，而不是当前主线的替代物。

## 4. 真实 Workload 泛化时应控制什么变量

做第二 workload 时，最容易犯的错误是一次性改变太多变量，最后无法判断收益变化来自哪里。建议显式控制以下因素：

### 4.1 长度分布

报告两套结果：

- `natural distribution`：保留真实 workload 自然长度分布。
- `length-matched subset`：从真实 workload 中采样出与 Dolly eval 近似匹配的长度分布。

前者回答“现实里效果如何”，后者回答“变化是否只是因为 prompt 更长或更短”。

### 4.2 策略训练来源

对每个 workload，都应保持：

- `train trace` 只用于 `static_hot` 与 WREC prior；
- `eval trace` 只用于最终评测。

不能把 Dolly train prior 直接拿去证明 LMSYS 正结果；那样测到的是跨 workload 迁移，不是 workload 内泛化。

### 4.3 采样和解码设置

如果 workload 变了，而 `target_max_new_tokens` 和采样策略也完全变了，结果会很难解释。建议在 prefill 主线上保持：

- decode 预算规则与当前主实验一致；
- 先固定 greedy 或 temperature=0 的 trace 构造方式；
- 不要把 sampling 变化和 workload 变化绑在一起。

## 5. 真实 Workload 泛化的最低可交付结果

如果只做一轮最有价值的扩展，建议最低交付为：

1. `Dolly category split`
2. `LMSYS single-turn prefill train/eval trace`
3. 一张跨 workload 主表：
   - `Dolly`
   - `LMSYS single-turn`
   - 每个 workload 上的 `LRU / static-hot / WREC / Belady`

论文中最有力的表述将变成：

> WREC 的收益并不局限于单一指令数据集。在保留相同 simulator、总缓存预算和公平 baseline 的前提下，该方法在公开指令数据和真实对话 workload 上均能稳定优于 LRU。

这会明显降低“只对 toy prompt set 有效”的质疑。

## 6. Runtime 落地需要先澄清的边界

### 6.1 当前已有的 runtime 资产

本地已经有三类与 runtime 直接相关的基础设施：

1. `scripts/runtime/` 下的服务启动、请求发送、GPU 监控和环境检查脚本。
2. `patches/vllm_policy_scheduler_20260424/` 中的 `vLLM 0.19.1` scheduler patch。
3. 先前的服务端记录表明，`length_only / ltr / moe_affinity` 这类策略可以接到 `vllm` 的 waiting queue 层。

此外，官方 vLLM 文档已经支持两类与本课题高度相关的机制：

- `scheduler_cls`：允许替换 scheduler class，但官方明确提示这个接口不是稳定公共接口。
- `enable_return_routed_experts`：允许返回 routed experts，并且官方提供了 `Routed Experts E2E` 示例。

来源：

- [vLLM SchedulerConfig 文档](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)
- [vLLM Routed Experts E2E 示例](https://docs.vllm.ai/en/latest/examples/offline_inference/routed_experts_e2e/)
- [vLLM routed_experts_capturer API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/routed_experts_capturer/)

### 6.2 为什么“接到 vLLM 调度器里”不等于 WREC 落地

这点必须说透。当前 vLLM scheduler patch 处理的是：

- request waiting queue 的排序；
- running batch 的选择；
- KV cache admission / preemption 的一部分调度逻辑。

它不直接管理：

- 哪些 MoE expert weight 常驻 GPU；
- expert miss 时何时从 CPU 搬到 GPU；
- eviction/prefetch 的异步拷贝与 overlap。

换言之：

- `length_only / ltr / moe_affinity` 是请求级调度；
- `WREC` 是 expert weight cache policy。

两者相关，但不是一回事。

因此，runtime 落地必须分层：

1. `observability integration`：在真实 runtime 中观测 routed experts，并在线计算 WREC 决策。
2. `queue-level integration`：用这些观测信号影响请求排队或 batch 选择。
3. `execution-level integration`：真正控制 expert weight residency / transfer / prefetch。

只有第 3 层才是“WREC 完整落地”。

## 7. Runtime 落地的推荐三级路线

## 7.1 R1：Routed experts 采集与影子模式

这是最推荐先做的 runtime 版本。

目标不是改变执行，而是把 vLLM 作为真实运行时，在线采集 routed experts，并让 WREC 以 `shadow mode` 运行：

- 读取每个请求的 routed experts
- 维护一个虚拟 expert cache bitmap
- 输出“如果这是可控 runtime，WREC 此刻会 keep/admit/evict 哪些 experts”

这一层已经足够支撑如下论文表述：

> 我们进一步在真实 serving runtime 中实现了 routed expert capture 与在线决策影子运行，以验证 WREC 所依赖的观测信号能够在实际推理框架中被实时获取。

建议新增脚本：

- `scripts/runtime/vllm_capture_routed_experts.py`
- `scripts/runtime/wrec_shadow_runtime.py`

实施顺序：

1. 先重建 `vllm_moe` 环境，并修正当前脚本中仍指向旧 `miniconda3` 路径的问题。
2. 用官方 `enable_return_routed_experts=True` 跑离线 example，确认当前 MoE 模型和版本支持 routed expert 输出。
3. 将 routed experts 写成与现有 trace 兼容的 JSONL 格式。
4. 复用 WREC decision engine，跑 shadow cache。
5. 将 shadow 决策与离线 replay 结果对齐，验证线上观测链路没有语义漂移。

这一层的优点是：

- 不需要立刻改动模型执行；
- 能直接展示“真实 runtime 里能拿到什么信号”；
- 能把 simulator 和 runtime 之间建立一条可验证映射。

### 7.2 R2：请求队列层 prototype

如果你希望 runtime 部分不只是“观测”，可以继续做一个请求级 prototype，但必须明确它不是完整 WREC，而是“利用 routed expert 信号的 batch/scheduling prototype”。

这条路本地其实已有基础：旧 patch 已经实现过 `moe_affinity` top-K rerank，逻辑是保留一个稳定 base key，再对 waiting queue 的小窗口按与当前 running batch 的 expert signature 相似度重排。

这条路的研究价值是：

- 它能展示 routed expert 信号在真实 scheduler 中可以被消费；
- 它比完整 expert cache manager 的侵入性低得多；
- 它可以作为 WREC runtime 章节中的“相关但不等价”的工程补充。

但要把边界写清楚：

- 它优化的是请求组合；
- 它不是 expert weight cache residency 控制；
- 它不能替代主文的 WREC 结论。

### 7.3 R3：真正的 expert cache manager prototype

这是最难的一层，也是唯一真正对应 WREC 的执行级落地。

如果坚持走这条路，建议不要一开始就试图在 `vllm serve` 的 OpenAI 服务链路里完成全部改造，而应先做一个单机、单模型、单 worker 的最小原型。

更实际的切入方式是：

1. 固定 `tp=1` 或最小 `tp`。
2. 关闭复杂并行与多余优化，优先保证可控性。
3. 在 MoE layer 权重访问路径上加入：
   - resident bitmap
   - miss handler
   - pinned CPU storage
   - H2D async copy
   - completion callback / readiness state
4. 把 WREC 决策器接成：
   - `on route event -> score`
   - `on miss -> admit`
   - `on pressure -> evict`
   - `optional prefetch`

这里最大的现实问题是：stock vLLM 当前公开的稳定接口更偏向 request scheduler 和 KV cache，而不是 expert weight residency。要实现执行级 WREC，通常要深入：

- model executor
- fused MoE layer
- weight storage / offload buffer
- H2D copy orchestration

这已经接近“研究性 runtime fork”，不再是轻量插件。

## 8. 针对你当前论文阶段，runtime 最合适做到哪里

结合现有成果，我建议 runtime 部分的推荐停止点是：

### 推荐停止点 A

完成 `R1 shadow mode` 即可。

这是性价比最高的方案，因为它能回答：

- 真实 runtime 中是否能在线拿到 routed experts？
- WREC 是否能在线产出决策？
- runtime 观测语义是否与离线 trace 一致？

如果完成这一层，你的论文就可以从“纯 simulator”提升到：

> trace-driven 主验证 + 真实 runtime 观测链路验证

### 推荐停止点 B

若时间仍充裕，再补一个 `R2 queue-level prototype`。

这能展示：

- routed expert 信号不仅可采集，还能进入真实 scheduler；
- 但必须明确它是请求调度补充，不是 WREC 主方法本体。

### 不推荐当前直接做的事情

不建议当前直接进入 `R3`，除非你愿意把剩余时间的大头投入 runtime fork。

原因不是它不重要，而是它会把任务从“完成论文加深实验”升级为“实现一个新的 MoE offload serving 原型”。这对当前论文收束风险很高。

## 9. 可以直接执行的落地清单

### 9.1 真实 workload 泛化

1. 新增 `build_wrec_lmsys_manifests.py`
2. 先做 `single-turn` LMSYS manifest
3. 采集 `train/eval` prefill trace
4. 生成 workload stats
5. 用 train trace 重建 `static_hot` 与 WREC prior
6. 在相同 total budgets 下跑 `LRU / static-hot / WREC / Belady`
7. 输出跨 workload 对比表

### 9.2 runtime 观测链路

1. 修复当前环境路径漂移问题：
   - 现有 runner 仍引用 `/root/miniconda3`
   - 当前机器 `conda` 在 `/opt/conda/bin/conda`
2. 重建或确认 `vllm_moe` 环境
3. 用官方 `enable_return_routed_experts` 示例做最小验证
4. 将 routed experts 落盘为现有 trace 兼容格式
5. 接入 WREC shadow decision
6. 对齐 shadow 决策与离线 replay

## 10. 最终建议

如果目标是“让论文看起来不再像 toy，同时控制实现风险”，最推荐的组合是：

1. `LMSYS single-turn prefill workload` 作为真实 workload 泛化。
2. `vLLM routed experts capture + WREC shadow mode` 作为 runtime 落地。

这两个补充刚好对应两个最常见质疑：

- “是不是只对 Dolly 这种离线指令数据成立？”
- “是不是只在你自己的 simulator 里成立？”

而且它们都不会把主线从 `prefill-stage expert cache scheduling` 扩大成一个难以收束的完整 serving 系统工程。
