# WREC 实验计划：HRM-guided bottleneck selection + workload-aware expert cache/prefetch

日期：2026-04-27

本文件是 `cost_aware_route_trace_expert_cache_manager_20260426.md` 第 12 节的实验展开版。后续实验以本文件为执行计划，不再扩展 speculative execution、cache-conditional routing、cross-layer gate predictor 或 request scheduling。

核心目标固定为：

> 先证明目标模型与硬件配置确实受 expert weight transfer 限制，再验证 WREC 是否能在不改变 routing 的前提下降低 expert miss stall、CPU-GPU expert transfer 和 prefetch waste。

论文级 story 固定为：

```text
HRM bottleneck selector
  -> router event tracer
  -> expert working-set / locality analysis
  -> expert cache/offload replay simulator
  -> baseline + oracle gap
  -> WREC workload-aware cache/prefetch policy
  -> ablation / sensitivity
```

本计划后续不再把“线性打分公式”本身作为主创新。WREC 的可展示工作量必须来自一个可证伪的 trace-driven cache/offload evaluation framework：

1. HRM 先证明目标场景确实 transfer-bound。
2. 真实 router event trace 证明 expert access sequence 可采集、可统计。
3. Belady oracle 证明该 workload/model 存在可利用的 expert-cache 上界收益。
4. 多个 baseline 证明 WREC 不是只超过弱对手。
5. ablation 证明收益来自 workload-aware 与 transfer-aware，而不是偶然调参。

术语固定：

```text
baseline = 可在线实现或常见的普通对比策略，例如 on-demand、LRU、static-hot、route-window。
oracle = 使用未来访问序列的理想上界策略，例如 Belady eviction；oracle 不可部署，只用于判断问题是否值得优化。
```

降级为背景或废线的内容：

- request scheduling、LTR-lite、MoE-affinity request sorting 只作为早期探索背景，不再作为论文主贡献。
- 不做 speculative execution，不改变 router，不声称质量提升。
- 不以完整 vLLM runtime 改造作为第一版完成门槛；真实 runtime 集成最多作为附加工作。
- 不把 `U = alpha * length + beta * affinity` 形式的 EALR 线性拼接作为主方法。

## 0. 固定目录与产物命名

新增脚本固定放在：

- `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py`
- `scripts/moe_affinity/build_moe_router_event_trace.py`
- `scripts/moe_affinity/simulate_expert_cache_offload.py`

最终论文代码整理时应收束为 WREC 专用模块，而不是展示全部历史脚本：

```text
wrec/
  hrm.py
  hardware.py
  trace.py
  locality.py
  simulator.py
  baselines.py
  policies.py
  metrics.py
  experiments/
```

当前 `scripts/moe_affinity/` 是实现验证区；论文定稿前再整理为上述结构。

新增数据与结果固定放在：

- `models/`：外部模型权重，不提交。
- `data/external/`：外部数据集缓存或转换后的中间文件，不提交。
- `data/prompts/`：实验 request manifest。
- `logs/processed/wrec/`：router event trace、trace stats。
- `results/wrec/`：HRM、simulator、policy 结果。
- `figures/wrec/`：论文图表草稿。

所有实验结果文件名必须包含：

```text
model
workload
phase
n_requests
date
```

示例：

```text
results/wrec/mixtral8x7b_dolly_hrm_matrix_20260427.json
logs/processed/wrec/mixtral8x7b_dolly_router_events_n512_20260427.jsonl
results/wrec/mixtral8x7b_dolly_cache_sim_n256_20260427.csv
```

## Phase A：环境、模型、数据准备

### 目标

把后续实验依赖一次性准备好，避免进入 HRM 或 trace 阶段后才发现模型、数据、显存、磁盘或 license 不满足要求。

### 必须准备

1. 本地环境检查。
   - 记录 GPU 型号、GPU 显存、CPU 内存、CUDA 版本、PyTorch 版本、Transformers 版本、vLLM 版本。
   - 记录 CPU-GPU 拓扑和可用 PCIe 带宽。
   - 输出到 `results/wrec/env_inventory_YYYYMMDD.json`。

2. Python 依赖。
   - `torch`
   - `transformers`
   - `accelerate`
   - `datasets`
   - `huggingface_hub`
   - `safetensors`
   - `numpy`
   - `pandas`
   - `tqdm`

3. 模型准备。
   - 已有本地 debug 模型：`/root/workspace/qwen1.5-MoE-A2.7B`
   - 主实验模型：`mistralai/Mixtral-8x7B-Instruct-v0.1`
   - 扩展规模配置：Mixtral-8x22B、DBRX、DeepSeek-MoE 系列只读取 config 或公开参数，不强制下载全量权重。

4. 数据准备。
   - 本地 smoke workload：`data/prompts/debug_requests.jsonl`
   - 本地 eval workload：`data/prompts/eval_requests.jsonl`
   - 主实验 workload：`databricks/databricks-dolly-15k`
   - 可选真实对话 workload：`lmsys/lmsys-chat-1m`

### 下载与转换要求

Mixtral 权重下载到：

```bash
hf download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir models/Mixtral-8x7B-Instruct-v0.1
```

Dolly 数据下载后转换为 request manifest：

```text
data/prompts/wrec_dolly_debug_n64.jsonl
data/prompts/wrec_dolly_train_n1024.jsonl
data/prompts/wrec_dolly_eval_n256.jsonl
```

每条 request 必须包含：

```json
{
  "request_id": "dolly-train-000001",
  "prompt": "...",
  "source": "databricks-dolly-15k",
  "category": "open_qa",
  "split": "train"
}
```

LMSYS-Chat-1M 需要先接受 Hugging Face 数据集协议；未完成授权时不得阻塞主线，直接跳过该 workload。

### 预期产出

- `results/wrec/env_inventory_YYYYMMDD.json`
- `data/prompts/wrec_dolly_debug_n64.jsonl`
- `data/prompts/wrec_dolly_train_n1024.jsonl`
- `data/prompts/wrec_dolly_eval_n256.jsonl`
- `results/wrec/prompt_stats_dolly_YYYYMMDD.json`

### 验收标准

1. `scripts/data_prep/check_requests.py` 能通过所有新增 request manifest。
2. Mixtral config 可读取。
3. Dolly 三个 split 的 request 数分别等于 64、1024、256。
4. 每个 split 都有 input token 长度统计，包括 min、p50、p90、p99、max。

## Phase 0：Pre-HRM 瓶颈筛查与 HRM 建模准备

### 目标

先做一个轻量的 pre-HRM 容量与传输压力筛查，判断目标模型和硬件配置是否值得进入完整 HRM / cache/prefetch 实验。

注意：`scripts/moe_affinity/estimate_moe_hrm_bottleneck.py` 第一版不是 MoE-Lightning 论文中的完整 HRM 实现。它只做 config-based capacity / transfer screen，用于快速排除明显 all-resident 的小模型或明显不可行的配置。

完整 MoE-Lightning HRM 需要进一步建模：

- policy search：`(N, micro-batch size, A_g, F_g, r_w, r_c)`。
- CPU/GPU attention placement。
- CPU/GPU MoE FFN placement。
- GPU-resident weight ratio。
- GPU-resident KV cache ratio。
- CPU/GPU memory constraints。
- `T(M,H,W,P) = max(comm_cpu_to_gpu, T_cpu, T_gpu)` 的 per-layer decode latency。

### 必须准备

1. 编写 `scripts/moe_affinity/estimate_moe_hrm_bottleneck.py` 作为 pre-HRM 筛查工具。
2. 编写 `scripts/moe_affinity/evaluate_moe_lightning_hrm_policy.py` 作为 Phase 0B HRM policy evaluator。
3. 从模型 config 或手写 profile 文件中读取：
   - 层数。
   - 每层 expert 数。
   - 每 token 激活 expert 数。
   - hidden size。
   - intermediate size。
   - dtype bytes。
   - dense 参数估算。
   - expert 参数估算。
4. 从环境 inventory 中读取：
   - GPU 显存。
   - 预留 KV cache 显存。
   - CPU-GPU bandwidth。
   - decode active tokens。
   - batch size。

### Phase 0B：MoE-Lightning-style HRM policy evaluation

`estimate_moe_hrm_bottleneck.py` 只做 pre-HRM 筛查。若模型进入候选范围，必须继续执行 Phase 0B，用 `evaluate_moe_lightning_hrm_policy.py` 评估论文第 4.2 节中的 policy tuple：

```text
P = (N, mu, A_g, F_g, r_w, r_c)
```

其中：

- `N`：batch size。
- `mu`：micro-batch size。
- `A_g`：attention 是否在 GPU 执行。
- `F_g`：MoE FFN 是否在 GPU 执行。
- `r_w`：GPU-resident weight ratio。
- `r_c`：GPU-resident KV cache ratio。

该脚本必须输出：

- memory feasibility。
- `comm_cpu_to_gpu`。
- `T_cpu`。
- `T_gpu`。
- `T = max(comm_cpu_to_gpu, T_cpu, T_gpu)`。
- bottleneck label。
- best throughput policy。

### 实验矩阵

模型：

- Qwen1.5-MoE-A2.7B：debug，不进入主结论。
- Mixtral-8x7B-Instruct-v0.1：主实验。
- Mixtral-8x22B：scale-out。
- DBRX：scale-out。
- DeepSeek-MoE / DeepSeek-V2 系列：scale-out。

精度：

- BF16/FP16。
- INT8。
- INT4。

硬件带宽假设：

- 8 GB/s。
- 16 GB/s。
- 32 GB/s。
- 实测 bandwidth。

expert-cache memory budget：

- 主线使用固定 total expert-cache budget，而不是固定每层相同 resident expert 数。
- budget 用 expert slots 或 bytes 表示，二者必须可互相换算：
  - `total_expert_cache_slots = floor(total_expert_cache_bytes / expert_bytes)`。
  - Mixtral-8x7B 每层 8 experts、32 层，共 256 expert slots。
- 主实验 budget sweep：
  - 12.5% total resident experts。
  - 25% total resident experts。
  - 37.5% total resident experts。
  - 50% total resident experts。
  - 75% total resident experts。
- all experts resident 只作为 sanity / upper-capacity endpoint。

per-layer fixed budget：

- `cache_per_layer = 1/2/4` 保留为 stress test 和 ablation，不作为唯一主表。

### 必须输出

每个配置输出：

```json
{
  "model": "mixtral8x7b",
  "dtype": "bf16",
  "gpu_memory_gb": 24,
  "kv_cache_budget_gb": 8,
  "all_experts_resident_feasible": false,
  "expert_bytes_per_layer": 123456789,
  "resident_expert_budget_per_layer": 2,
  "estimated_transfer_ms_per_step": 12.3,
  "estimated_compute_ms_per_step": 18.5,
  "expert_transfer_ratio": 0.399,
  "bottleneck_type": "expert-transfer-bound"
}
```

### 预期验证

1. Qwen1.5-MoE-A2.7B 很可能不是主瓶颈模型，它的作用是验证工具链。
2. Mixtral-8x7B 在 24GB 或更低有效显存预算下必须出现不能 all-experts-resident 的配置。
3. 只有 `expert_transfer_ratio >= 0.30` 的配置进入 Phase 1 到 Phase 4。

### 预期产出

- `results/wrec/hrm_bottleneck_matrix_YYYYMMDD.csv`
- `results/wrec/hrm_bottleneck_matrix_YYYYMMDD.json`
- `figures/wrec/hrm_transfer_ratio_heatmap_YYYYMMDD.png`
- 一段论文可用结论：哪些模型和配置是 expert-transfer-bound，哪些不是。
- 一段方法边界说明：当前结果是 pre-HRM 筛查，不等价于复现 MoE-Lightning HRM。

### 进入下一阶段门槛

必须满足：

```text
Mixtral-8x7B 至少有一个配置：
all_experts_resident_feasible = false
expert_transfer_ratio >= 0.30
```

如果不满足，停止 WREC 主实验，改用更大模型配置或更低显存预算重新做 Phase 0。

## Phase 1：workload 与 prompt manifest 构建

### 目标

构建稳定、可复现实验输入，避免 trace 与 simulator 结果来自随机 prompt 子集。

### 必须准备

1. 保留本地已有：
   - `data/prompts/debug_requests.jsonl`
   - `data/prompts/eval_requests.jsonl`
2. 从 Dolly 构建三个 split：
   - debug：64 条。
   - train：1024 条。
   - eval：256 条。
3. 对每条 prompt 计算 token 长度。
4. 固定随机种子：
   - `seed = 20260427`

### 实验内容

1. 统计不同 workload 的输入长度分布。
2. 统计 Dolly category 分布。
3. 过滤超长样本：
   - debug/eval 默认 `max_input_tokens = 1024`。
   - main trace 默认 `max_input_tokens = 512`，减少 router trace 成本。
4. 生成统一 schema。

### 预期验证

1. Dolly workload 覆盖 open QA、closed QA、summarization、classification、brainstorming 等类别。
2. train/eval split 不共享 `request_id`。
3. 每个 split 的 p99 token length 小于设定上限。

### 预期产出

- `data/prompts/wrec_dolly_debug_n64.jsonl`
- `data/prompts/wrec_dolly_train_n1024.jsonl`
- `data/prompts/wrec_dolly_eval_n256.jsonl`
- `results/wrec/workload_stats_dolly_YYYYMMDD.json`
- `figures/wrec/workload_input_length_hist_YYYYMMDD.png`

### 进入下一阶段门槛

必须满足：

```text
debug/train/eval 三个 manifest 都通过 check_requests.py
train_n = 1024
eval_n = 256
p99_input_tokens <= 1024
```

## Phase 2：token/layer router event trace 采集

### 目标

把 request-level expert signature 升级为 token/layer 级 expert access trace，为 cache/offload simulator 提供真实访问序列。

### 必须准备

1. 编写 `scripts/moe_affinity/build_moe_router_event_trace.py`。
2. 支持两类模型：
   - Qwen debug：复用已有 `build_qwen_router_trace_signatures.py` 的加载逻辑。
   - Mixtral main：使用 `mistralai/Mixtral-8x7B-Instruct-v0.1`。
3. 支持以下参数：
   - `--model-path`
   - `--request-file`
   - `--output`
   - `--limit`
   - `--max-input-tokens`
   - `--phase prefill`
   - `--dtype`
   - `--device-map`
   - `--max-memory`
   - `--offload-folder`
   - `--trust-remote-code`
4. 采集时不记录 hidden states，不做生成质量评估。
5. Phase 2 必须显式处理 Mixtral offload 风险。不能默认 Mixtral 一定能在 2x3090 Ti 上稳定跑通。

### trace schema

固定输出 JSONL：

```json
{
  "request_id": "dolly-eval-000001",
  "model": "mixtral8x7b",
  "phase": "prefill",
  "step": 0,
  "layer": 12,
  "token_pos": 37,
  "selected_experts": [3, 6],
  "expert_probs": [0.57, 0.31],
  "num_routed_tokens": 1
}
```

第一版必须记录真实 selected experts；router entropy、margin、domain、prompt embedding 不进入主实验。

### 实验顺序

#### Phase 2A：Qwen debug trace

输入：

- `data/prompts/debug_requests.jsonl`

规模：

- 24 条。

目的：

- 验证脚本、schema、统计逻辑。
- 确认不用 Mixtral 权重时，event trace 采集链路本身是正确的。

产出：

- `logs/processed/wrec/qwen_debug_router_events_n24_YYYYMMDD.jsonl`

#### Phase 2B：Mixtral offload feasibility probe，必须先做

输入：

- `data/prompts/wrec_dolly_debug_n64.jsonl`

规模：

- 先跑 1 条。
- 通过后跑 4 条。
- 通过后才允许进入 64 条 debug trace。

目的：

- 验证 Mixtral 全量权重下载是否完整。
- 验证 CUDA-compatible PyTorch + Accelerate 环境是否可用。
- 验证 CPU offload / device map 是否能完成一次 prefill forward。
- 验证能否从 Mixtral router 输出中提取每层每 token 的 selected experts。
- 记录单条 request 的加载时间、峰值 GPU 显存、CPU 内存、trace 采集耗时。

必须使用保守配置：

```text
batch_size = 1
max_input_tokens = 128 或 256
phase = prefill
output_hidden_states = false
output_attentions = false
output_router_logits = true
device_map = auto 或 balanced_low_0
offload_folder = models/offload/mixtral8x7b
```

可接受结果：

```text
1 条 request 能完成 prefill router trace
4 条 request failure rate = 0
单条 request trace time <= 30 分钟
峰值 GPU memory 不触发 OOM
CPU memory 不触发 OOM
selected_experts 长度 = 2
```

如果失败：

```text
不进入 Mixtral 64/512/256 trace。
Phase 2 主线改为 Qwen debug trace + OLMoE public trace + HRM/simulator scale-out。
Mixtral 只保留 HRM 和模型规模动机，不作为真实 route trace 主结果。
```

失败时必须输出：

- `results/wrec/mixtral8x7b_offload_probe_failure_YYYYMMDD.json`

成功时必须输出：

- `results/wrec/mixtral8x7b_offload_probe_success_YYYYMMDD.json`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n4_YYYYMMDD.jsonl`

#### Phase 2C：Mixtral debug trace

只有 Phase 2B 成功后才执行。

输入：

- `data/prompts/wrec_dolly_debug_n64.jsonl`

规模：

- 64 条。

目的：

- 验证 Mixtral router 输出可稳定采集。
- 估算 512/256 条主 trace 的总耗时。

#### Phase 2D：Mixtral train trace

只有 Phase 2C 的 failure rate <= 5% 且耗时可接受时才执行。

输入：

- `data/prompts/wrec_dolly_train_n1024.jsonl`

规模：

- 512 条。

目的：

- 构建 static_hot、workload statistics、WREC 参数。

#### Phase 2E：Mixtral eval trace

只有 Phase 2D 成功后才执行。

输入：

- `data/prompts/wrec_dolly_eval_n256.jsonl`

规模：

- 256 条。

目的：

- 主实验评估。

旧版顺序不再使用：

<!--
1. Qwen debug trace。
   - 输入：`data/prompts/debug_requests.jsonl`
   - 数量：24 条。
   - 目的：验证脚本、schema、统计逻辑。

2. Mixtral debug trace。
   - 输入：`data/prompts/wrec_dolly_debug_n64.jsonl`
   - 数量：64 条。
   - 目的：验证 Mixtral router 输出可采集。

3. Mixtral train trace。
   - 输入：`data/prompts/wrec_dolly_train_n1024.jsonl`
   - 数量：512 条。
   - 目的：构建 static_hot、workload statistics、WREC 参数。

4. Mixtral eval trace。
   - 输入：`data/prompts/wrec_dolly_eval_n256.jsonl`
   - 数量：256 条。
   - 目的：主实验评估。
-->

### 预期验证

1. 每条 trace event 的 `selected_experts` 长度等于模型 top-k。
2. Mixtral 每 token 每 MoE 层应选择 2 个 experts。
3. 同一个 request 的 layer 数与模型 config 一致。
4. trace stats 能显示 expert hotness、reuse distance、window locality。
5. 采集失败的 request 必须进入 failure log，不允许静默跳过。
6. Mixtral offload probe 必须先于 Mixtral debug/train/eval trace 成功。
7. 如果 offload probe 显示单条 trace 过慢，必须降低 Mixtral trace 规模，不能直接跑 512 条。

### 预期产出

- `logs/processed/wrec/qwen_debug_router_events_n24_YYYYMMDD.jsonl`
- `results/wrec/mixtral8x7b_offload_probe_success_YYYYMMDD.json`
- `results/wrec/mixtral8x7b_offload_probe_failure_YYYYMMDD.json`
- `logs/processed/wrec/mixtral8x7b_dolly_probe_router_events_n4_YYYYMMDD.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_debug_router_events_n64_YYYYMMDD.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_YYYYMMDD.jsonl`
- `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_YYYYMMDD.jsonl`
- `results/wrec/router_trace_stats_mixtral8x7b_dolly_YYYYMMDD.json`
- `figures/wrec/expert_frequency_by_layer_YYYYMMDD.png`
- `figures/wrec/reuse_distance_cdf_YYYYMMDD.png`

### 进入下一阶段门槛

必须满足：

```text
Mixtral offload probe 成功
Mixtral eval trace 有效 request 数 >= 240
trace failure rate <= 5%
至少 80% 的 MoE 层存在非均匀 expert hotness
短窗口 locality 指标优于随机访问基线
```

如果 Mixtral offload probe 失败，不进入 Mixtral 主 trace；改用 Qwen/OLMoE trace 继续 simulator debug，并将 Mixtral 真实 trace 标记为当前平台不可行风险。

如果短窗口 locality 不优于随机访问基线，停止 WREC 方法实验，只保留 negative finding。

## Phase 3：expert cache/offload simulator 与 baseline

### 目标

建立可控 replay simulator，先证明 trace locality 在 oracle 和简单 baseline 下有上界收益。

本阶段是论文工作量的核心之一，不是临时脚手架。它要回答：

```text
给定真实 token/layer expert access trace，
在固定 total expert-cache memory budget、expert bytes、CPU-GPU bandwidth 下，
不同 allocation + cache/offload 策略会造成多少 demand miss、transfer bytes、stall proxy 和 prefetch waste？
```

如果 simulator 无法清楚回答这个问题，后续 WREC policy 的结果没有可信度。

### 必须准备

1. 编写 `scripts/moe_affinity/simulate_expert_cache_offload.py`。
2. simulator 输入：
   - router event trace。
   - model profile。
   - expert bytes。
   - total expert-cache budget（slots 或 bytes）。
   - allocation mode。
   - CPU-GPU bandwidth。
   - prefetch queue depth。
   - compute overlap window。
3. simulator 输出：
   - per-event decision log。
   - per-policy summary。
   - per-layer summary。

### 必须实现的策略

策略分为三类：

```text
online baseline：真实系统可实现，不偷看未来。
oracle upper bound：偷看未来，只用于上界，不作为部署策略。
WREC candidate：本文方法，Phase 4 做主实验。
```

必须实现：

1. `on_demand` baseline
   - miss 后同步加载真实 expert。

2. `lru` baseline
   - stress-test 模式支持每层独立 LRU cache。
   - 主实验模式必须支持 fixed total budget 下的 global LRU 或 budget-aware per-layer LRU allocation。

3. `static_hot` baseline
   - 根据 train trace 的 expert frequency 固定缓存热门 experts。
   - 主实验模式按 train trace 全局或 per-layer normalized hotness 在 total budget 内分配 slots。

4. `belady_oracle` oracle
   - 已知未来访问，用于最优 eviction 上界。
   - 不允许在论文中描述为可部署方法。
   - 用于计算 oracle gap：`policy_stall - oracle_stall`。

5. `route_window_prefetch`
   - 若使用真实未来窗口，则归入 oracle-style upper bound。
   - 若使用训练统计或历史窗口预测，则归入 online baseline。
   - 实验表必须标明使用的是哪一种，不能混写。

6. `workload_aware`
   - WREC，Phase 4 做主分析；本阶段先接入接口。
   - 主实验模式必须在 total budget 内决定 per-layer allocation 和 expert keep/admit/evict。

### simulator sanity tests

必须通过以下 sanity：

```text
total_expert_cache_slots = total_num_experts -> miss = 0
total_expert_cache_slots = 0 -> every demanded expert is on-demand miss
belady_oracle stall <= lru stall under same total budget
static_hot 在高 skew trace 上优于 lru under comparable budget
prefetch waste bytes >= 0
transfer bytes >= demand miss bytes
```

### 实验矩阵

主实验 total expert-cache budget：

- 12.5% total resident experts。
- 25% total resident experts。
- 37.5% total resident experts。
- 50% total resident experts。
- 75% total resident experts。

allocation modes：

- uniform per-layer allocation。
- static-hot global allocation。
- workload-aware WREC allocation。
- Belady oracle under the same total budget。

stress-test / ablation cache capacity：

- 1 expert/layer。
- 2 experts/layer。
- 4 experts/layer。

bandwidth：

- 8 GB/s。
- 16 GB/s。
- 32 GB/s。
- measured bandwidth。

window：

- 1 step。
- 4 steps。
- 8 steps。
- 16 steps。

### 预期验证

1. Belady oracle 相对 LRU 的 stall proxy 降低必须达到 20%。
2. route_window_prefetch 必须优于 on_demand；如果它使用未来窗口，则只能解释上界，不得作为公平 online baseline。
3. static_hot 能解释 expert global popularity 是否足够。
4. 若 oracle gap 很小，说明该 workload/model 不适合继续做 WREC。
5. simulator 的主要结论必须同时报告 miss、transfer、stall、waste，不允许只报告 cache hit rate。

### 预期产出

- `results/wrec/cache_sim_baselines_mixtral8x7b_dolly_n256_YYYYMMDD.csv`
- `results/wrec/cache_sim_baselines_mixtral8x7b_dolly_n256_YYYYMMDD.json`
- `logs/processed/wrec/cache_sim_decisions_lru_YYYYMMDD.jsonl`
- `logs/processed/wrec/cache_sim_decisions_belady_YYYYMMDD.jsonl`
- `figures/wrec/baseline_stall_proxy_bar_YYYYMMDD.png`
- `figures/wrec/oracle_gap_by_total_budget_YYYYMMDD.png`

### 进入下一阶段门槛

必须满足：

```text
Belady oracle 相对 LRU 的 stall proxy 降低 >= 20%
至少一个 non-oracle baseline 或 WREC candidate 显示 locality 可转化为 transfer/stall 降低
```

如果不满足，先换 workload 或 cache budget，不进入 WREC 主实验。

## Phase 4：WREC 主方法实验

### 目标

验证 WREC 是否能用 workload-aware 与 transfer-aware 的 cache/offload 决策稳定超过 LRU、static_hot 和可在线实现的 route-window baseline，并尽量缩小到 Belady oracle 的差距。

本阶段不再把一个线性 score 当作全部贡献。线性 score 只是 WREC-H 的第一版实现，真正需要展示的是：

```text
1. 状态：每层 resident expert set、cache age、train/eval hotness、reuse distance。
2. 动作：keep、evict、prefetch、fetch_on_demand。
3. 代价：miss stall、transfer bytes、prefetch waste、cache contention。
4. 约束：total expert-cache memory budget、per-layer allocation、prefetch queue depth、compute overlap window。
5. 证据：相对 baseline 的收益，以及相对 oracle 的剩余 gap。
```

### WREC 分层实现

#### WREC-H：可解释启发式，必须实现

第一版用 workload-aware score 选择预取和驱逐对象：

```text
score(layer, expert) =
  P_window_use(layer, expert)
  * expected_routed_tokens(layer, expert)
  * miss_stall_ms(layer, expert)
  - transfer_ms(layer, expert)
  - cache_contention_penalty(layer, expert)
```

要求：

- `P_window_use` 不能偷看 eval trace 的未来；只能来自 train trace 统计、历史窗口或在线可观测上下文。
- `expected_routed_tokens` 必须体现 workload，而不只是 expert 是否被访问。
- `transfer_ms` 必须来自 HRM / hardware calibration。
- prefetch 与 eviction 要共用同一个 cost model，不能各写一套互相矛盾的规则。

#### WREC-C：约束优化版本，若 WREC-H 太弱则实现

如果 WREC-H 相比 LRU 有收益但距离 oracle 很远，继续实现一个小规模 constrained planner：

```text
在未来预测窗口内选择 prefetch/keep 集合 S：
maximize  sum benefit(layer, expert)
subject to |resident_layer| <= cache_capacity
           prefetch_count <= queue_depth
           transfer_ms(S) <= overlap_budget_ms
```

第一版可用 greedy approximation，不要求 MILP。该版本的价值是把 WREC 从“线性打分排序”推进到“带容量和 overlap 约束的 action selection”。

更新：主实验中的 WREC-C 约束应优先写成 total-budget 形式：

```text
maximize  sum benefit(layer, expert)
subject to sum_layer |resident_layer| <= total_expert_cache_slots
           prefetch_count <= queue_depth
           transfer_ms(S) <= overlap_budget_ms
           optional min/max slots per layer
```

per-layer `|resident_layer| <= cache_capacity` 只用于 stress-test / ablation。

#### WREC-L：learned cost ranker，可选

只有当：

```text
Belady oracle gap 明显存在
WREC-H / WREC-C 已有稳定收益
trace 数量足够训练/验证划分
```

才考虑训练轻量 ranker。否则不训练 predictor，避免把论文变成不稳定的模型训练故事。

#### WREC-D：decode-specific extension，独立分支

WREC-D 不替代当前 prefill 主线，也不计入第一版 WREC 主实验成功标准。它用于回答一个独立问题：

```text
prefill-trained WREC prior 不能直接泛化到 decode 时，
是否可以用 decode train trace 构建 phase-aware expert cache policy？
```

WREC-D 的核心约束：

- 必须使用 decode train trace 构建 prior，不能继续用 prefill train trace 作为主 prior。
- 必须显式区分 `phase=prefill` 与 `phase=decode` 的 replay 结果。
- decode prefetch 默认关闭；只有在 confidence、transfer benefit、waste risk 和 overlap budget 都满足时才开启。
- WREC-D 的结果必须单独成表，不得与 prefill WREC-H/H2/C 主表混写。

WREC-D 第一版优先实现：

```text
WREC-D0: decode-specific step/layer prior
WREC-D2: TinyLFU-style admission guard
WREC-D3: ARC-style ghost history / recency-frequency adaptation
WREC-D4: confidence + overlap constrained decode prefetch
```

WREC-D 的进入条件：

```text
prefill WREC 主线已经完成，且 decode-only eval replay 显示：
1. Belady oracle 明显优于 LRU；
2. prefill-trained WREC-H/H2 不能稳定超过 LRU；
3. decode trace 采集成本可接受。
```

### 必须准备

1. 从 train trace 计算：
   - per-layer expert frequency。
   - per-window expert use probability。
   - expected routed tokens。
   - reuse distance distribution。
   - per-layer working-set size distribution。
   - train/eval hotness shift。
2. 从 HRM 读取：
   - transfer_ms。
   - miss_stall_ms。
   - compute overlap window。
3. 固定 WREC 超参：
   - window sizes：4、8、16。
   - cache contention penalty：从 0、0.1、0.5、1.0 网格选择。
   - prefetch queue depth：1、2、4。
4. 固定 total-budget sweep：
   - 12.5%、25%、37.5%、50%、75% total resident experts。
   - 同时记录换算后的 total slots 和 total bytes。

### 主实验

在 Mixtral eval trace 上比较：

- on_demand
- lru
- static_hot
- belady_oracle
- route_window_prefetch_online
- route_window_prefetch_oracle
- WREC-H
- WREC-C（若实现）

主表按 total expert-cache budget 展开：

```text
policy | total_cache_slots | allocation_mode | stall_ms/token | transfer_bytes/token | workload_weighted_miss | waste_bytes/token | oracle_gap
```

`cache_per_layer = 1/2/4` 的表保留为 stress-test / ablation，不作为唯一主表。

### 预期验证

WREC 必须同时满足：

```text
相对 LRU stall proxy 降低 >= 10%
prefetch waste bytes/token <= route_window_prefetch 的 1.2 倍
oracle gap 小于 LRU oracle gap
在至少 3 个 total expert-cache budget 下不劣于 static_hot allocation 和 online route-window
```

### 预期产出

- `results/wrec/wrec_main_mixtral8x7b_dolly_n256_YYYYMMDD.csv`
- `results/wrec/wrec_main_mixtral8x7b_dolly_n256_YYYYMMDD.json`
- `logs/processed/wrec/wrec_decisions_mixtral8x7b_dolly_n256_YYYYMMDD.jsonl`
- `figures/wrec/wrec_main_stall_vs_cache_YYYYMMDD.png`
- `figures/wrec/wrec_transfer_waste_tradeoff_YYYYMMDD.png`
- `figures/wrec/wrec_oracle_gap_YYYYMMDD.png`

### 进入下一阶段门槛

必须满足第 12 节固定成功标准：

```text
1. HRM 显示目标配置 expert transfer time / decode step time >= 30%。
2. Belady oracle 相对 LRU 的 stall proxy 降低 >= 20%。
3. WREC 相对 LRU 的 stall proxy 降低 >= 10%。
4. WREC 的 prefetch waste bytes/token 不超过 route_window_prefetch 的 1.2 倍。
```

如果第 3 条失败，不训练 predictor，先修改 WREC score 或窗口定义。

如果 WREC-H 只是轻微超过 LRU，但显著弱于 oracle，应优先尝试 WREC-C，而不是马上引入 learned predictor。

如果 WREC-H/WREC-C 都不能稳定超过 baseline，论文结论应改为 negative finding：该 workload/model/total-budget setting 下 trace locality 不足以支撑 WREC，而不是继续堆复杂模块。

## Phase 5：WREC ablation 与敏感性实验

### 目标

证明 WREC 的收益来自 workload-aware、transfer-aware 与 total-budget/overlap-aware action selection，而不是某个偶然窗口、cache budget 或线性权重。

### 2026-05-03 首轮结果状态

已完成 representative total-budget slots `64/96/128` 的 Phase 5 首轮归档：

- `results/wrec/phase5/wrec_ablation_mixtral8x7b_dolly_20260503.csv`
- `results/wrec/phase5/wrec_sensitivity_mixtral8x7b_dolly_20260503.csv`
- `results/wrec/phase5/wrec_phase5_findings_20260503.md`
- `figures/wrec/phase5_20260503/`

当前结论需要收敛为：

- WREC-H2 的主要有效信号是 request-local 与 cross-layer signal。
- recent signal 在当前 total-budget prefill replay 中不是稳定正贡献；`recent_weight=0` 版本在 slots `64/96/128` 上略优于默认 WREC-H2。
- train-window-only 接近 LRU，说明仅靠 train prior 不足以支撑主收益。
- `no_workload_term` 与 `no_transfer_term` 在同大小 expert 的 demand-load simulator 中几乎不改变结果，不能作为强证据；更严格的 transfer-aware ablation 需要 heterogeneous expert size、layer-specific bandwidth 或显式 prefetch/overlap 模型。
- 当前 WREC-H2 没有主动 prefetch 路径，因此 prefetch-only ablation 暂不产出。

补充：`2026-05-03` 已完成 `recent_weight=0` 的五档 total-budget 主表刷新：

- `results/wrec/wrec_total_budget/wrec_total_budget_recent0_mixtral8x7b_dolly_n256_20260503.md`

刷新后结论进一步收敛为：

- `recent_weight=0` 的 request/cross-layer WREC-H2 在 slots `32/64/96/128` 上均优于默认 WREC-H2。
- 在 `192` slots 上默认 WREC-H2 略优于 `recent_weight=0` 版本。
- 若主论文重点放在低中预算 regime，主方法可收束为 request/cross-layer WREC-H2。
- 若必须强调全预算最稳健，则应并列报告 default H2 与 `recent=0` H2，并明确 high-budget 例外点。

### 必须实验

1. 去掉 workload term。

```text
score = P_window_use * miss_stall_ms - transfer_ms - penalty
```

2. 去掉 transfer term。

```text
score = P_window_use * expected_routed_tokens * miss_stall_ms - penalty
```

3. 去掉 prefetch，只做 eviction。

4. 去掉 eviction，只做 prefetch。

5. 替换窗口大小。

```text
window = 1, 4, 8, 16, 32
```

6. 替换 bandwidth。

```text
bandwidth = 8, 16, 32 GB/s, measured
```

7. 替换 total expert-cache budget。

```text
total resident expert fraction = 12.5%, 25%, 37.5%, 50%, 75%
```

`cache_per_layer = 1, 2, 4` 仅作为 stress-test / ablation。

8. WREC-H vs WREC-C。

```text
比较启发式 score 与 constrained planner：
WREC-H: score rank
WREC-C: score + capacity / queue depth / overlap budget constraints
```

9. oracle leakage 检查。

```text
online policy 不允许读取 eval future window。
任何使用 eval future 的策略必须标记为 oracle-style upper bound。
```

### 预期验证

1. 完整 WREC 必须优于去掉 workload term 的版本。
2. 完整 WREC 必须在至少 3 个 total expert-cache budget 下优于 LRU。
3. WREC 对 bandwidth 变化的趋势必须符合 HRM：bandwidth 越低，cache/prefetch 收益越明显。
4. WREC-C 若实现，必须解释相比 WREC-H 的额外收益是否来自容量/overlap 约束，而不是更多超参。
5. 若所有 WREC 变体都只在单个 total expert-cache budget 下有效，不得写成稳定方法收益。

### 预期产出

- `results/wrec/wrec_ablation_mixtral8x7b_dolly_YYYYMMDD.csv`
- `results/wrec/wrec_sensitivity_mixtral8x7b_dolly_YYYYMMDD.csv`
- `figures/wrec/wrec_ablation_bar_YYYYMMDD.png`
- `figures/wrec/wrec_bandwidth_sensitivity_YYYYMMDD.png`
- `figures/wrec/wrec_window_sensitivity_YYYYMMDD.png`

## Phase 6：scale-out 分析

### 目标

证明该方法的意义随模型规模和资源受限程度增强。该阶段不作为 WREC 正确性的唯一证据，只作为论文扩展分析。

### 必须准备

1. 为 Mixtral-8x22B、DBRX、DeepSeek-MoE 系列准备 model profile。
2. 不强制下载全量权重。
3. 使用公开 config、论文参数或模型卡参数估算 expert bytes。
4. 使用 Phase 2 的 Mixtral access pattern 做 normalized access pattern，不声称这是目标模型真实 route。

### 实验内容

1. HRM scale-out。
   - 模型规模增大时 all-experts-resident 是否可行。
   - expert transfer ratio 如何变化。

2. total expert-cache budget scale-out。
   - 在相同 GPU 显存预算下，resident expert fraction 如何下降。

3. transfer pressure scale-out。
   - 在 8、16、32 GB/s 下，理论 stall 上界如何变化。

### 预期验证

1. 更大 MoE 在相同 GPU 显存预算下更容易 expert-transfer-bound。
2. WREC 的目标场景不是小 MoE，而是 expert weights 无法常驻的大 MoE。
3. scale-out 图能支撑论文动机，而不是替代真实 Mixtral-8x7B 实验。

### 预期产出

- `results/wrec/hrm_scaleout_moe_models_YYYYMMDD.csv`
- `figures/wrec/scaleout_expert_transfer_ratio_YYYYMMDD.png`
- `figures/wrec/scaleout_resident_fraction_YYYYMMDD.png`

## Phase 7：轻量真实 profiling 校准

### 目标

用真实硬件测量校准 simulator 的 transfer_ms 和 bandwidth，避免主实验完全依赖手写带宽假设。

### 必须准备

1. 编写或复用一个 CPU pinned memory 到 GPU tensor copy microbenchmark。
2. tensor size 覆盖：
   - 单 expert FP16/BF16 大小。
   - 2 experts。
   - 4 experts。
   - 一个 prefetch batch。
3. 每个 size 重复至少 100 次。

### 实验内容

1. 测量 pageable CPU memory -> GPU。
2. 测量 pinned CPU memory -> GPU。
3. 测量不同 tensor size 的平均和 p95 copy time。
4. 将 measured bandwidth 回填 Phase 0 和 Phase 3。

### 预期验证

1. measured bandwidth 与 HRM 默认 bandwidth 的差距必须记录。
2. 主实验必须至少报告一组 measured bandwidth 结果。
3. 若 measured bandwidth 明显低于假设值，以 measured bandwidth 为准重跑 WREC 主表。

### 预期产出

- `results/wrec/cpu_gpu_transfer_benchmark_YYYYMMDD.csv`
- `results/wrec/cpu_gpu_transfer_benchmark_YYYYMMDD.json`
- `figures/wrec/transfer_time_by_expert_bytes_YYYYMMDD.png`

### 2026-05-04 校准状态

已新增 `workspace/scripts/wrec/benchmark_expert_transfer.py`，用于测量单个 Mixtral expert 大小的 host-to-GPU copy 时间。

输出：

- `workspace/results/wrec/calibration/wrec_expert_transfer_mixtral8x7b_20260504.json`
- `workspace/results/wrec/calibration/wrec_expert_transfer_mixtral8x7b_20260504.md`

当前测量使用 Mixtral-8x7B-Instruct-v0.1 config 推断单 expert 大小：

- expert bytes: `352321536`
- expert size: `336.00 MiB`
- simulator assumed bandwidth: `41.37 GB/s`
- simulator assumed transfer time: `8.516353 ms/expert`

实测结果：

- repeats / warmups: `100 / 5`
- pageable host memory: median `23.455377 ms/expert`, median `15.020929 GB/s`
- pinned host memory: median `8.467264 ms/expert`, median `41.609846 GB/s`

结论：

- pinned memory 的实测 host-to-GPU copy time 与主实验 simulator 使用的 `expert_bytes / 41.37 GB/s` 基本一致。
- pageable memory 明显更慢，因此论文中若讨论 runtime 落地，应明确 expert offload/prefetch 假设依赖 pinned host memory 或等价的高速 host staging。
- 该实验仍是 transfer microbenchmark，不是端到端 serving latency。

## Phase 8：论文图表与结论整理

### 目标

把实验产出整理成论文可用的最小闭环。

### 2026-05-03 收束状态

已新增：

- `results/wrec/phase8/wrec_phase8_paper_ready_summary_20260503.md`
- `results/wrec/phase8/wrec_phase8_figure_manifest_20260503.md`
- `results/wrec/phase8/wrec_phase8_writing_pack_20260503.md`
- `figures/wrec/phase8_20260503/wrec_hrm_transfer_bound_heatmap_mixtral8x7b_20260503.svg`
- `figures/wrec/phase8_20260503/wrec_locality_summary_mixtral8x7b_dolly_train_20260503.svg`
- `results/wrec/hrm_scaleout_moe_models_20260503.md`
- `figures/wrec/scaleout_20260503/scaleout_resident_fraction_20260503.svg`
- `figures/wrec/scaleout_20260503/scaleout_expert_transfer_ratio_20260503.svg`

当前 Phase 8 收束原则：

- 主文主方法使用 `recent_weight=0` 的 request/cross-layer WREC-H2。
- `default H2`、`WREC-C`、decode-only replay 放入 appendix / negative findings。
- `scale-out` 不再作为主文必备项，只保留为 optional extension；若时间不足可不进入第一版主文。
- `Phase 9 runtime integration` 不是当前 paper-ready simulator story 的前置门槛。

### 必须图表

1. HRM bottleneck heatmap。
   - 证明为什么选择 Mixtral-8x7B 和某个显存/带宽配置。

2. expert access pattern 图。
   - expert frequency by layer。
   - reuse distance CDF。
   - short-window locality。

3. baseline comparison。
   - on-demand、LRU、static_hot、online route-window、Belady oracle。
   - 图中必须区分 online baseline 和 oracle upper bound。

4. WREC main result。
   - stall proxy vs total expert-cache budget。
   - transfer/waste tradeoff。
   - oracle gap capture ratio。

5. ablation。
   - 去 workload term、去 transfer term、只 prefetch、只 eviction。
   - WREC-H vs WREC-C（若实现）。

6. scale-out。
   - Mixtral-8x7B、Mixtral-8x22B、DBRX、DeepSeek-MoE 的 transfer pressure 对比。

### 必须结论

论文第一版只允许写以下类型结论：

1. HRM 证明哪些配置确实 expert-transfer-bound。
2. route trace 中存在可利用的短窗口 expert locality。
3. Belady oracle 证明 expert cache 有上界收益。
4. WREC 在不改变 routing 的前提下降低 stall proxy 和 transfer pressure。
5. WREC 捕获了部分 oracle gap，但仍与理想上界存在差距。

不得写：

1. WREC 改善模型质量。
2. WREC 改善 TTFT。
3. WREC 已经是完整 vLLM runtime。
4. WREC 已经解决 speculative execution。
5. WREC 是 learned predictor，除非 Phase 4 明确实现并验证 WREC-L。
6. 使用未来 eval window 的 oracle route-window 是可部署 online 策略。

## Phase 9：vLLM/runtime integration 与生产式验证

### 目标

把 WREC 从 trace-driven simulator 推进到可运行的 serving prototype，验证方法是否能在真实 runtime 中降低 expert loading stall 或端到端 latency proxy。

本阶段不是第一版算法正确性的前置门槛，但如果论文要声称 WREC 具有生产环境实践价值，必须完成至少一个 runtime prototype。没有 Phase 9 时，论文只能声称：

```text
WREC is validated by trace-driven expert cache/offload simulation.
```

不能声称：

```text
WREC is production-ready in vLLM.
```

### 设计原则

- Phase 9 不改变 routing，不改变模型输出质量。
- 只集成在线可用策略：LRU、static-hot、WREC-H2/WREC-C。Belady 和 oracle route-window 不得进入 runtime。
- 先实现最小可测 prototype，再考虑完整 vLLM PR 级别改造。
- runtime 结果必须与 simulator 对齐：同一 workload、同一 expert-cache budget、同一带宽/显存假设。
- 如果 vLLM 当前代码路径没有稳定的 expert weight offload hook，允许先实现 `vLLM-compatible external prototype`，但论文必须明确不是 upstream vLLM integration。

### Phase 9A：vLLM 集成点调研

目标是确定 WREC 应插入到 vLLM 的哪个抽象层，而不是直接改大块代码。

必须检查：

- MoE model runner / model executor 中 expert weight 的加载与调用路径。
- Mixtral / MoE layer 的 expert module 表示方式。
- 是否已有 CPU offload、weight loading、paged weights 或 quantized weight 管理接口可复用。
- scheduler 是否能暴露 request phase、batch composition、prefill/decode 状态。
- 是否能在不改 router 的情况下拦截 selected experts 或 expert module access。

预期产出：

- `results/wrec/runtime/vllm_integration_survey_YYYYMMDD.md`
- 列出至少两个可行集成点：
  - expert weight cache manager hook。
  - model runner 内 MoE layer wrapper。
  - external offload runtime shim。

### Phase 9B：WREC runtime API 抽象

先把 simulator policy 抽象成 runtime 可调用接口：

```python
class ExpertCachePolicy:
    def on_request_start(request_id, metadata): ...
    def on_expert_access(request_id, phase, step, layer, selected_experts): ...
    def choose_resident_set(total_budget): ...
    def admit_or_bypass(layer, expert): ...
    def evict(layer): ...
    def prefetch_candidates(context): ...
```

第一版只要求支持：

- `lru_runtime`
- `static_hot_runtime`
- `wrec_h2_runtime`
- `wrec_c_runtime`，若 Phase 4 已实现。

预期产出：

- `scripts/runtime/wrec_runtime_policy.py`
- `tests` 或 smoke script，使用已采集 trace replay 调用 runtime API，确认结果与 simulator policy 在同一输入下接近。

### 2026-05-04 runtime shadow 状态

已新增 `workspace/scripts/wrec/shadow_runtime.py`，先实现 `JSONL event stream -> WREC shadow policy` 的最小旁路原型。该版本不控制真实 expert loading，而是按 runtime-compatible contract 逐条消费 routed expert refs，在线维护 WREC state 和 shadow resident set，并记录 would-admit / would-bypass / would-evict。

输出：

- `workspace/results/wrec/runtime_shadow/wrec_shadow_mixtral8x7b_dolly_eval_n256_slots64_20260504.json`
- `workspace/results/wrec/runtime_shadow/wrec_shadow_mixtral8x7b_dolly_eval_n256_slots64_20260504.md`
- `workspace/results/wrec/runtime_shadow/wrec_shadow_mixtral8x7b_dolly_eval_n256_slots64_decisions_20260504.jsonl`
- `workspace/results/wrec/runtime_shadow/wrec_shadow_alignment_mixtral8x7b_dolly_eval_n256_slots64_20260504.json`
- `workspace/results/wrec/runtime_shadow/wrec_shadow_alignment_mixtral8x7b_dolly_eval_n256_slots64_20260504.md`

配置：

- event stream: Mixtral Dolly eval n256 router trace
- train prior: Mixtral Dolly train n512 router trace
- policy: WREC-H2 no-recent
- total slots: `64`
- weights: recent `0`, request `1024`, cross-layer `1024`
- bandwidth: `41.37220609315469 GB/s`

对齐结果：

- shadow miss rate: `0.16333409910228316`
- total-budget replay miss rate: `0.16333409910228316`
- shadow stall per input token: `89.01995013339396 ms`
- total-budget replay stall per input token: `89.01995013339396 ms`
- alignment validation overall pass: `True`
- validation metrics with zero delta: expert refs, demand hits, demand misses, hit rate, miss rate, demand transfer bytes, stall ms, stall ms per input token

在线开销：

- shadow loop: `14.658765 s`
- overhead: `16.190 us/expert ref`, `32.380 us/router event`
- decision cost: `91.532 us/miss`

结论：

- WREC policy/state 已能以 event-by-event 方式在线消费 routed expert events，且不读取 future trace。
- shadow mode 与现有 total-budget replay 在同一输入、同一 prior、同一 cache budget 下完全对齐。
- 该结果只能支撑 runtime-compatible shadow prototype，不能声称 WREC 已经控制真实 expert loading 或改善端到端 serving latency。

### Phase 9C：最小 serving prototype

实现一个最小可测 serving prototype，优先级如下：

1. vLLM 内部 hook。
2. 若 vLLM 内部改造过重，则实现 HuggingFace/Transformers prototype，并保持 OpenAI-compatible request manifest 与 WREC runtime API。
3. 若真实 expert weight offload 暂时不可行，则实现 timing-injected prototype：用真实 router/expert access + measured copy benchmark 注入 transfer stall，验证 runtime scheduling overhead 与 policy decision overhead。

prototype 必须测：

- on-demand。
- LRU。
- static-hot。
- WREC-H2 或 WREC-C。

必须记录：

- TTFT。
- prefill latency。
- per-token latency，若包含 decode。
- expert transfer bytes。
- cache hit/miss。
- policy decision overhead。
- peak GPU memory。
- CPU memory。

预期产出：

- `results/wrec/runtime/wrec_runtime_smoke_mixtral8x7b_dolly_n*_YYYYMMDD.json`
- `results/wrec/runtime/wrec_runtime_smoke_mixtral8x7b_dolly_n*_YYYYMMDD.md`

### Phase 9D：production-style benchmark

在 runtime prototype 可用后，跑小规模 serving benchmark：

```text
workload: Dolly eval subset
request_count: 32 / 64 / 128
phase: prefill first, decode optional
concurrency: 1 / 2 / 4
expert_cache_budget: 25% / 50% / 75%
```

主表：

```text
policy | budget | concurrency | prefill_latency_p50 | prefill_latency_p95 | TTFT_p50 | TTFT_p95 | transfer_bytes/request | hit_rate | peak_gpu_mem
```

### Phase 9 成功标准

Phase 9 只有在以下条件满足时，才能支撑“生产式实践”表述：

```text
1. WREC runtime prototype 能完成至少 64 条真实 requests。
2. WREC 相对 LRU 在 prefill latency 或 injected stall 上降低 >= 10%。
3. runtime policy decision overhead <= 总 latency 的 5%。
4. peak GPU memory 不超过设定 expert-cache budget + runtime margin。
5. runtime policy 不读取 future trace，不使用 eval oracle 信息。
```

如果 Phase 9 失败，论文仍可保留 simulator 主结果，但必须把 runtime integration 写成 limitation：

```text
WREC currently validates the cache policy with trace-driven replay;
full vLLM integration remains future work.
```

## 最终成功标准

主实验成立必须同时满足：

```text
1. HRM 显示目标配置 expert transfer time / decode step time >= 30%。
2. Belady oracle 相对 LRU 的 stall proxy 降低 >= 20%。
3. WREC 相对 LRU 的 stall proxy 降低 >= 10%。
4. WREC 的 prefetch waste bytes/token 不超过 route_window_prefetch 的 1.2 倍。
5. WREC 在至少 3 个 total expert-cache budget 下优于 static_hot allocation 和 online route-window。
```

任一条件失败时，结论固定为：

```text
当前模型、workload 或硬件配置下，WREC 主线不成立；需要先换模型、换 workload、换 cache budget 或修改 score，而不是继续扩展 predictor。
```

### runtime 声明边界

若 Phase 9 未完成，最终论文不得使用以下表述：

```text
production-ready
vLLM-integrated
end-to-end serving speedup
real runtime latency improvement
```

只能使用以下表述：

```text
trace-driven expert cache/offload simulation
runtime-informed stall proxy
production-style integration left as future work
```

## 计划变更：fixed total expert-cache budget

日期：2026-05-02

### 变更内容

主实验从固定 `cache_per_layer = 1/2/4` 改为固定 total expert-cache memory budget。

```text
旧主线：每层固定相同 resident expert 数。
新主线：总 expert-cache slots / bytes 固定，由 policy 决定 allocation 和 replacement。
```

`cache_per_layer = 1/2/4` 不删除，但降级为 stress-test / ablation。

### 可复用产物

以下产物不依赖 cache budget，可继续复用：

- Mixtral offload probe。
- Mixtral debug/train/eval router trace。
- router trace stats。
- workload statistics。
- train/eval hotness shift。
- reuse distance、working-set、cross-layer transition。

### 必须重跑产物

以下产物依赖 cache budget / allocation，必须按 total-budget 重跑：

- baseline simulator 主表。
- static-hot allocation。
- LRU / on-demand / route-window。
- Belady oracle。
- WREC-H / WREC-H2 / WREC-C。
- oracle gap 与最终主图。

### 新主实验对照

在同一 total budget 下比较：

- uniform allocation + LRU。
- static-hot allocation。
- online route-window。
- WREC adaptive allocation。
- Belady oracle。

### 设计理由

真实系统约束来自总 GPU 显存余量，而不是每层天然拥有相同 expert slots：

```text
expert_cache_budget = HBM_budget - dense_weights - KV_cache - activations - runtime_margin
```

因此主实验应验证 WREC 是否能在固定总显存预算下做更好的 allocation / admission / eviction / prefetch。

如果只有 oracle 有显著收益、所有 online policy 都收益很弱，结论固定为：

```text
该 trace 存在离线可利用 locality，但当前可在线观测信号不足以稳定转化为 cache/prefetch 收益。
```

## 计划变更：WREC-D decode-specific extension

日期：2026-05-03

### 变更背景

已完成 small decode eval trace 与 decode-only replay：

- `logs/processed/wrec/decode_eval/mixtral8x7b_dolly_eval_decode_only_n64_new16_mem48_20260502.jsonl`
- `results/wrec/decode_replay/wrec_total_budget_decode_only_mixtral8x7b_dolly_eval_n64_new16_20260502.md`

结果显示 prefill train prior 到 decode workload 的迁移不稳定：

- prefill/decode per-layer hotness shift 明显，top-2 overlap mean 为 `0.5625`。
- WREC-H2 在 decode-only slots `64/96/128/192` 下弱于 LRU。
- Belady 仍明显优于在线策略，说明 decode 有 locality，但当前 WREC-H/H2 的在线信号没有抓住它。

### 变更内容

新增 `WREC-D` 作为 decode-specific extension，不并入第一版 prefill 主线。

当前主线固定为：

```text
prefill router trace
  -> prefill expert cache/offload simulator
  -> prefill WREC-H/H2/C
  -> total-budget main table
  -> ablation / sensitivity
```

WREC-D 后续分支为：

```text
decode train trace
  -> decode-specific prior
  -> decode-only cache replay
  -> WREC-D0/D2/D3/D4
  -> decode robustness table
```

### Decode trace 采集要求

WREC-D 需要单独采集 decode train/eval trace。建议从小规模开始：

```text
train decode: n=128, max_new_tokens=16, max_input_tokens=128
eval decode:  n=64,  max_new_tokens=16, max_input_tokens=128
phase: prefill_decode, replay 时筛选 phase=decode
```

如果小规模结果稳定，再扩大到：

```text
train decode: n=256, max_new_tokens=32
eval decode:  n=128, max_new_tokens=32
```

decode trace 文件命名必须包含 `decode`、`n_requests`、`new_tokens`、`date`：

```text
logs/processed/wrec/decode_train/mixtral8x7b_dolly_train_decode_n128_new16_mem48_YYYYMMDD.jsonl
results/wrec/decode_train/router_trace_stats_mixtral8x7b_dolly_train_decode_n128_new16_mem48_YYYYMMDD.json
results/wrec/decode_replay/wrec_d_decode_only_mixtral8x7b_dolly_eval_n64_new16_YYYYMMDD.csv
```

### WREC-D policy 设计

WREC-D0 使用 decode train trace 统计：

```text
P_decode(expert | layer, step_bucket)
P_decode(expert | layer, previous_layer_experts, step_bucket)
P_decode(expert | layer, request_category, step_bucket)
```

step bucket 第一版固定为：

```text
1, 2-4, 5-8, 9-16, 17-32
```

WREC-D score：

```text
D_score =
  a * decode_step_prior
  + b * decode_cross_layer_prior
  + c * online_recency_frequency
  + d * prefill_static_hot_fallback
```

其中 `prefill_static_hot_fallback` 只能作为低置信 fallback，不得主导 decode admission/eviction。

WREC-D2 admission guard：

```text
admit(new) only if
  score(new) > score(victim) + margin
  and recent_frequency(new) >= recent_frequency(victim)
```

否则 bypass，避免 decode 一次性 expert 污染 cache。

WREC-D4 prefetch gate：

```text
expected_saved_stall =
  P_future_use * transfer_stall_ms
  - P_false_positive * waste_cost_ms
  - cache_pollution_penalty

prefetch only if expected_saved_stall > 0
and transfer can overlap with available compute window
and prefetch queue budget is not exceeded
```

### WREC-D 对照实验

decode-only replay 必须比较：

- LRU。
- static-hot from prefill train。
- static-hot from decode train。
- WREC-H2 using prefill prior。
- WREC-D0 using decode prior。
- WREC-D0 + TinyLFU-style admission guard。
- Belady oracle。

主表字段：

```text
policy | phase | train_prior_source | total_cache_slots | stall_ms/decode_token | transfer_bytes/decode_token | waste_bytes/decode_token | oracle_gap
```

### WREC-D 成功标准

WREC-D 只有在以下条件同时满足时，才可作为论文扩展结果：

```text
1. decode Belady oracle 相对 LRU stall proxy 降低 >= 20%。
2. WREC-D 相对 decode LRU stall proxy 降低 >= 10%。
3. WREC-D 在至少 3 个 total expert-cache budget 下优于 static-hot from decode train。
4. 若开启 decode prefetch，waste bytes/decode_token 不超过 constrained prefetch baseline 的 1.2 倍。
```

如果 WREC-D 失败，结论固定为：

```text
decode trace 存在离线 oracle locality，但当前在线可观测信号不足以稳定转化为 decode expert cache收益；
prefill WREC 与 decode WREC 应作为不同 phase-specific policies 处理。
```

### 论文定位

第一版论文主结论仍只覆盖 prefill-stage expert cache scheduling。WREC-D 最多作为：

- robustness stress test。
- negative finding。
- future work。
- 若成功，再作为 optional extension。

## 外部资料来源

- MoE-Lightning / HRM: https://arxiv.org/abs/2411.11217
- Mixtral official release: https://mistral.ai/news/mixtral-of-experts
- Mixtral Hugging Face model card: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
- Dolly 15k dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- LMSYS-Chat-1M dataset: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- DuoServe-MoE: https://arxiv.org/abs/2509.07379
- ProMoE: https://arxiv.org/abs/2410.22134
- Fate: https://arxiv.org/abs/2502.12224
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- TinyLFU: https://arxiv.org/abs/1512.00727
- ARC: https://www.usenix.org/conference/fast-03/presentation/arc-self-tuning-low-overhead-replacement-cache
