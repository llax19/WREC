# WREC runtime integration steps

## 目标

记录本实验中 WREC runtime 集成的实际步骤。本文只保留 runtime 接入、事件导入、sidecar 对齐验证以及后续接入路径相关内容。

## 1. 建立 runtime shadow mode

### 操作步骤

- 复核最小 runtime shadow mode：
  - `workspace/scripts/wrec/shadow_runtime.py`
  - `workspace/scripts/wrec/validate_shadow_alignment.py`
  - `workspace/results/wrec/runtime_shadow/*`
- 使用 trace event-by-event 输入 shadow runtime。
- 将在线 shadow runtime 的累计状态与离线 total-budget replay 结果对齐。

### 原理

- shadow runtime 不读取 future trace，只按事件顺序消费 routed expert refs。
- 每个 expert ref 触发一次 WREC policy/state 更新。
- 对齐验证用于确认 runtime 形态没有改变 WREC 策略行为。

### 结论

- 最小 runtime shadow mode 可在线消费 routed expert refs。
- shadow runtime 与离线 replay 对齐后，可作为后续 runtime 集成的行为基线。

## 2. 建立 HTTP sidecar 集成边界

### 操作步骤

- 新增 sidecar 与客户端脚本：
  - `workspace/scripts/wrec/runtime_sidecar.py`
  - `workspace/scripts/wrec/send_trace_to_sidecar.py`
  - `workspace/scripts/wrec/validate_sidecar_alignment.py`
- 启动 WREC HTTP sidecar，并用 Mixtral Dolly train trace 构建 WREC prior。
- 使用 `send_trace_to_sidecar.py` 将 Mixtral Dolly eval trace 转换为 runtime event，经 HTTP POST 推入 sidecar。
- 使用本地 `shadow_runtime.py` 对同一 event prefix 生成 local shadow 对照。
- 使用 `validate_sidecar_alignment.py` 比较 sidecar metrics 与 local shadow metrics。

### 接口格式

Endpoint:

```text
POST /event
```

请求格式：

```json
{
  "request_id": "dolly-eval-000000",
  "event_index": 0,
  "layer": 0,
  "token_pos": 0,
  "selected_experts": [1, 4]
}
```

响应包含每个 selected expert 的 shadow decision：

- `shadow_hit`
- `would_admit`
- `would_bypass`
- `would_evict`

### 原理

- 当前环境先采用外部 sidecar 作为 runtime integration boundary。
- runtime 负责提交 routed expert event。
- sidecar 在线维护 WREC recent history、request-local history、cross-layer history 和 shadow resident set。
- sidecar 返回 would-admit / would-evict 等决策，但暂不控制真实 expert loading。

### 输出文件

- `workspace/results/wrec/runtime_sidecar/wrec_runtime_env_check_20260504.json`
- `workspace/results/wrec/runtime_sidecar/wrec_sidecar_smoke_mixtral8x7b_dolly_eval_4096events_slots64_20260504.json`
- `workspace/results/wrec/runtime_sidecar/wrec_sidecar_smoke_mixtral8x7b_dolly_eval_4096events_slots64_20260504.md`
- `workspace/results/wrec/runtime_sidecar/wrec_shadow_prefix8192_mixtral8x7b_dolly_eval_slots64_20260504.json`
- `workspace/results/wrec/runtime_sidecar/wrec_shadow_prefix8192_mixtral8x7b_dolly_eval_slots64_20260504.md`
- `workspace/results/wrec/runtime_sidecar/wrec_sidecar_alignment_mixtral8x7b_dolly_eval_4096events_slots64_20260504.json`
- `workspace/results/wrec/runtime_sidecar/wrec_sidecar_alignment_mixtral8x7b_dolly_eval_4096events_slots64_20260504.md`

### 结论

- HTTP sidecar 可以在线接收 runtime event 并返回 WREC shadow decisions。
- sidecar 与 local shadow 对齐通过，说明 HTTP 集成边界没有改变 WREC 行为。
- 该阶段验证的是 runtime-facing API 与策略状态维护，不是端到端 serving latency 或真实 expert residency 控制。

## 3. 接入 vLLM routed experts 事件源

### 操作步骤

- 确认可用 vLLM 环境：
  - 使用 `conda run -n wrec python ...` 执行 vLLM 脚本。
- 复核 vLLM routed experts 接口：
  - `vllm.LLM(..., enable_return_routed_experts=True)`
  - `CompletionOutput.routed_experts`
  - routed experts 形状约定为 `[seq_len, layer_num, topk]`。
- 新增 vLLM 内部路由捕获脚本：
  - `workspace/scripts/wrec/vllm_routed_experts_capture.py`
- 使用本地 Qwen MoE 模型做最小 smoke：
  - 模型：`workspace/qwen1.5-MoE-A2.7B`
  - `max_model_len=512`
  - `max_tokens=1`
  - 启用 `enable_return_routed_experts=True`
- 将 vLLM 返回的 routed experts 展开为 WREC sidecar event JSONL：
  - `request_id`
  - `event_index`
  - `layer`
  - `token_pos`
  - `selected_experts`

### 原理

- vLLM 0.19.0 已内置 routed experts 捕获路径。
- 请求结束后，vLLM 通过 `CompletionOutput.routed_experts` 返回每个 token、每层的 top-k expert id。
- WREC sidecar 接收的是按 token 和 layer 展开的 event，因此需要将 `[seq_len, layer_num, topk]` 数组转换为 JSONL event stream。

### 输出文件

- `workspace/scripts/wrec/vllm_routed_experts_capture.py`
- `workspace/results/wrec/runtime_vllm/vllm_routed_experts_qwen_smoke_20260505.json`
- `workspace/results/wrec/runtime_vllm/vllm_routed_experts_qwen_smoke_20260505.md`
- `workspace/results/wrec/runtime_vllm/vllm_routed_experts_qwen_events_20260505.jsonl`

### 结论

- vLLM 内部 routed experts 导入已经完成最小闭环：
  - 真实 vLLM MoE forward 产生 routed experts。
  - 脚本读取 `CompletionOutput.routed_experts`。
  - 路由结果被转换成 WREC sidecar event JSONL。
- 当前完成的是“vLLM 内部事件源接入 WREC”，不是 vLLM expert residency/loading 控制。

## 4. 集成困难与处理

- 默认 Python 环境无法导入 `vllm`。
  - 处理：切换到 `wrec` conda 环境运行 vLLM 捕获脚本。
- 现有 WREC sidecar prior 来自 Mixtral，维度为 32 层、8 experts；Qwen MoE 为 24 层、60 experts、top-k 4。
  - 处理：本轮只验证 Qwen 的 routed experts 捕获和 WREC event 导出，不将 Qwen event 强行送入 Mixtral prior。
- Qwen MoE 首次 vLLM 加载较慢。
  - 处理：使用 `max_model_len=512`、`max_tokens=1` 做最小 smoke。

## 5. 下一步

- 加载 Mixtral 运行同一 capture adapter，使事件维度与现有 Mixtral prior 对齐。
- 在事件源与 prior 对齐后，再推进 expert loading control，将 sidecar 的 would-admit / would-evict 接入真实 expert residency manager。

## 6. Mixtral runtime 口径纠偏

### 操作步骤

- 尝试用 vLLM `enable_return_routed_experts=True` 直接加载本地 Mixtral，目标是验证 capture adapter 能导出 32 层、8 experts、top-k 2 的事件。
- 运行中发现该命令按单卡大显存 BF16 路径加载，显存占用接近 `90GB`，与 WREC 主实验的 `mem48` CPU-offload 口径不一致。
- 终止该 vLLM Mixtral 进程，复查 `wrec_experiment_plan_20260427.md`、`4.30record.md`、`5.1record.md`、`5.2record.md` 和主结果文件。

### 原理

- 现有 WREC 主实验 trace 与 prior 来自 `48GiB` 有效显存 cap：
  - `device_map=auto`
  - `max_memory 0=48GiB,cpu=110GiB`
  - Transformers/Accelerate CPU offload
- vLLM 单卡近全驻留 capture 只能说明 routed expert 事件维度可采集，不能证明它处在相同 expert-transfer-bound/offload 条件下。
- runtime 集成必须区分两个问题：事件 schema 对齐，以及真实 expert residency/loading 控制是否符合 `mem48` offload 实验口径。

### 结论

- 本次 `90GB` vLLM Mixtral 尝试不进入主实验结果，也不作为 WREC prior 对齐证据。
- 当前可用且符合主实验口径的 runtime 输入仍应是既有 `mem48` Mixtral trace：
  - `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
  - `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- 下一步若继续做真实 runtime，应优先实现或验证 `48GiB` offload 口径下的事件捕获与 expert loading control，而不是用全驻留 vLLM 事件替代主实验 trace。

## 7. 下载 vLLM 源码前的 prior/event 合约对齐

### 操作步骤

- 新增轻量校验脚本：
  - `workspace/scripts/wrec/validation/validate_runtime_trace_contract.py`
- 使用该脚本检查 Mixtral `mem48` train prior trace 与 eval runtime event trace：
  - `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
  - `logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- 对照对应 stats 文件验证 request 数、input token 数、router event 数、层数和 expert 数。
- 输出 pre-vLLM readiness 摘要：
  - `results/wrec/runtime_contract/mixtral8x7b_mem48_prior_event_contract_20260505.json`
  - `results/wrec/runtime_contract/mixtral8x7b_mem48_prior_event_contract_20260505.md`
  - `results/wrec/runtime_contract/wrec_pre_vllm_source_readiness_20260505.md`

### 原理

- WREC sidecar prior 由 train trace 构建；runtime event stream 必须与该 prior 的模型维度一致。
- Mixtral 主实验口径固定为 `mem48` CPU-offload trace，因此合约验证只使用该口径下的 train/eval trace。
- 校验脚本只读取 JSONL，不加载模型，不产生新的 router trace。

### 结论

- 合约验证通过。
- train prior trace：`512` requests、`29123` input tokens、`931936` router events、`1863872` expert refs。
- eval runtime event trace：`256` requests、`14147` input tokens、`452704` router events、`905408` expert refs。
- 两个 trace 均为 `32` layers、`8` experts/layer、top-k `2`，可用于同一 Mixtral WREC prior/runtime event contract。
- 下载 vLLM 源码前，prior/event schema 对齐工作已完成；后续重点转向 vLLM expert loading / residency hook。

## 8. 下载 vLLM 源码并接入 WREC sidecar 事件 hook

### 操作步骤

- 确认当前环境安装的 vLLM 版本为 `0.19.0`。
- 下载匹配版本源码：
  - `external/vllm-v0.19.0`
  - tag: `v0.19.0`
- 创建修改分支：
  - `wrec-runtime-hooks`
- 新增源码文件：
  - `external/vllm-v0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`
- 修改 scheduler：
  - `external/vllm-v0.19.0/vllm/v1/core/sched/scheduler.py`
- 运行静态与本地 smoke：
  - `py_compile`
  - 本地 HTTP server 接收 `routed_experts` 展开的 WREC event。
- 输出说明：
  - `results/wrec/runtime_contract/vllm_wrec_sidecar_hook_20260505.md`

### 原理

- vLLM 已有 `enable_return_routed_experts=True`，请求结束时 scheduler 能读到 `[seq_len, layer_num, topk]`。
- 本次修改在该读回点增加可选 WREC sidecar client，将 routed experts 展开成 `/event` 请求。
- hook 默认关闭，只有设置 `WREC_SIDECAR_URL` 时启用。
- HTTP client 采用 fail-open：sidecar 出错时禁用 bridge，不中断 vLLM 请求。

### 结论

- vLLM 源码已完成最小 WREC runtime event export hook。
- hook 的输出 schema 与已验证通过的 Mixtral `mem48` prior/event contract 一致。
- 本阶段仍不控制真实 expert loading/residency；下一步应进入 vLLM offloader 或 MoE expert weight path，把 WREC would-admit / would-evict 接入真实 expert residency manager。

## 9. WREC 开始介入真实 residency manager

### 操作步骤

- 复查 vLLM `v0.19.0` 中的真实权重 offload 路径：
  - `vllm/model_executor/offloader/prefetch.py`
  - `vllm/model_executor/offloader/uva.py`
  - `vllm/model_executor/layers/fused_moe/layer.py`
  - `vllm/model_executor/models/mixtral.py`
- 确认当前 `PrefetchOffloader` 的 residency 粒度是整层 module，而不是单个 `(layer, expert)`。
- 新增 WREC residency state：
  - `external/vllm-v0.19.0/vllm/model_executor/offloader/wrec_residency.py`
- 修改 sidecar client，使其解析 `/event` response，并把 `would_admit` / `would_evict` 写入进程内 residency state。
- 修改 `PrefetchOffloader`，在原本 circular prefetch 的位置优先选择 WREC 标记的 layer。
- 运行验证：
  - `py_compile`
  - `wrec_residency_state_smoke_pass`

### 原理

- WREC sidecar 返回的是 expert 粒度决策。
- vLLM 现有 prefetch offloader 控制的是真实 H2D copy 和 GPU static buffer，但粒度是被选中的 transformer layer module。
- 因此本阶段先做 layer-level bridge：
  - expert 被 `would_admit` 时，将该 expert 所在 layer 记为 desired layer；
  - 下一次真实 offloader prefetch 时，优先选择 desired layer；
  - 如果 WREC 记录到 evicted layer，则降低对应 desired layer 计数。
- hook 默认关闭；只有设置 `WREC_RESIDENCY_MANAGER=1` 时才改变真实 offloader prefetch 选择。

### 主要困难

- Mixtral expert weights 在 vLLM 中被打包为 `FusedMoE` 的 `w13_weight` / `w2_weight`，形状按 expert 维度聚合。
- 当前 kernel 默认认为所有本地 experts 的权重行可被直接索引。
- 真正的 expert-slice residency 需要新增 GPU expert slots、CPU backing store、logical expert 到 slot 的映射，以及 miss 时的同步 H2D copy。
- 现有 `PrefetchOffloader` 无法直接表达 `(layer, expert)` eviction，因此本阶段先把 WREC 决策接入真实 layer/module prefetch manager。

### 结论

- WREC 已开始介入 vLLM 的真实 residency manager：它可以改变 `PrefetchOffloader` 的下一次真实 H2D prefetch target。
- 当前介入粒度是 layer-level approximation，不是最终 expert-slice cache。
- 下一步应设计并实现 Mixtral `FusedMoE` expert-slice cache：对 `w13_weight` / `w2_weight` 做按 expert 的 CPU backing 与 GPU slot 映射，再让 WREC sidecar 的 admit/evict 直接控制 slot residency。

## 10. FusedMoE expert-slice residency 骨架

### 操作步骤

- 继续阅读 vLLM Mixtral `FusedMoE` 权重与 forward 路径：
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/layer.py`
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/router/base_router.py`
- 新增 expert-slice residency manager：
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- 在 `FusedMoE.forward_native` 入口接入默认关闭的 hook：
  - `WREC_EXPERT_RESIDENCY=1`
- 修改 sidecar client，使 `/event` response 同时写入 expert-slice manager。
- 运行验证：
  - `py_compile`
  - `wrec_expert_residency_sidecar_smoke_pass`

### 原理

- unquantized Mixtral 的 `w13_weight` / `w2_weight` 第一维是 expert 维。
- 当前骨架在第一次触达 layer 时建立 CPU backing tensor。
- sidecar `would_admit` 会记录 desired `(layer, expert)`；`would_evict` 会降低对应 desired expert 计数。
- 每次 forward 前优先取 WREC desired experts，再按 `router_logits` 补当前 batch top-k expert id，并把未标记 resident 的 expert 行从 CPU backing copy 回当前权重张量的同一 expert 行。
- 该 hook 只允许标准 softmax top-k routing；遇到 grouped/custom routing 会跳过，避免与真实 router 选择不一致。
- 该阶段不改变 tensor shape、不改 `expert_map`、不缩小 GPU 权重，因此不改变 vLLM kernel indexing。

### 主要困难

- 真正省显存的实现不能保留全量 GPU expert tensor。
- 若要进入最终形态，需要把 full expert rows 改为有限 GPU slots，并让 logical expert id 映射到 slot id。
- 这会影响 `expert_map`、kernel 对 `global_num_experts/local_num_experts` 的假设，以及 evict/admit 时的同步边界。

### 结论

- 已建立 expert 粒度 CPU backing、sidecar desired expert 状态与真实 row-level copy 路径。
- 当前结果是 expert-slice residency 的安全骨架，不是最终显存节省版本。
- 下一步应设计有限 GPU slot 与 logical-to-slot 映射，让 resident set 真正约束 GPU expert tensor。

### 逻辑解释

- WREC policy 的 `resident set` 只是策略状态，表示哪些 `(layer, expert)` 应该在 cache 中。
- 如果 GPU 上仍保留完整 `w13_weight` / `w2_weight`，evict 只会改变策略状态，不会减少 GPU expert 权重，也不会阻止 kernel 访问被标记 evicted 的 expert 行。
- 因此要让 cache 驱逐产生真实作用，必须让 GPU 端只持有有限 expert weight slots。
- CPU backing 保存完整 expert 权重，作为 evicted experts 的权威副本。
- logical-to-slot 映射把 router 产生的 logical expert id 转成当前 GPU slot id；miss/admit 时从 CPU backing 拷入空闲或被驱逐 slot。
- 这里“约束 tensor”指约束 GPU 上实际分配并可被 MoE kernel 索引的 expert weight tensor 行数，而不是替代 WREC 的驱逐策略。

## 11. FusedMoE finite GPU expert slot 初版

### 操作步骤

- 扩展 expert-slice residency manager：
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`
- 修改 `FusedMoE.expert_map`，优先返回 WREC finite-slot expert map：
  - `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/layer.py`
- 新增启用参数：
  - `WREC_EXPERT_RESIDENCY=1`
  - `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY=<k>`
- 运行验证：
  - `py_compile`
  - `wrec_finite_slot_smoke_pass`

### 原理

- 当 `0 < WREC_EXPERT_RESIDENCY_SLOT_CAPACITY < global_num_experts` 时，manager 在第一次触达 FusedMoE layer 时：
  - 将完整 `w13_weight` / `w2_weight` 复制到 CPU backing；
  - 在 GPU 上分配有限 expert slot tensor；
  - 用 slot tensor 替换原始完整 expert tensor；
  - 创建 `expert_map[logical_expert] = slot_id`。
- forward 前按当前 batch top-k experts 装载必要 experts；sidecar desired experts 只在剩余 slot 内预取。
- evict 时清除 `logical_to_slot`、`slot_to_logical` 和 `expert_map`。

### 主要困难

- 一个 fused MoE forward 必须同时覆盖当前 batch 的所有 unique top-k experts。
- 如果当前 batch 需要的 unique experts 数量超过 slot capacity，有限 slot 版本无法保持正确性。
- 当前实现遇到该情况会直接报错，不做静默 fallback。
- 本阶段只支持非 EP、非 EPLB、无 shared experts、无 bias、标准 softmax top-k、unquantized FusedMoE。

### 结论

- resident set 现在可以约束 GPU expert tensor 的第一维：启用有限 slot 后，GPU `w13_weight` / `w2_weight` 不再保留完整 expert 行。
- `expert_map` 已接入 vLLM kernel 路径，用于把 logical expert id 映射到有限 GPU slot。
- 该版本仍需真实 vLLM/Mixtral smoke 验证；slot capacity 必须不小于单次 forward 的 unique routed experts 数量。

## 12. vLLM overlay 与默认路径 smoke

### 操作步骤

- 基于已安装 vLLM `0.19.0` 创建 runtime overlay：
  - `workspace/runtime/vllm_patch_overlay`
- 将 WREC 修改过的 Python 文件覆盖到 overlay 中。
- 验证 overlay 加载路径：
  - `vllm.__file__` 指向 overlay；
  - `vllm._C` 可正常导入。
- 默认关闭 WREC 环境变量，运行小模型 smoke：
  - model: `facebook/opt-125m`
  - `max_model_len=256`
  - `gpu_memory_utilization=0.3`

### 原理

- 源码树未构建，直接用 `PYTHONPATH=external/vllm-v0.19.0` 会缺少 `vllm._C`。
- overlay 复用 conda 环境里已安装的编译扩展，只替换本次修改过的 Python 层文件。
- 默认关闭所有 WREC 开关，验证补丁不会破坏 vLLM 基础路径。

### 主要困难

- 环境中没有 `rsync`，改用 `cp -a` 构建 overlay。
- 初次 smoke 中把 `WREC_RESIDENCY_MANAGER` 置为空字符串触发解析错误；已修复为把空字符串按关闭处理。
- `facebook/opt-125m` 首次运行需要下载权重和 torch compile，初始化耗时约数分钟。

### 结论

- overlay 加载成功，`vllm._C` 可导入。
- 默认关闭 WREC 的 `LLM.generate()` smoke 通过，输出 `default_vllm_smoke_pass`。
- 运行末尾出现 vLLM client shutdown 噪声日志，但生成已完成，engine 随后完成 shutdown。

## 13. sidecar 与 expert-residency 非 MoE smoke

### 操作步骤

- 继续使用 runtime overlay：
  - `PYTHONPATH=/root/workspace/runtime/vllm_patch_overlay`
- 使用 `facebook/opt-125m` 做两档非 MoE smoke：
  - 只设置 `WREC_SIDECAR_URL=http://127.0.0.1:8765/event`
  - 只设置 `WREC_EXPERT_RESIDENCY=1`
- 两档均使用：
  - `max_model_len=128`
  - `gpu_memory_utilization=0.1`
  - `max_tokens=2`

### 原理

- OPT 不是 MoE 模型，不会产生 routed expert events。
- 该阶段不验证 Mixtral 事件导出或 expert slot 正确性，只验证打开 WREC Python hook 后不会破坏普通 vLLM dense 模型路径。
- sidecar 未启动时，非 MoE 请求不会触发 event submit，因此不会触发 HTTP failure path。

### 结论

- sidecar bridge 非 MoE smoke 通过，输出 `sidecar_bridge_non_moe_smoke_pass`。
- expert residency 非 MoE smoke 通过，输出 `expert_residency_non_moe_smoke_pass`。
- 两次运行末尾均出现 vLLM client shutdown 噪声日志，但请求已完成。
- 下一步需要进入 MoE/Mixtral 路径，验证 routed expert capture、sidecar response、row-copy skeleton，再验证 finite slot。

## 14. Mixtral CPU-offload sidecar smoke

### 操作步骤

- 启动 WREC sidecar：
  - train trace: `logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
  - total slots: `64`
  - expert bytes: `352321536`
- 使用 runtime overlay 加载本地 Mixtral：
  - model: `models/Mixtral-8x7B-Instruct-v0.1`
  - `max_model_len=64`
  - `max_num_seqs=1`
  - `max_num_batched_tokens=64`
  - `gpu_memory_utilization=0.48`
  - `cpu_offload_gb=70`
  - `enforce_eager=True`
  - `enable_return_routed_experts=True`
- 先尝试 `WREC_EXPERT_RESIDENCY=1` 的 full-shape row-copy skeleton。
- 发现内存问题后，修改实现：
  - 默认 `WREC_EXPERT_RESIDENCY=1` 只记录 sidecar desired expert 状态；
  - 只有显式设置 `WREC_EXPERT_RESIDENCY_ROW_COPY=1` 才复制完整 CPU backing。
- 重跑 Mixtral state-only smoke。

### 原理

- Mixtral 主实验需要 CPU-offload 口径，不能用全量 GPU 常驻路径替代。
- 本次 vLLM 使用 `UVAOffloader`，日志显示 `Total CPU offloaded parameters: 70.28`。
- `gpu_memory_utilization=0.48` 下，运行中 GPU 显存贴近 `47.8GB`，符合 mem48 smoke 口径。
- state-only expert residency 不额外复制完整 expert 权重，只把 sidecar 决策留在进程内状态，避免破坏 CPU-offload 内存预算。

### 主要困难

- 首次 full-shape row-copy skeleton 在加载完成后逐层创建 CPU backing。
- 该做法会在 vLLM 已有 CPU offload 副本之外再次复制 Mixtral expert 权重，导致 EngineCore initialization failed。
- 失败时模型加载本身已成功：
  - CPU offload: `70.28GB`
  - model loading GPU memory: `16.72GiB`
  - 失败发生在 WREC CPU backing 创建到第 14 层附近。
- 修复方式是把 full-shape row-copy 改为显式 opt-in：
  - `WREC_EXPERT_RESIDENCY_ROW_COPY=1`

### 结论

- Mixtral vLLM CPU-offload state-only smoke 通过，输出 `mixtral_sidecar_stateonly_smoke_pass`。
- routed experts capturer 初始化成功，sidecar bridge 启用成功。
- sidecar metrics 查询显示：
  - router events: `66`
  - expert refs: `132`
  - would admit: `100`
  - would evict: `100`
  - final resident: `64`
- 本阶段验证了 `Mixtral vLLM -> routed experts -> WREC sidecar -> decision state` 链路。
- 下一步应在该 state-only 基线上测试 finite slot；不能再默认使用 full-shape row-copy。

## 15. finite expert slot 语义澄清

### 操作步骤

- 复核 `wrec_expert_residency.py`、`layer.py`、`unquantized_fused_moe_method.py` 和 `prefetch.py`。
- 对比 vLLM 原生 module/layer offload slot 与 WREC expert-slice slot。

### 原理

- vLLM 原生 offloader 已有替换，但粒度是 module/layer 参数缓冲，不是单个 `(layer, expert)`。
- WREC 默认 `WREC_EXPERT_RESIDENCY=1` 只记录 sidecar desired state，不限制 GPU expert 权重。
- 只有设置 `0 < WREC_EXPERT_RESIDENCY_SLOT_CAPACITY < global_num_experts` 时，才启用有限 expert slot。
- 有限 expert slot 会缩小单层 `w13_weight` / `w2_weight` 的 GPU 第一维，并用 `expert_map` 做 logical expert 到 slot 的映射。

### 结论

- “GPU 端只持有有限 expert weight slots”不是说 vLLM 原本没有任何替换，而是说原生替换不是 WREC 需要的 expert 粒度替换。
- finite slot 是显式实验开关；它会引入 miss 时 H2D copy 和映射维护开销。
- 若实验目标只是验证 WREC sidecar 决策链路，应使用 state-only；若目标是验证 expert cache 是否真实影响显存和 eviction，才需要 finite slot。

### state-only 与 finite-slot 区别

- state-only 只维护 WREC 策略状态，不改 GPU 上的 expert weight tensor。
- finite-slot 会真的把 GPU expert tensor 缩小为有限 slot，并用 `expert_map` 重写 kernel 看到的 expert 映射。
- state-only 适合验证 routed experts、sidecar、admit/evict 决策链路；finite-slot 适合验证 expert 粒度 cache 是否真的节省显存和触发 miss/onload。
- finite-slot 会带来额外 H2D copy、slot 选择、`expert_map` 更新和 batch unique experts 溢出风险；state-only 基本没有这些推理路径开销。

### finite-slot capacity 约束

- 单次 FusedMoE forward 会把当前 batch 在该层的所有 top-k routing 一起交给 MoE kernel。
- kernel 执行期间，`topk_ids` 中出现过的所有 logical experts 都必须能通过 `expert_map` 找到有效 GPU slot。
- 如果 unique routed experts 数量超过 slot capacity，就必然至少有一个当前 forward 需要的 expert 没有 slot。
- 在同一个 kernel 调用中不能一边计算一边驱逐另一个仍可能被 token 访问的 expert；否则会覆盖仍被本次 forward 使用的权重。
- 可行替代是把 forward 再拆成更小 microbatch 或按 expert 分组多次执行，但这会改变调度、kernel 调用次数和性能模型，当前 finite-slot 骨架没有实现。

### slot capacity 设置原则

- `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY` 是启动前设置的环境变量，当前 manager 初始化时读取一次。
- 当前实现只有 `0 < slot_capacity < global_num_experts` 才启用 finite-slot；若等于或超过 `global_num_experts`，就不再是有限 expert cache。
- capacity 越大，overflow 风险和 miss/onload 次数越低，但 GPU expert 权重占用越接近全驻留，WREC eviction 的显存收益越小。
- capacity 越小，显存节省越明显，但更容易出现单次 forward unique experts 超过 capacity，或产生更多 H2D onload。
- 对 Mixtral 每层 `8` 个 experts、top-k `2`，单次 forward 最多可能触达 `8` 个 unique experts；因此小于 `8` 的 finite-slot 设置都需要通过实际 trace/smoke 验证 overflow 频率。

## 16. 原版 vLLM slot 语义澄清

### 操作步骤

- 复核原版 vLLM 权重创建与 offloader buffer pool：
  - `unquantized_fused_moe_method.py`
  - `prefetch.py`
  - `config/offload.py`

### 原理

- 原版 unquantized `FusedMoE` 创建的是完整 expert 维权重：`w13_weight[num_experts, ...]`、`w2_weight[num_experts, ...]`。
- 原版 vLLM 没有“每层只保留 k 个 expert weight slots”的公开参数。
- 原版 prefetch offloader 的 `slot_capacity` 是静态 GPU buffer pool 的层级预取缓冲数，由 `offload_prefetch_step` 间接决定。
- 调小 `offload_prefetch_step` 可以减少预取缓冲显存，但不改变单个 MoE layer 内的 expert 数量。
- 原版可通过 `cpu_offload_gb`、`offload_group_size`、`offload_num_in_group`、`offload_params` 控制整参数或整层/整模块 offload；粒度不是单个 expert。

### 结论

- 对原版 vLLM 来说，不能通过调小某个 expert slot capacity 来按 expert 粒度节省显存。
- 能调小的是 layer/module 预取缓冲或 offload 范围；这会改变整层/整参数驻留和预取行为，不是 WREC finite expert cache。

## 17. MoE gate 与 CPU offload 关系澄清

### 操作步骤

- 复核 `FusedMoE.forward_native`、unquantized MoE kernel 调用与 prefetch offloader forward hook。
- 区分 router/gate 选择 experts 与 offloader 选择参数搬运目标。

### 原理

- MoE gate/router 输出 `router_logits`、`topk_ids`，表示每个 token 本次计算要路由到哪些 experts。
- 原版 MoE kernel 根据 `topk_ids` 和 `expert_map` 索引 expert 权重完成计算。
- 原版 CPU offload/prefetch 决定的是 layer/module/parameter 级别的权重何时可被访问或预取。
- gate/router 本身不是原版 vLLM 的 expert 权重搬运调度器；它不等价于“只把被选中的 experts 从 CPU transfer 到 GPU”。
- 只有实现 expert-granular backing、logical-to-slot 映射和 miss/onload 路径后，router 选择才能驱动 expert 粒度搬运。

### 结论

- “门控网络选择 expert”是计算语义；“CPU offload transfer 哪些权重”是内存管理语义。
- 原版 vLLM 的 MoE gate 不会自动提供 expert 粒度 weight cache。
- WREC finite-slot 的作用正是把 router 事件与 expert 粒度权重 residency 连接起来。

## 18. full-shape row-copy 语义澄清

### 操作步骤

- 复核前文 `state-only`、`row-copy`、`finite-slot` 三种路径的差异。

### 原理

- `full-shape row-copy` 指每层仍保留完整 GPU expert tensor，同时再创建完整 CPU backing，用于把某个 expert row copy 回同一 expert row。
- 该路径不缩小 GPU tensor，因此不带来 expert 权重显存节省。
- 在 CPU-offload 场景下，它还会在 vLLM 已有 CPU offload 副本之外额外复制完整 expert 权重，容易破坏内存预算。

### 结论

- “不能再默认使用 full-shape row-copy”表示默认 WREC runtime 只能做 state-only 记录；只有显式设置 `WREC_EXPERT_RESIDENCY_ROW_COPY=1` 时才启用旧 row-copy 骨架。
