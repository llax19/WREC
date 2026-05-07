# vLLM WREC Sidecar Hook

## 操作

- 下载 vLLM `v0.19.0` 源码到：
  - `external/vllm-v0.19.0`
- 创建源码分支：
  - `wrec-runtime-hooks`
- 新增可选 WREC sidecar client：
  - `external/vllm-v0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`
- 修改 scheduler，在 request stopped 后读取 `routed_experts` 并可选提交到 WREC sidecar：
  - `external/vllm-v0.19.0/vllm/v1/core/sched/scheduler.py`

## 启用方式

默认关闭。设置环境变量后启用：

```bash
export WREC_SIDECAR_URL=http://127.0.0.1:8765/event
export WREC_SIDECAR_TIMEOUT_SECONDS=0.05
export WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0
```

`WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0` 表示不限制单个 request 导出的事件数。

## 原理

- vLLM 已有 `enable_return_routed_experts=True`，在请求结束时返回 `[seq_len, layer_num, topk]`。
- 新 hook 在 scheduler 中复用该数组，展开为 WREC sidecar event：
  - `request_id`
  - `event_index`
  - `layer`
  - `token_pos`
  - `selected_experts`
- HTTP 提交采用 fail-open：sidecar 错误默认只禁用 bridge，不中断 vLLM 请求。

## 验证

- `py_compile` 通过：
  - `wrec_sidecar_client.py`
  - `scheduler.py`
- 本地 HTTP smoke 通过：
  - 输入 routed experts shape `[1, 2, 2]`
  - 输出 2 条 `/event` 请求
  - payload 与 WREC sidecar contract 一致。

## 结论

- vLLM 源码已具备可选 WREC runtime event export hook。
- 该 hook 仍不控制真实 expert loading/residency。
- 下一步应在 vLLM offloader 或 MoE expert weight path 上接入 WREC sidecar decision，而不是继续只导出事件。

## Residency Manager 初步接入

新增：

- `external/vllm-v0.19.0/vllm/model_executor/offloader/wrec_residency.py`

修改：

- `external/vllm-v0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`
- `external/vllm-v0.19.0/vllm/model_executor/offloader/prefetch.py`

启用方式：

```bash
export WREC_RESIDENCY_MANAGER=1
export WREC_SIDECAR_URL=http://127.0.0.1:8765/event
```

当前行为：

- sidecar response 中 `would_admit=True` 的 expert 会把对应 layer 标记为 desired resident layer。
- `PrefetchOffloader` 原本按 circular order 选择下一层 prefetch。
- 启用 `WREC_RESIDENCY_MANAGER=1` 后，真实 offloader 会优先选择 WREC 标记的 layer 作为下一次 prefetch target。
- sidecar 错误仍 fail-open，不中断 vLLM 请求。

当前限制：

- vLLM 当前 `PrefetchOffloader` 的真实 residency 粒度是整层 module，不是 `(layer, expert)`。
- 因此本阶段是 layer-level residency 介入，用 WREC expert decision 影响真实 H2D prefetch target。
- 最终 expert-slice residency 还需要改造 Mixtral `FusedMoE` 的 `w13_weight` / `w2_weight` expert 维度存储、GPU slot、CPU backing store 和 expert_map。

## FusedMoE Expert-Slice 骨架

新增：

- `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`

修改：

- `external/vllm-v0.19.0/vllm/model_executor/layers/fused_moe/layer.py`
- `external/vllm-v0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`

启用方式：

```bash
export WREC_EXPERT_RESIDENCY=1
```

当前行为：

- 默认关闭，不影响 vLLM 原始行为。
- 启用后，`FusedMoE.forward_native` 会在 kernel forward 前按 `router_logits` 取 top-k experts。
- 第一次触达 layer 时复制 `w13_weight` / `w2_weight` 到 CPU backing。
- sidecar response 中的 `would_admit/would_evict` 会更新 desired `(layer, expert)`。
- 对 WREC desired experts 和当前 selected experts，执行 CPU backing 到当前权重张量同 expert 行的 copy。
- 仅支持标准 softmax top-k routing；遇到 grouped/custom routing 会跳过。

当前限制：

- 该骨架保留完整 GPU expert tensor，不产生显存节省。
- 尚未改造为有限 GPU slot，也未修改 `expert_map`。
- 当前目标是验证 expert 维度 backing、WREC desired expert 状态与真实 H2D copy plumbing，下一步再接入 logical-to-slot 映射。

## FusedMoE Finite Slot 初版

新增行为：

- 设置 `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY=<k>` 且 `0 < k < global_num_experts` 时启用有限 GPU expert slot。
- 第一次触达 layer 时，完整 expert 权重进入 CPU backing。
- GPU 侧 `w13_weight` / `w2_weight` 被替换为 `k` 行 slot tensor。
- `FusedMoE.expert_map` 返回 WREC 维护的 `logical expert -> slot` 映射。
- sidecar `would_evict` 会清理 resident set 和 expert map。

限制：

- 默认仍关闭。
- 只覆盖非 EP、非 EPLB、无 shared experts、无 bias、标准 softmax top-k、unquantized FusedMoE。
- 单次 forward 的 unique top-k experts 数量不能超过 slot capacity；超过时直接报错，避免静默错误结果。
