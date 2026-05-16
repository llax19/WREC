## 2026-05-16 prefetched 空值排查

- 操作：检查 `results/wrec/runtime_qwen_real_table_b_full_n64_alpha1_wrec_only_timing_20260516T051932Z/wrec_residency_stats.jsonl`、`configs/runtime/qwen_real_table_b_full_n64_maxnew16_alpha1_wrec_only_timing.env.sh`、`external/vllm-0.19.0/vllm/model_executor/layers/fused_moe/wrec_expert_residency.py`、`external/vllm-0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py` 与 `scripts/wrec/runtime/runtime_engine.py`，统计 `prefetched=[]` 分布并追踪赋值链路。
- 原理：`wrec_residency_stats.jsonl` 中的 `prefetched` 直接来自 `state.last_prefetched_experts`，只有 `maybe_prefetch_for_attention() -> _onload_experts_async()` 拿到 `predicted_experts` 后才会更新；若 sidecar 没返回 `ranked_experts_by_layer`，且 `ranking_only=1` 导致 `desired_experts_by_layer` 也不积累，则该字段会持续为空列表。
- 结果：
  - 该 run 共 `792` 行，其中 `prefetched=[]` 有 `456` 行，非空有 `336` 行。
  - 空列表全部集中在前 `19` 个 24-layer block（`forward_index=1..456`）；从 `forward_index=457` 开始转为非空。
  - 用户定位的第 `432` 行是 `forward_index=432, layer=23, num_tokens=756`，当时 `H=0, M=60, prefetched=[]`，说明这一层进入 MoE 前没有任何 attention-window 预取候选。
  - 当前配置是 `SIDECAR_MODE=inproc`、`SIDECAR_RANKING_ONLY=1`、`SIDECAR_RANKING_REFRESH_ROUTER_EVENTS=256`。inproc 路径不会像 HTTP 模式那样启动时调用 `/rankings` 做 warm fetch，因此前期只能等 sidecar 在处理足够多 router events 后，随 event response 回传 `ranked_experts_by_layer`。
  - `ranking_only=1` 下 sidecar 只更新 online history，不产生 `would_admit`/`desired_experts`，所以在首次 ranking refresh 之前，runtime 没有任何后备候选来源，`prefetched` 为空是预期行为。
- 结论：这批 `prefetched` 空值的主因不是统计字段丢失，而是当前实现和配置下“前半段没有 ranking payload 可供 runtime 预取”。如果想减少空值，需要优先改 warm-start / refresh 策略，而不是改 stats 序列化。

## 2026-05-16 inproc 初始 rankings 开销估算

- 操作：检查 `scripts/wrec/runtime/runtime_engine.py` 的 `ranked_experts_payload()` / `_should_include_rankings()`、`scripts/wrec/policy.py` 的 `score()`，并结合 `results/wrec/runtime_qwen_real_table_b_full_n64_alpha1_wrec_only_timing_20260516T051932Z/finite_slot_slot60_mbt1280_r1_sidecar_metrics.json` 中的 ranking timing 估算 `inproc` 补一次初始 rankings 的成本。
- 原理：当前 ranking 计算是纯 CPU 打分与排序，不涉及 GPU copy；warm-start 额外做的只是一次 `ENGINE.ranked_experts_payload()`，即对若干 layer 各自遍历 `60` 个 expert 打分并排序。运行期 metrics 给出了总 ranking 时间与 router event 数，可以反推单次 refresh 的毫秒级开销，再按“初始 rankings 覆盖 24 层、常规 refresh 多数只覆盖局部层”估算放大倍数。
- 结果：
  - 该 run 的 `ranking_seconds=3.605s`，`router_events=468096`，`ranking_refresh_router_events=256`。
  - 按 refresh 频率粗估，运行中一共约有 `468096 / 256 = 1828.5` 次 ranking refresh，因此单次 refresh 平均 ranking 成本约 `3.605 / 1828.5 ≈ 1.97ms`。
  - 常规 refresh 代码通常只回传 `当前 layer + 下一 layer`，而初始 `/rankings` 会对全部 `24` 层做 `ranked_experts_payload()`；在“常规 refresh 约 2 层”的口径下，初始全层 warm-start 粗估约是其 `12x`，即大约 `20~30ms` 量级。
  - 即使按更保守口径估到几十毫秒，它也明显小于当前 run 中单层 `sync_onload_ms≈120~160ms` 的 demand copy 开销，更远小于 sidecar 启动时 `load_seconds≈6.8s` 与 `prior_seconds≈5.46s`。
- 结论：`inproc` 补一次初始 rankings` 的时间开销大概率不大，属于一次性的毫秒到几十毫秒级 CPU 开销；如果把它放在 sidecar 初始化阶段，基本不会成为主实验瓶颈。真正需要关注的是它带来的 ranking 质量是否足够好，以及是否会把“冷启动时完全无候选”改成“冷启动时有一份仅基于离线统计的候选”。

## 2026-05-16 inproc warm rankings patch

- 操作：修改 `external/vllm-0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`，把初始化后的 warm-start 统一成 `_warm_initial_rankings()`；`inproc` 直接从本地 engine 构造初始 payload，external 继续走 `/rankings`，两者都通过 `_record_response()` 把 `ranked_experts_by_layer` 喂回 runtime residency manager。
- 原理：`prefetched=[]` 的直接原因是 runtime 在早期 attention window 没拿到 `ranked_experts_by_layer`。因此 warm-start 不能只“取到 rankings”，还必须显式调用 `record_sidecar_response()` 更新 `_ranked_experts_by_layer`；否则 external 模式即便发了 `/rankings` 请求，也不会改变 runtime 可见状态。
- 过程问题：按约定优先使用 `apply_patch`，但本机 `bwrap` namespace 限制导致 `apply_patch` 无法读取待改文件。处理：改用提权的非交互文本替换完成同一逻辑修改，并随后运行 `python3 -m py_compile` 做语法校验。
- 结果：
  - `WrecSidecarClient.__init__()` 现在无论 `inproc` 还是 `external`，都会在初始化完成后执行一次 warm rankings。
  - `inproc` 模式会直接生成 `{"ranked_experts_by_layer": ..., "metrics": ...}` 初始 payload，并立即写入 runtime residency manager。
  - external 模式顺带修复了“拉取初始 `/rankings` 但未调用 `_record_response()`”的问题。
  - 语法校验通过：`python3 -m py_compile external/vllm-0.19.0/vllm/v1/core/sched/wrec_sidecar_client.py`。
- 结论：当前补丁已经把 `inproc` 冷启动阶段“没有任何 warm ranking 注入 runtime”的缺口补上；后续若 `prefetched` 仍然大面积为空，就应继续排查 ranking 质量和 budget/filter，而不是初始化链路。

## 2026-05-16 早期 U=4 解释

- 操作：检查 `wrec_expert_residency.py` 中 `U` 与 `current_experts` 的定义，并统计 `runtime_qwen_real_table_b_full_n64_alpha1_wrec_only_timing_20260516T060205Z/wrec_residency_stats.jsonl` 里 `U` 的分布。
- 原理：`U` 直接等于 `len(current_experts)`；`current_experts` 是把当前 MoE forward 中所有 token 的 top-k routed experts 去重后的集合。若这一整个 forward 的所有 token 都只落在同一组 4 个 experts 上，则 `U=4`，这正好也是 `top_k=4` 模型的最小非零值。
- 结果：新 run 的前 `48` 行 `U=4`，从 `forward_index=49` 开始不再固定为 `4`，后续大多升到 `56~60`。因此 `U=4` 只对应最早两个 24-layer block，不是全局现象，也不是 warm rankings patch 引入的变化。
- 结论：早期 `U=4` 表示那两个 prefill chunk 的 routing 非常同质化，即“虽然 token 数很多，但整层去重后只碰到 4 个不同 experts”。这和 `active_slots=60` 不冲突，后者只是容量上限，不代表本次 forward 一定会用满 60 个 experts。
