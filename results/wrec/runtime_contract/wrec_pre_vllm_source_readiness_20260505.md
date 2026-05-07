# WREC Pre-vLLM Source Readiness

## 已完成

- Mixtral `mem48` train prior 与 eval runtime event 合约验证通过。
  - `results/wrec/runtime_contract/mixtral8x7b_mem48_prior_event_contract_20260505.json`
  - `results/wrec/runtime_contract/mixtral8x7b_mem48_prior_event_contract_20260505.md`
- 合约固定为 Mixtral-8x7B：`32` layers、`8` experts/layer、top-k `2`。
- prior 仍使用 `48GiB` CPU-offload 口径的 train trace。
- runtime event stream 仍使用同一 `48GiB` CPU-offload 口径的 eval trace。
- shadow runtime 与离线 replay 对齐已通过。
- HTTP sidecar 与 local shadow 对齐已通过。

## 现有边界

- 当前 sidecar 只返回 would-admit / would-bypass / would-evict。
- 当前验证不控制真实 vLLM expert residency。
- 当前验证不测端到端 serving latency。
- 全驻留或近全驻留 vLLM Mixtral capture 不进入主实验口径。

## 下载 vLLM 源码前的结论

- WREC prior/event schema 已经与 Mixtral `mem48` 主实验 trace 对齐。
- 源码修改阶段不需要再证明 prior 维度；重点应转向 vLLM expert loading / residency hook。
- 下载源码应对齐当前环境 vLLM 版本，再定位 MoE routed expert capture 与 expert weight loading 路径。
