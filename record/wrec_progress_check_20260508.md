# WREC progress check 2026-05-08

## 操作

- 检查仓库结构、git 状态、`record/`、`results/wrec/`、`figures/wrec/`、`scripts/wrec/`、`scripts/moe_affinity/` 和 `patches/`。
- 读取主实验指南、WREC 论文收束摘要、主结果、消融、runtime 集成记录、vLLM hook 记录和早期 TP=2 调度结果。

## 原理

- 进度判断按证据链分类：trace 可用性、离线 replay 主结果、消融与图表、runtime 接入、负结果和剩余工程缺口。
- 只把有明确结果文件或记录文件支撑的内容标记为完成；对只通过 smoke 或尚未端到端验证的内容保留边界。

## 结论

- WREC 当前主线已经从早期在线请求调度收束为 Mixtral prefill expert-cache scheduling。
- 主实验离线 replay、HRM 动机、trace gate、fair baselines、WREC-H2 主表、消融、sensitivity、论文图表和写作包基本完成。
- runtime 侧已完成 shadow、HTTP sidecar、vLLM routed experts 事件导出、state-only Mixtral CPU-offload smoke 和 vLLM patch 打包。
- 尚未完成端到端 serving latency 证明和真实 expert finite-slot Mixtral 验证；decode 与 WREC-C 当前应作为负结果或附录边界处理。
