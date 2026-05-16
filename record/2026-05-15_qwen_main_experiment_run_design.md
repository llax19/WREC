# 2026-05-15 Qwen 主实验运行设计

## 操作
- 检查了当前 Qwen runtime 主入口 `scripts/runners/run_qwen_real_inference_main.sh`、finite-slot sweep 脚本和 runtime 汇总脚本。
- 查看了 `runtime_long_mbt1280_threshold_m037_smoke_20260515T`、`runtime_ranked_prefetch_threshold_m037_smoke_20260515T` 与 2026-05-14 n16 memory-selected 结果。
- 新增运行方案文档：`notes/qwen_real_runtime_main_run_plan_20260515.md`。

## 原理
- 之前限制 `mbt` 主要是为了绕开每层保留 slots 造成的显存爆炸；当前实现已改为有限 slot 复用，因此正式实验不应再把 `mbt=4/8` 作为主变量。
- 在 `MAX_MODEL_LEN=1280`、`MAX_NUM_SEQS=1` 下，将 `max_num_batched_tokens=1280` 视为不做人为限制。
- smoke 已证明 `slot32 + mbt1280 + threshold=0.37` 可跑通；正式实验先把 GPU memory utilization 提到 0.90，再用 gate run 控制 OOM 风险。
- 主表只保留 `no_wrec/state_only/finite_slot slot32`，slot24/48 作为容量敏感性实验单独跑，避免主结论被参数扫描稀释。

## 结论
- 推荐正式主实验：`GPU_MEMORY_UTILIZATION=0.90`、`mbt=1280`、`SIDECAR_RANKING_SCORE_THRESHOLD=0.37`。
- 推荐先跑 n16 gate，再跑 n64 主实验；如果 n64 3 repeats 过慢，先保留 r1 并补齐 n16 r1-r3。
- 输出目录按 gate、main、slot_sensitivity 三类拆分到 `results/wrec/` 和 `logs/server/`，不混放。
