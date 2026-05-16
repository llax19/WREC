# Qwen WREC Real Runtime Main Run Plan 2026-05-15

## Decision

- `mbt` no longer作为主变量。正式运行统一设为 `1280`，等同于在 `MAX_MODEL_LEN=1280` 和 `MAX_NUM_SEQS=1` 下不再人为限制 batch token。
- `GPU_MEMORY_UTILIZATION` 提到 `0.90`。先做一个 90% gate run，再跑正式矩阵。
- 主实验优先使用 long prefill workload：`wrec_dolly_prefill_eval_long_n64_maxnew16_20260514.jsonl`。
- 如果时间或稳定性不足，先用 `wrec_dolly_prefill_eval_long_n16_maxnew1_20260514.jsonl` 得到主表候选，再把 n64 作为最终确认。
- 正式方法口径统一为 `WREC 主线 selective prefetch`，简称 `WREC 主线`。当前 runner 里仍沿用遗留 case 名 `finite_slot`，当前实现也通过 `SIDECAR_RANKING_ONLY=1` 打开轻量 sidecar 路径，但这些都只视为实现细节，不作为方法名写入主表。
- 主表暂时只保留已确认可跑通且口径清楚的必要对照：`no_wrec` 与当前 `WREC 主线`。slot 预算敏感性单独作为补充实验，不和主表混在一个结论里。

## Gate run

目的：验证 `GPU_MEMORY_UTILIZATION=0.90` 加 `mbt=1280` 在当前机器上不会 OOM，并确认 `WREC 主线` 当前实现参数仍工作。

```bash
cd /root/WREC
RUN_ID=20260515T_gmem90_mbt1280_gate
RESULT_ROOT=/root/WREC/results/wrec/runtime_qwen_gate_${RUN_ID}
LOG_ROOT=/root/WREC/logs/server/qwen_gate_${RUN_ID}

GPU_MEMORY_UTILIZATION=0.90 \
MAX_MODEL_LEN=1280 \
MAX_NUM_SEQS=1 \
MAX_CONCURRENCY=1 \
REQUEST_FILE=/root/WREC/data/prompts/wrec_dolly_prefill_eval_long_n16_maxnew1_20260514.jsonl \
METHODS="finite_slot" \
FINITE_SLOT_CASES="32:1280" \
REPEATS=1 \
SIDECAR_MODE=inproc \
SIDECAR_RANKING_ONLY=1 \
SIDECAR_RANKING_SCORE_THRESHOLD=-0.37 \
RESULT_ROOT="$RESULT_ROOT" \
LOG_ROOT="$LOG_ROOT" \
EXPERIMENT_TAG="qwen_gate_${RUN_ID}" \
WREC_EXPERT_RESIDENCY_STATS_PATH="$RESULT_ROOT/wrec_residency_stats.jsonl" \
bash scripts/runners/run_qwen_real_inference_main.sh
```

通过标准：
- `run_manifest.csv` 中 case 为 `success`。
- `serving_metrics_summary.csv` 中 `num_requests=16`。
- `gpu0_peak_memory_utilization` 未明显超过 0.90，server log 无 OOM。
- `wrec_residency_stats.jsonl` 中后段有 `prefetch_hit_rate=1.0` 或稳定上升的记录。

## Main formal run

主表建议先跑 3 repeats。n64 每个 case 会较慢，如果第一轮耗时过长，可先把 `REPEATS=1` 跑完确认趋势，再补 r2/r3。

注意：下面命令里的 `finite_slot` / `FINITE_SLOT_CASES` 只是当前 runner 的遗留参数名；这里实际指向的是当前 `WREC 主线 selective prefetch`，而不是旧的 per-layer reserved-slot 失败方案。`SIDECAR_RANKING_ONLY=1` 仅表示当前实现采用轻量 sidecar 路径，不作为方法命名的一部分。

```bash
cd /root/WREC
RUN_ID=20260515T_formal_n64_gmem90_mbt1280_thr037
RESULT_ROOT=/root/WREC/results/wrec/runtime_qwen_main_${RUN_ID}
LOG_ROOT=/root/WREC/logs/server/qwen_main_${RUN_ID}

GPU_MEMORY_UTILIZATION=0.90 \
MAX_MODEL_LEN=1280 \
MAX_NUM_SEQS=1 \
MAX_CONCURRENCY=1 \
REQUEST_FILE=/root/WREC/data/prompts/wrec_dolly_prefill_eval_long_n64_maxnew16_20260514.jsonl \
METHODS="no_wrec finite_slot" \
NO_WREC_MBTS="1280" \
FINITE_SLOT_CASES="32:1280" \
REPEATS=3 \
SIDECAR_MODE=inproc \
SIDECAR_RANKING_ONLY=1 \
SIDECAR_RANKING_SCORE_THRESHOLD=-0.37 \
RESULT_ROOT="$RESULT_ROOT" \
LOG_ROOT="$LOG_ROOT" \
EXPERIMENT_TAG="qwen_main_${RUN_ID}" \
WREC_EXPERT_RESIDENCY_STATS_PATH="$RESULT_ROOT/wrec_residency_stats.jsonl" \
bash scripts/runners/run_qwen_real_inference_main.sh
```

主指标：
- `input_tokens_per_s`
- `total_tokens_per_s`
- `output_tokens_per_s`
- `p95_ttft_ms`
- `p95_e2e_ms`
- `p95_tpot_ms`
- `gpu0_peak_memory_utilization`
- 机制解释不再使用 `shadow_miss_rate`。当前 `WREC 主线` 下更应该看 `wrec_residency_stats.jsonl` 中的真实 residency 指标，例如 `sum(H)/sum(U)`、`prefetch_hit_rate`、`sync_copy_bytes`，并区分 prefill 与 decode 阶段。

## Slot Budget Sensitivity

主表稳定后再跑，不建议和主表同一个 `RESULT_ROOT`。

```bash
cd /root/WREC
RUN_ID=20260515T_slot_sensitivity_n16_gmem90_mbt1280_thr037
RESULT_ROOT=/root/WREC/results/wrec/runtime_qwen_slot_sensitivity_${RUN_ID}
LOG_ROOT=/root/WREC/logs/server/qwen_slot_sensitivity_${RUN_ID}

GPU_MEMORY_UTILIZATION=0.90 \
MAX_MODEL_LEN=1280 \
MAX_NUM_SEQS=1 \
MAX_CONCURRENCY=1 \
REQUEST_FILE=/root/WREC/data/prompts/wrec_dolly_prefill_eval_long_n16_maxnew1_20260514.jsonl \
METHODS="finite_slot" \
FINITE_SLOT_CASES="24:1280 32:1280 48:1280" \
REPEATS=3 \
SIDECAR_MODE=inproc \
SIDECAR_RANKING_ONLY=1 \
SIDECAR_RANKING_SCORE_THRESHOLD=-0.37 \
RESULT_ROOT="$RESULT_ROOT" \
LOG_ROOT="$LOG_ROOT" \
EXPERIMENT_TAG="qwen_slot_sensitivity_${RUN_ID}" \
WREC_EXPERT_RESIDENCY_STATS_PATH="$RESULT_ROOT/wrec_residency_stats.jsonl" \
bash scripts/runners/run_qwen_real_inference_main.sh
```

解释方式：
- 这里的 `slot` 指当前主线方法可用的共享 expert residency 预算，不再指旧的 per-layer reserved-slot 方案。
- `slot32` 是当前主配置，因为已通过 `mbt1280` long smoke / rerun。
- `slot24` 验证更紧预算下的退化。
- `slot48` 验证更宽共享 budget 能否换到更低同步 copy 和更高命中，但前提仍要满足单层 expert 数与 `top-k * mbt` 给出的可行上界，避免 overslot。

## Failure policy

- 如果 90% gate OOM，先退到 `GPU_MEMORY_UTILIZATION=0.85`，不要改 `mbt=1280`。
- 如果 slot48 失败，不影响主表；记录为容量上探失败即可。
- 如果 n64 3 repeats 时间过长，保留 n64 r1 作为正式趋势确认，优先补齐 n16 r1-r3。
- 不再回到 `mbt=4/8` 做正式结论；这些只保留为历史 smoke/calibration。
