# WREC Phase 8 图表清单

## 主文图表

| id | 图表 | 在论文中的作用 | 状态 | 来源 |
|---|---|---|---|---|
| F1 | HRM transfer-bound heatmap | 建立 bottleneck 设定 | ready | `figures/wrec/phase8_20260503/wrec_hrm_transfer_bound_heatmap_mixtral8x7b_20260503.svg` |
| F2 | Prefill locality summary | 证明 route-trace locality | ready | `figures/wrec/phase8_20260503/wrec_locality_summary_mixtral8x7b_dolly_train_20260503.svg` |
| T1 | Fair baseline comparison | 区分 online baseline 与 oracle | ready | `results/wrec/cache_sim/mixtral8x7b_dolly_eval_n256_mem48_fair_baselines_bw41p37_20260502.md` |
| F3 | WREC stall vs total budget | 主结果 | ready | `figures/wrec/recent0_20260503/wrec_total_budget_stall_vs_budget_20260502.svg` |
| F4 | WREC oracle-gap capture | 展示剩余 headroom | ready | `figures/wrec/recent0_20260503/wrec_total_budget_oracle_gap_capture_20260502.svg` |
| F5 | WREC waste vs total budget | 支撑 zero-waste 表述 | ready | `figures/wrec/recent0_20260503/wrec_total_budget_waste_vs_budget_20260502.svg` |
| F6 | Ablation bar | 解释什么信号真正有效 | ready | `figures/wrec/phase5_20260503/wrec_ablation_bar_20260503.svg` |
| F7 | Bandwidth sensitivity | 支撑 transfer-bound 叙事 | ready | `figures/wrec/phase5_20260503/wrec_bandwidth_sensitivity_20260503.svg` |

## 附录图表

| id | 图表 | 在论文中的作用 | 状态 | 来源 |
|---|---|---|---|---|
| A1 | Default H2 total-budget result | method refinement 对照 | ready | `figures/wrec/wrec_total_budget_stall_vs_budget_20260502.svg` |
| A2 | WREC-C no-prefetch result | constrained planner 负结果 | ready | `figures/wrec/wrec_c_20260503/wrec_total_budget_stall_vs_budget_20260502.svg` |
| A3 | WREC-C constrained-prefetch waste | 说明 WREC-C 为什么失败 | ready | `figures/wrec/wrec_c_prefetch_q1_ov10_i16_20260503/wrec_total_budget_waste_vs_budget_20260502.svg` |
| A4 | Decode-only replay table | phase-transfer 负结果 | ready | `results/wrec/decode_replay/wrec_total_budget_decode_only_mixtral8x7b_dolly_eval_n64_new16_20260502.md` |
| A5 | Scale-out resident fraction | 扩展动机 | ready | `figures/wrec/scaleout_20260503/scaleout_resident_fraction_20260503.svg` |
| A6 | Scale-out transfer pressure | 扩展动机 | ready | `figures/wrec/scaleout_20260503/scaleout_expert_transfer_ratio_20260503.svg` |
| A7 | Window sensitivity | robustness detail | ready | `figures/wrec/phase5_20260503/wrec_window_sensitivity_20260503.svg` |
| A8 | History sensitivity | robustness detail | ready | `figures/wrec/phase5_20260503/wrec_history_sensitivity_20260503.svg` |

## 主文推荐顺序

1. `F1` HRM transfer-bound heatmap
2. `F2` Prefill locality summary
3. `T1` Fair baseline comparison
4. `F3` WREC stall vs total budget
5. `F4` Oracle-gap capture
6. `F6` Ablation bar
7. `F7` Bandwidth sensitivity

`F5` 可以进主文，也可以放补充材料；如果主文需要强调“zero prefetch waste”，则保留。

## 保留与降级

主文应保留：

- recent0 WREC-H2 主结果
- Belady oracle gap
- ablation 与 bandwidth sensitivity
- locality 证据

主文应降级到 appendix：

- default H2 作为旧版本
- WREC-C 作为负结果
- decode replay
- scale-out
- window/history sensitivity

## 图注书写原则

主文图注应满足三点：

1. 说清楚该图回答的研究问题，而不是重复坐标轴。
2. 明确区分 online baseline 和 oracle upper bound。
3. 避免把 appendix figure 写成主方法正结果。
