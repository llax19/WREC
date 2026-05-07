# WREC 后续实验加深建议

日期：2026-05-03

## 当前判断

当前 WREC 主线已经不应被描述为简单 toy experiment。已有证据包括 HRM bottleneck screen、Mixtral prefill route trace、fair baselines、Belady oracle gap、fixed total-budget 主结果、ablation、bandwidth sensitivity、decode 负结果和 scale-out appendix。

如果继续加深，重点不应是重新发明 WREC score 或继续堆同类 simulator 表，而应补足以下四类审稿/答辩中容易被追问的问题：

1. 结果是否只对 Dolly workload 成立。
2. simulator proxy 是否有足够的硬件校准依据。
3. 结果是否有统计稳定性，而不是单次 trace replay。
4. 方法是否有清晰 runtime 落地点，即使第一版不做完整 vLLM 集成。

## 优先级 A：最值得补的三类实验

### A1. Workload generalization

目的：降低“只在 Dolly eval n256 上有效”的风险。

建议先做低风险版本，再决定是否扩展真实第二 workload：

- 低风险版本：按 Dolly category 切分 eval trace，分别报告 WREC vs LRU/static-hot/Belady。
- 加强版本：补一个第二 workload 的 prefill trace，例如真实对话或指令类数据；若数据授权不顺利，不应阻塞主线。
- 报告方式：不需要重新画大量图，做一张 `workload/category x gain` 表即可。

推荐产物：

- `results/wrec/generalization/wrec_category_generalization_mixtral8x7b_dolly_YYYYMMDD.{csv,md}`
- `figures/wrec/generalization_YYYYMMDD/wrec_category_gain.svg`

论文作用：证明 WREC 不是只贴合一个 eval trace 的单点结果。

### A2. Statistical robustness

目的：把单次 replay 结果改成有置信区间或分段稳定性的结果。

建议做两种轻量统计：

- 按 request bootstrap：从 eval n256 的 request 粒度重采样，报告 gain vs LRU 的 mean / p05 / p95。
- 按 trace chunk：将 eval trace 按 request 顺序分成若干 block，报告每个 block 的 WREC gain。

注意：bootstrap 不能伪装成独立实验，只能作为 trace 内稳定性分析。论文表述应写成“within-trace robustness”。

推荐产物：

- `results/wrec/robustness/wrec_bootstrap_mixtral8x7b_dolly_n256_YYYYMMDD.{csv,md}`
- `results/wrec/robustness/wrec_blockwise_mixtral8x7b_dolly_n256_YYYYMMDD.{csv,md}`

论文作用：降低“偶然 trace 顺序导致收益”的质疑。

### A3. Simulator validity calibration

目的：让 simulator 不只是抽象公式，而是和当前机器上的实际传输能力有对应关系。

已有 `hrm_hardware_profile_20260501.json` 记录了 measured bandwidth，后续可补：

- expert-size copy microbenchmark：按 Mixtral expert weight size 估计或构造等价 tensor copy，测 p50/p90 copy time。
- policy overhead microbenchmark：统计 WREC-H2 per-event scoring / eviction 的 CPU 时间。
- sensitivity sanity check：说明主表使用的 `41.37 GB/s` 与硬件 profile 一致，且低带宽下 absolute saved stall 增大。

推荐产物：

- `results/wrec/calibration/wrec_transfer_microbench_mixtral8x7b_YYYYMMDD.{json,md}`
- `results/wrec/calibration/wrec_policy_overhead_mixtral8x7b_YYYYMMDD.{json,md}`

论文作用：把 simulator 结果从“纯玩具模拟”推进到“trace-driven + hardware-calibrated simulation”。

## 优先级 B：有余力时补

### B1. Larger trace scale

如果时间和显存允许，可以把 eval trace 从 `n256` 扩到 `n512` 或 `n1024`。这比继续调超参更能提升可信度。

风险：Mixtral trace 采集成本较高，且失败恢复复杂。若执行，必须记录失败原因和处理过程。

### B2. Decode-specific extension only as appendix

当前已有 decode 迁移失败结果。若继续加深 decode，不建议让 prefill-trained WREC 继续调参，而应做最小 WREC-D：

- 使用 decode train trace 构建 decode-specific prior。
- 报告 decode-only LRU/static-hot/WREC-D/Belady。
- 若 WREC-D 仍失败，作为 phase-specific difficulty 的负结果。

论文定位：appendix 或 future work，不进入主方法正结果。

### B3. Runtime integration survey / API sketch

不必直接承诺完整 vLLM 集成，但可以补一份系统化设计：

- vLLM / serving runtime 中 expert loading、routing、cache manager 的可能接入点。
- WREC runtime API：输入 router event，输出 admission/eviction/prefetch decision。
- policy overhead 和异步拷贝约束。

推荐产物：

- `notes/wrec_runtime_integration_survey_YYYYMMDD.md`
- `results/wrec/runtime/wrec_runtime_api_overhead_YYYYMMDD.md`

论文作用：说明 WREC 有明确系统落地点，但不越界声称 production-ready。

## 不建议继续投入的方向

1. 不建议继续深挖 WREC-C prefetch，除非先做 next-use precision 诊断。当前结果已经说明 constrained-prefetch 会带来 waste 和 cache pollution。
2. 不建议把 decode 并入主文正结果。decode 会把问题扩大成 phase-aware scheduling，容易冲淡当前 prefill 主线。
3. 不建议继续增加 config-only scale-out 表。scale-out 现在作为 appendix 已经足够，继续增加模型不会显著增强主结论。
4. 不建议只做更多 bandwidth 数值点。已有 sensitivity 已经支撑 transfer-bound 叙事。

## 推荐执行顺序

建议按以下顺序推进：

1. 做 A2 statistical robustness，成本最低，能立刻提升论文严谨性。
2. 做 A1 Dolly category generalization，优先复用现有 trace 与 manifest。
3. 做 A3 simulator validity calibration，补足 hardware-calibrated simulation 表述。
4. 若仍有时间，再尝试第二 workload 或更大 eval trace。
5. 最后才考虑 runtime integration survey；它应是系统讨论补强，而不是主实验前置条件。

## 论文表述建议

完成 A1-A3 后，论文实验章节可以从：

> We evaluate WREC using a trace-driven simulator.

提升为：

> 本文采用硬件校准的 trace-driven replay simulator 评估 WREC。实验覆盖 Mixtral-8x7B 的 prefill router trace、不同 total expert-cache budgets、多个在线基线、Belady oracle upper bound、消融与带宽敏感性，并进一步通过 workload/category 切分和 request-level robustness 分析检验结果稳定性。

这样可以明显降低 toy 感，同时不需要把论文扩展成完整 serving runtime 系统。
