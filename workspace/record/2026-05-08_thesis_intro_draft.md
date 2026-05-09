# 2026-05-08 thesis intro draft

## 操作

- 阅读 `thesis_writing_req.md`、`thesis_req.md`、`/root/thesis_template/zjuthesis.tex`、本科最终论文正文入口和 WREC 写作素材。
- 确认绪论应兼容模板中的 `body/undergraduate/final/1-introduction.tex`，该模板当前使用 `\section{绪论}` 作为本科最终论文正文标题层级。
- 因 `/root/thesis_template` 位于 WREC 工作区外，直接修改共享模板被安全策略拦截；改为在 `thesis/final/` 下生成可迁移草稿：
  - `thesis/final/1-introduction.tex`
  - `thesis/final/ref_wrec.bib`

## 原理

- 绪论按毕业论文要求覆盖课题背景、研究问题、研究意义、技术路线选择、主要工作和全文结构。
- 内容围绕 WREC 主线收束：资源受限 MoE 推理、真实路由序列、专家缓存/卸载/预取、trace-driven replay 和 WREC 策略。
- 按用户要求假定 runtime final slot smoke 已完成，但绪论不写端到端时延或未最终验证的量化结论。

## 结论

- 已完成绪论草稿，可复制到 `/root/thesis_template/body/undergraduate/final/1-introduction.tex`。
- 若要在模板中直接编译，还需要将 `ref_wrec.bib` 中的 `jiang2024mixtral` 条目合入模板 `body/ref.bib`。

## 同步补记

- 用户确认允许同步后，已将 `thesis/final/1-introduction.tex` 写入 `/root/thesis_template/body/undergraduate/final/1-introduction.tex`。
- 已在 `/root/thesis_template/body/ref.bib` 中补充 `jiang2024mixtral` 条目；检查确认绪论引用的 `shazeer2017outrageously`、`fedus2022switch`、`jiang2024mixtral`、`kwon2023pagedattention`、`sheng2023flexgen` 均存在。
- `cmp` 检查显示模板绪论与 WREC 草稿一致。
