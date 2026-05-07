# Workspace Map

这份清单的目标只有一个：让你快速判断 `workspace/` 里的东西，到底是
“后面还要继续用的输入/脚本”，还是“跑实验时自动生成、可以重建的中间产物”。

## 一眼判断

### 应该长期保留

- `scripts/`
  - 实验入口脚本、日志采集脚本、汇总分析脚本。
- `data/prompts/`
  - 实验输入请求集，属于源数据。
- `results/`
  - 已经汇总好的指标结果，适合论文表格和后续对比直接引用。
- `qwen1.5-MoE-A2.7B/`
  - 当前实验实际依赖的本地模型权重。
- `experiment_guide.md`
  - 实验方法和边界定义。
- `README_experiments.md`
  - 工作区快速说明。

### 生成后可重建

- `logs/raw/`
  - 原始请求日志和 GPU 监控日志。
  - 这是最典型的中间产物，保留它可以重算结果；不保留也不会影响脚本本身。
- `logs/processed/`
  - 从原始 GPU 日志进一步提炼出来的摘要。
  - 可以由 `logs/raw/` 重建。
- `logs/server/`
  - 运行脚本自动写出的 vLLM 服务端日志。
  - 主要用于排障，不是核心输入数据。
- `logs/runs/`
  - 运行脚本自动写出的单轮运行说明和正式实验 launch 记录。
  - 主要用于排障和追溯，不是核心输入数据。
- `paper_extracted.txt`
  - 更像从 PDF 提取出来的辅助文本，属于可重建的参考材料。

### 参考资料或旁路资源

- `LightLLM/`
  - 一份框架源码工作树。
  - 目前你这套主实验链路主要跑的是 `vLLM`，所以它更像参考/备选实现，不是当前主流程必需目录。
- `NeurIPS-2024-efficient-llm-scheduling-by-learning-to-rank-Paper-Conference.pdf`
  - 参考论文。
- `开题报告-终版.pdf`
  - 论文相关材料。

### 临时/草稿性质

- `test.py`
  - 一个独立的 CUDA 预取实验草稿，不属于当前主实验流水线。
- `scripts/**/__pycache__/`
  - Python 运行缓存，出现后可直接清理，后续会自动再生成。
- `logs/raw/manual_probe.jsonl`
  - 已删除。原为空文件，属于临时探针。
- `logs/raw/sample_metrics.jsonl`
  - 已删除。原为样例文件，不是正式实验结果。

## 目录说明

### `scripts/`

这是当前最重要的“可执行资产”目录，已经按职责拆成子目录。

- `scripts/runners/`
  - 实验入口和批量运行脚本，例如 `run_stage1_single_gpu.sh`、
    `run_stage1_dual_gpu.sh`、`run_tp2_load_levels.sh`、
    `run_tp2_fcfs_formal.sh`、`run_tp2_length_only_server_formal.sh`、
    `run_tp2_ltr_server_formal.sh`、`run_tp2_moe_affinity_server_formal.sh`。
- `scripts/runtime/`
  - 实验运行时辅助脚本，包括请求发送、服务等待、GPU 监控、环境检查和 smoke test。
- `scripts/analysis/`
  - 原始日志摘要和多轮结果聚合脚本。
- `scripts/data_prep/`
  - prompt/request 清单生成与校验脚本。
- `scripts/ltr/`
  - LTR score 文件生成与轻量 predictor 训练脚本。
- `scripts/moe_affinity/`
  - MoE expert signature 构建、Qwen/OLMoE router trace 处理和离线 locality replay。

结论：`scripts/` 整体都该保留。

## 输出文件怎么判断

### 一次完整实验通常会生成 4 组东西

1. `logs/raw/*_fcfs_*.jsonl` 或类似文件
   - 请求级原始日志。
2. `logs/raw/*_gpu_metrics.jsonl`
   - GPU 监控原始日志。
3. `logs/processed/*_gpu_summary.json`
   - GPU 日志汇总结果。
4. `results/*_summary.json`
   - 论文最常用的指标摘要。

### 日志文件再细分

- `logs/server/*_vllm_server.log`
  - 服务端标准输出/报错日志。
  - 排障时有用，但通常不是最终结果。
- `logs/runs/*_stage1_run.md` / `logs/runs/*_dual_gpu_run.md`
  - 自动生成的运行说明。
  - 价值是“知道这次实验怎么跑出来的”。
- `record/4.21record.md` / `record/4.22record.md` / `record/tp2_*_20260422.md`
  - 手写实验记录和过程总结。
  - 这类应该保留。

## 现在这份工作区的保留建议

### 建议保留

- `scripts/`
- `data/prompts/`
- `results/`
- `record/` 里的手写总结文档
- `qwen1.5-MoE-A2.7B/`
- `experiment_guide.md`
- `README_experiments.md`

### 建议按需归档，不必一直堆在工作区

- `logs/raw/`
- `logs/processed/`
- `logs/server/`
- `logs/runs/`

### 可以视为临时文件

- `test.py`
- `paper_extracted.txt`
- `logs/raw/manual_probe.jsonl`，已删除
- `logs/raw/sample_metrics.jsonl`，已删除

## 这次整理已经做了什么

- `record/` 已整理为只保留人工工作记录。
- 旧的 `*_vllm_server.log` 已迁移到 `logs/server/`。
- 旧的自动运行记录 `*_stage1_run.md`、`*_dual_gpu_run.md` 和正式 launch 记录已迁移到 `logs/runs/`。
- 后续脚本新产生的服务日志会写到 `logs/server/`，运行说明会写到 `logs/runs/`。
- 删除了空探针日志 `logs/raw/manual_probe.jsonl` 和样例日志 `logs/raw/sample_metrics.jsonl`。
- 2026-04-27：`scripts/` 已按职责拆分为 `runners/`、`runtime/`、
  `analysis/`、`data_prep/`、`ltr/`、`moe_affinity/`，并清理了旧的
  `scripts/__pycache__/`。

## 后续建议

如果你准备继续整理第二轮，我建议按下面顺序做，不容易出错：

1. 先归档 `logs/server/` 中已经不再排障使用的旧服务日志。
2. 再归档 `logs/raw/` 中已经有 `results/*.json` 对应结果的旧原始日志。
3. 最后再决定是否保留 `LightLLM/` 和 `paper_extracted.txt` 这类参考/草稿材料。
