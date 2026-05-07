# Scripts Directory

`scripts/` 按实验生命周期拆分，避免所有入口和工具堆在同一层。

- `runners/`: 可直接执行的实验入口与批量运行脚本。
- `runtime/`: 运行时辅助脚本，例如发请求、等待服务、监控 GPU、环境检查。
- `analysis/`: 请求日志摘要与多轮结果聚合。
- `data_prep/`: prompt/request 清单生成与校验。
- `ltr/`: LTR score 构建和轻量 predictor 训练。
- `moe_affinity/`: expert signature 构建、router trace 处理和离线 replay。

常用入口：

```bash
python scripts/runtime/check_vllm_env.py
bash scripts/runners/run_vllm_smoke_tp2.sh
bash scripts/runners/run_tp2_fcfs_formal.sh
bash scripts/runners/run_tp2_length_only_server_formal.sh
python scripts/analysis/analyze_logs.py logs/raw/<run>.jsonl
```
