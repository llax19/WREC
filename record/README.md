# Record Notes

`record/` 现在只保留人工整理的工作记录：

- 手写实验记录
- 校准笔记
- 参数扫描总结
- 实验阶段性复盘

说明：

- vLLM 服务端日志统一放在 `/root/workspace/logs/server/`。
- 自动生成的单轮运行说明统一放在 `/root/workspace/logs/runs/`。
- 请求级原始日志仍放在 `/root/workspace/logs/raw/`。
- GPU 摘要日志仍放在 `/root/workspace/logs/processed/`。

调度实现边界：

- 当前客户端脚本里的 `length_only` 只是请求发送前的代理重排。
- 这类结果只能作为预实验或链路验证，不能代表最终嵌入 vLLM 服务端调度器后的结果。
- 后续论文主结果应以服务端内嵌调度实现为准。
- 当前 conda 环境中的 vLLM 已增加 `--scheduling-policy length_only`，用于服务端内嵌
  Length-only 调度实验。
