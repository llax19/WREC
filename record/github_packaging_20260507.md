# GitHub packaging record 2026-05-07

## 操作

- 将 `workspace` 初始化为 git 仓库，并把默认分支改为 `main`。
- 新增仓库级文件：
  - `README.md`
  - `.gitignore`
  - `requirements.txt`
  - `docs/DATA.md`
  - `logs/processed/wrec/README.md`
  - `models/README.md`
- 将本地 vLLM `v0.19.0` 的 WREC runtime hook 修改导出为补丁：
  - `patches/vllm_wrec_runtime_hooks_20260507/vllm_wrec_runtime_hooks.patch`
  - `patches/vllm_wrec_runtime_hooks_20260507/README.md`
- 检查超过 `10MB` 的文件，确认无大文件会进入 git 跟踪。
- 创建首个本地提交：
  - `b182bf5 Prepare WREC experiment repository`
- 基于该提交生成发布压缩包：
  - `wrec-github-project_20260507.tar.gz`

## 原理

- GitHub 项目应保留代码、轻量实验摘要、图表、记录和补丁。
- 模型权重、完整 router event JSONL、原始服务日志、上游源码树和论文 PDF 属于大体积或可重建/外部材料，不进入 git。
- vLLM 源码不直接 vendor；只保留可应用 patch，避免仓库膨胀并保留集成边界。

## 结论

- 当前 `workspace` 已具备本地 GitHub 项目形态。
- 发布压缩包大小约 `2.1MB`，只包含 git 跟踪文件。
- `models/`、`qwen1.5-MoE-A2.7B/`、`logs/raw/`、`logs/server/`、`logs/runs/`、`logs/processed/**/*.jsonl`、`external/` 和顶层 PDF 已被忽略。
- 当前环境没有 `gh`，且仓库尚未配置 remote；实际 push 需要后续提供 GitHub 仓库 URL 或安装/登录 GitHub CLI。
- WREC runtime 下一步实验仍应从 state-only sidecar baseline 继续测试 finite slot，不应默认启用 full-shape row-copy。
