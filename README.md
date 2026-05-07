# WREC MoE Inference Experiments

This repository contains the code, lightweight experiment outputs, and runtime patch notes for WREC, an experimental MoE inference caching/scheduling workflow for resource-constrained single-node environments.

The repository is organized as a publishable subset of the local workspace. Large model weights, raw server logs, full router-event JSONL traces, downloaded papers, and vendored upstream source trees are intentionally excluded from git.

## Layout

- `scripts/`: experiment runners, runtime helpers, data preparation, analysis, WREC policy/runtime code, and MoE affinity replay tools.
- `data/prompts/`: prompt/request manifests used by the experiments.
- `results/wrec/`: compact metric summaries, CSV/JSON tables, and Markdown result notes.
- `figures/wrec/`: generated SVG figures for WREC analyses.
- `patches/`: vLLM patch material and integration notes.
- `record/`: concise experiment records and implementation notes.
- `docs/`: repository-level data and reproducibility notes.

## What Is Not Tracked

- `models/` and `qwen1.5-MoE-A2.7B/`: local Hugging Face model weights.
- `logs/raw/`, `logs/server/`, `logs/runs/`: rebuildable raw logs and run logs.
- `logs/processed/**/*.jsonl`: full processed router-event traces; see `logs/processed/wrec/README.md`.
- `external/`: upstream source checkouts such as vLLM; WREC changes are exported under `patches/`.
- Top-level PDFs and extracted paper text.

## Environment

Most offline analysis scripts use the Python standard library. Runtime tracing and model execution require a GPU environment with PyTorch, Transformers, and vLLM.

```bash
python -m pip install -r requirements.txt
```

For the original runtime experiments, the local environment used vLLM `0.19.0` and local Mixtral/Qwen MoE weights. The exact local inventory is recorded in `results/wrec/env_inventory_*.json`.

## Common Commands

Validate the current Python/vLLM environment:

```bash
python scripts/runtime/check_vllm_env.py
```

Run an offline WREC cache replay after restoring the processed trace files:

```bash
PYTHONPATH=scripts python scripts/moe_affinity/simulate_expert_cache_total_budget.py --help
```

Start the WREC sidecar boundary:

```bash
PYTHONPATH=scripts python scripts/wrec/runtime/runtime_sidecar.py --help
```

Apply the vLLM WREC runtime hook patch to a clean vLLM `v0.19.0` checkout:

```bash
git -C /path/to/vllm-v0.19.0 apply /path/to/this/repo/patches/vllm_wrec_runtime_hooks_20260507/vllm_wrec_runtime_hooks.patch
```

## Data Boundary

The tracked result files are intended for review, plotting, and paper writing. Full trace JSONL files are excluded because they are large generated artifacts. See `docs/DATA.md` for the file boundary and restoration policy.
