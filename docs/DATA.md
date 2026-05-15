# Data Boundary

This repository tracks compact experiment artifacts and excludes rebuildable heavy files.

## Tracked

- `data/prompts/*.jsonl`: request manifests and prompt samples.
- `results/wrec/**/*.md`: experiment conclusions and run summaries.
- `results/wrec/**/*.csv`: tabular results used by figures/tables.
- `results/wrec/**/*.json`: compact metric summaries and metadata.
- `figures/wrec/**/*.svg`: generated paper figures.
- `record/**/*.md`: operation notes and implementation records.

## Excluded

- `models/` and `qwen1.5-MoE-A2.7B/`: local model weights and tokenizer files.
- `data/external/`: downloaded upstream datasets such as Dolly.
- `logs/processed/**/*.jsonl`: full router-event traces and WREC decision streams.
- `logs/raw/`, `logs/server/`, `logs/runs/`: raw request logs, service logs, and launch records.
- `external/`: upstream source trees, represented by patches under `patches/`.
- Top-level PDFs and extracted reference text.

## Trace Restoration

Some runtime and replay scripts expect processed traces under `logs/processed/wrec/`, for example:

- `mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- `mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- `wrec_decisions_mixtral8x7b_dolly_n256_20260502.jsonl`

These files are generated from model runs and are intentionally not committed. To reproduce from scratch, regenerate them with the router trace builders in `scripts/moe_affinity/` and validate the contract with:

```bash
PYTHONPATH=scripts python scripts/wrec/validation/validate_runtime_trace_contract.py --help
```

The compact summaries in `results/wrec/runtime_contract/` document the expected Mixtral `mem48` trace dimensions.

## vLLM Python Patch Deployment

WREC runtime smoke tests should not manually edit conda `site-packages`.
Use the deployment helper to copy the Python-only patch set from
`external/vllm-0.19.0` into the installed vLLM package:

```bash
/root/miniconda3/envs/vllm_moe/bin/python scripts/wrec/runtime/deploy_vllm_python_patch.py \
  --python /root/miniconda3/envs/vllm_moe/bin/python
```

Each run writes a compact manifest under `results/wrec/runtime_patch_deploy/`
and backs up overwritten files under `/tmp/wrec_vllm_sitepkg_backup_*`.
Use `--dry-run` before deployment and `--restore-manifest <manifest.json>` to
restore the previous installed Python files. The source of truth remains the
modified tree under `external/vllm-0.19.0`; site-packages is only a deploy
target needed because the local tree does not provide a matching `vllm._C`.
