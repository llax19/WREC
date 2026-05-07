# Phase A status: environment, model, and data preparation

Date: 2026-04-27

## Completed

Created WREC directories:

- `models/Mixtral-8x7B-Instruct-v0.1/`
- `data/external/`
- `logs/processed/wrec/`
- `results/wrec/`
- `figures/wrec/`

Added reusable preparation scripts:

- `scripts/runtime/collect_env_inventory.py`
- `scripts/data_prep/build_wrec_dolly_manifests.py`

Generated environment inventory:

- `results/wrec/env_inventory_20260427.json`

Prepared Dolly 15k:

- Raw source: `data/external/databricks-dolly-15k.jsonl`
- Debug manifest: `data/prompts/wrec_dolly_debug_n64.jsonl`
- Train manifest: `data/prompts/wrec_dolly_train_n1024.jsonl`
- Eval manifest: `data/prompts/wrec_dolly_eval_n256.jsonl`
- Prompt stats: `results/wrec/prompt_stats_dolly_20260427.json`

Prepared Mixtral lightweight local metadata:

- `models/Mixtral-8x7B-Instruct-v0.1/config.json`
- `models/Mixtral-8x7B-Instruct-v0.1/tokenizer.model`
- `models/Mixtral-8x7B-Instruct-v0.1/tokenizer.json`
- `models/Mixtral-8x7B-Instruct-v0.1/tokenizer_config.json`
- `models/Mixtral-8x7B-Instruct-v0.1/special_tokens_map.json`

Verified Mixtral metadata:

```text
model_type: mixtral
num_hidden_layers: 32
num_local_experts: 8
num_experts_per_tok: 2
hidden_size: 4096
intermediate_size: 14336
tokenizer_vocab: 32000
```

Installed lightweight Phase A Python dependencies:

- `numpy`
- `pandas`
- `datasets`
- `huggingface_hub`
- `safetensors`
- `transformers`

## Validation

Request manifest validation passed:

```text
data/prompts/wrec_dolly_debug_n64.jsonl: 64 requests
data/prompts/wrec_dolly_train_n1024.jsonl: 1024 requests
data/prompts/wrec_dolly_eval_n256.jsonl: 256 requests
```

Approximate input token stats:

| split | n | p50 | p90 | p99 | max | categories |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| debug | 64 | 17.0 | 264.9 | 648.8 | 766 | 8 |
| train | 1024 | 19.0 | 234.7 | 629.2 | 984 | 8 |
| eval | 256 | 17.5 | 215.5 | 522.1 | 941 | 8 |

Environment summary:

```text
GPU 0: NVIDIA GeForce RTX 3090 Ti, 24564 MiB
GPU 1: NVIDIA GeForce RTX 3090 Ti, 24564 MiB
driver: 535.261.03
free disk after Phase A: 361.32 GiB
```

## Remaining blockers

Full Mixtral weights were not downloaded in Phase A.

Reason:

- Phase A only needs config/tokenizer to validate metadata and prepare HRM/profile inputs.
- The full Mixtral-8x7B weight download is large and should be performed immediately before Phase 2 trace collection, after confirming the PyTorch/CUDA environment.

PyTorch and Accelerate are still missing in the current base Python:

```text
missing_modules: torch, accelerate
```

Reason:

- Installing `accelerate` normally attempted to pull PyTorch 2.11 with CUDA 13 packages.
- The current GPU driver is 535.261.03, so CUDA 13 packages are not the safe default for this machine.
- Phase 2 should install a CUDA-compatible PyTorch build explicitly, or use an existing project environment if available.

`git-lfs` is not installed. This is not blocking Phase 0/Phase 1, but it matters if using git-based model download instead of `hf download`.

## Phase A conclusion

Phase A is complete for directory setup, environment inventory, Dolly workload preparation, request validation, and Mixtral config/tokenizer preparation.

Before Phase 2 router trace collection, one remaining environment step is required:

```text
Install or activate a CUDA-compatible PyTorch + Accelerate environment.
```
