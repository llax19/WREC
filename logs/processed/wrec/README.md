# Processed WREC Traces

Full processed router-event and WREC decision JSONL files are not tracked in git because this directory is about 1.9G locally.

The publishable repository keeps compact summaries in `results/wrec/` and documents the required trace contract in `results/wrec/runtime_contract/`.

Important local trace names:

- `mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- `mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- `wrec_decisions_mixtral8x7b_dolly_n256_20260502.jsonl`

Regenerate traces with scripts under `scripts/moe_affinity/` when reproducing the full pipeline.
