# WREC Workload Manifest Status

Date: 2026-04-28

## Goal

Close Phase 1 for the Dolly workload manifests before router event tracing.

## Inputs

- `data/prompts/wrec_dolly_debug_n64.jsonl`
- `data/prompts/wrec_dolly_train_n1024.jsonl`
- `data/prompts/wrec_dolly_eval_n256.jsonl`
- `results/wrec/prompt_stats_dolly_20260427.json`

## Validation

`check_requests.py` passed for all three manifests.

| split | path | requests | unique ids | p99 approx input tokens | max approx input tokens | categories |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| debug | `data/prompts/wrec_dolly_debug_n64.jsonl` | 64 | 64 | 648.82 | 766 | 8 |
| train | `data/prompts/wrec_dolly_train_n1024.jsonl` | 1024 | 1024 | 629.16 | 984 | 8 |
| eval | `data/prompts/wrec_dolly_eval_n256.jsonl` | 256 | 256 | 522.05 | 941 | 8 |

Additional checks:

- `id` and `request_id` match for every row.
- Debug, train, and eval splits have no overlapping `source_index`.
- All p99 approximate input lengths are below the Phase 1 limit of 1024 tokens.
- The Dolly category coverage includes 8 categories in each split.

## Conclusion

Phase 1 is closed for the current WREC main workload.

The next non-Mixtral-blocked step is Phase 2A: run Qwen router event tracing on the debug subset to validate the token/layer event schema and trace collection path.
