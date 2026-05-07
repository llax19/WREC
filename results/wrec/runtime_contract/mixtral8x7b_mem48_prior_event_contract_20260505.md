# WREC Runtime Trace Contract Validation

## Contract

- Model family: `mixtral8x7b`
- Expected layers: `32`
- Expected experts/layer: `8`
- Expected top-k: `2`
- Train prior trace: `/root/workspace/logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Runtime event trace: `/root/workspace/logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`

## Verdict

- Overall pass: `True`

## Trace Summary

| trace | requests | input tokens | router events | expert refs | layers complete | experts complete | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 512 | 29123 | 931936 | 1863872 | True | True | 0 |
| event | 256 | 14147 | 452704 | 905408 | True | True | 0 |

## Stats Checks

| trace | metric | actual | expected | pass |
|---|---|---:|---:|---:|
| train | requests | 512 | 512 | True |
| train | input_tokens | 29123 | 29123 | True |
| train | router_events | 931936 | 931936 | True |
| train | num_layers | 32 | 32 | True |
| train | num_experts | 8 | 8 | True |
| event | requests | 256 | 256 | True |
| event | input_tokens | 14147 | 14147 | True |
| event | router_events | 452704 | 452704 | True |
| event | num_layers | 32 | 32 | True |
| event | num_experts | 8 | 8 | True |

## Conclusion

- The existing Mixtral `mem48` train trace is a valid WREC prior source for 32 layers, 8 experts, and top-k 2.
- The existing Mixtral `mem48` eval trace satisfies the same runtime event contract and can be sent to the sidecar without model-dimension mismatch.
- This validates prior/event compatibility only; it does not modify vLLM or control real expert residency.
