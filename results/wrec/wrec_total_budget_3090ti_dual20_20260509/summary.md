# 3090Ti dual20 WREC total-budget replay

## Inputs

- Train trace: `logs/processed/wrec/3090ti_dual20_20260509/train_n512.jsonl`
- Eval trace: `logs/processed/wrec/3090ti_dual20_20260509/eval_n256.jsonl`
- Model: `models/Mixtral-8x7B-Instruct-v0.1`
- Effective loading cap: `0=20GiB,1=20GiB,cpu=110GiB`
- Replay output:
  - `results/wrec/wrec_total_budget_3090ti_dual20_20260509/main_recent0.json`
  - `results/wrec/wrec_total_budget_3090ti_dual20_20260509/main_recent0.csv`

## Trace check

| split | requests | input tokens | router events | failures |
|---|---:|---:|---:|---:|
| debug n64 | 64 | 3347 | 107104 | 0 |
| train n512 | 512 | 29123 | 931936 | 0 |
| eval n256 | 256 | 14147 | 452704 | 0 |

## WREC-H2 recent0 result

| total slots | LRU stall ms/token | WREC-H2 stall ms/token | gain vs LRU |
|---:|---:|---:|---:|
| 32 | 502.0173 | 141.7527 | 71.76% |
| 64 | 377.6413 | 89.0976 | 76.41% |
| 96 | 299.7991 | 58.8817 | 80.36% |
| 128 | 228.3395 | 42.5356 | 81.37% |
| 192 | 102.4833 | 25.3328 | 75.28% |

## Conclusion

- The 2 x RTX 3090 Ti container can regenerate the Mixtral `mem48`-style prefill router traces by using a safer `dual20` effective GPU cap.
- The regenerated traces match the expected request, token, layer, expert and event counts.
- The fixed total-budget replay reproduces the WREC-H2 main pattern: WREC-H2 beats LRU at all tested budgets with large transfer-stall proxy reductions.
