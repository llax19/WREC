# WREC Phase 5 Ablation

## full_wrec_h2

- rows: 3
- avg gain vs LRU: 77.80%
- avg saved stall: 233.0356 ms/input-token

## lru_baseline

- rows: 3
- avg gain vs LRU: 0.00%
- avg saved stall: 0.0000 ms/input-token

## no_cross_layer_signal

- rows: 3
- avg gain vs LRU: 68.54%
- avg saved stall: 204.9079 ms/input-token

## no_recent_signal

- rows: 3
- avg gain vs LRU: 79.40%
- avg saved stall: 238.4813 ms/input-token

## no_request_signal

- rows: 3
- avg gain vs LRU: 77.70%
- avg saved stall: 231.9129 ms/input-token

## no_transfer_term

- rows: 3
- avg gain vs LRU: 77.80%
- avg saved stall: 233.0356 ms/input-token

## no_workload_term

- rows: 3
- avg gain vs LRU: 77.82%
- avg saved stall: 233.1178 ms/input-token

## train_window_only

- rows: 3
- avg gain vs LRU: -0.93%
- avg saved stall: -2.0868 ms/input-token

## wrec_h_recent_only

- rows: 3
- avg gain vs LRU: 43.51%
- avg saved stall: 125.9300 ms/input-token

## Notes

- WREC-H2 is demand-load admission/eviction in this simulator, so prefetch-only ablation is not emitted here.
- `no_transfer` changes the WREC score formula, but same-sized experts make the transfer term mostly a constant-ranking term.
- `route_window_prefetch` uses eval future window in the existing simulator and should be read as oracle-style stress test, not deployable online policy.
