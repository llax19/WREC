# WREC Phase 5 Sensitivity

## full_wrec_h2_bw_16

- rows: 3
- avg gain vs LRU: 77.67%
- avg saved stall: 601.5417 ms/input-token

## full_wrec_h2_bw_32

- rows: 3
- avg gain vs LRU: 77.79%
- avg saved stall: 301.2620 ms/input-token

## full_wrec_h2_bw_41.3722

- rows: 3
- avg gain vs LRU: 77.80%
- avg saved stall: 233.0356 ms/input-token

## full_wrec_h2_bw_8

- rows: 3
- avg gain vs LRU: 77.42%
- avg saved stall: 1198.9577 ms/input-token

## full_wrec_h2_history_1

- rows: 3
- avg gain vs LRU: 76.33%
- avg saved stall: 227.8242 ms/input-token

## full_wrec_h2_history_16

- rows: 3
- avg gain vs LRU: 77.21%
- avg saved stall: 231.3369 ms/input-token

## full_wrec_h2_history_32

- rows: 3
- avg gain vs LRU: 77.09%
- avg saved stall: 231.0184 ms/input-token

## full_wrec_h2_history_4

- rows: 3
- avg gain vs LRU: 75.63%
- avg saved stall: 226.2349 ms/input-token

## full_wrec_h2_history_8

- rows: 3
- avg gain vs LRU: 77.80%
- avg saved stall: 233.0356 ms/input-token

## full_wrec_h2_window_1

- rows: 3
- avg gain vs LRU: 77.85%
- avg saved stall: 233.1915 ms/input-token

## full_wrec_h2_window_16

- rows: 3
- avg gain vs LRU: 77.48%
- avg saved stall: 232.0102 ms/input-token

## full_wrec_h2_window_32

- rows: 3
- avg gain vs LRU: 77.15%
- avg saved stall: 230.9496 ms/input-token

## full_wrec_h2_window_4

- rows: 3
- avg gain vs LRU: 77.80%
- avg saved stall: 233.0356 ms/input-token

## full_wrec_h2_window_8

- rows: 3
- avg gain vs LRU: 77.70%
- avg saved stall: 232.7382 ms/input-token

## lru_baseline

- rows: 18
- avg gain vs LRU: 0.00%
- avg saved stall: 0.0000 ms/input-token

## Notes

- WREC-H2 is demand-load admission/eviction in this simulator, so prefetch-only ablation is not emitted here.
- `no_transfer` changes the WREC score formula, but same-sized experts make the transfer term mostly a constant-ranking term.
- `route_window_prefetch` uses eval future window in the existing simulator and should be read as oracle-style stress test, not deployable online policy.
