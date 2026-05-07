# WREC Policy Overhead Benchmark

## Setup

- Eval trace: `workspace/logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train trace: `workspace/logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Policy: `wrec_h2_no_recent`
- Total slots: `64`
- History size: `8`
- Weights: recent `0.0`, request `1024.0`, cross-layer `1024.0`

## Counts

- Expert refs processed: `905408`
- Router events: `452704`
- Demand misses: `147884`
- Demand hits: `757524`
- Final resident experts: `64`

## Online Overhead

- Total timed online loop: `13.857839` s
- Mean overhead per expert ref: `15.306` us
- Mean overhead per router event: `30.611` us
- Mean history update cost per expert ref: `0.565` us
- Mean admission decision cost per miss: `88.290` us
- Chunk p50/p95 overhead per expert ref: `14.956` / `24.777` us

## Transfer Reference

- Expert bytes: `352321536.0`
- Bandwidth: `41.37` GB/s
- Estimated transfer time per expert miss: `8.516353` ms
- Policy loop / expert transfer ratio: `0.001797`

## Conclusion

- The benchmark isolates WREC online CPU decision overhead from trace loading and offline prior construction.
- The timed loop includes cache lookup/touch, online history update, and WREC admission/eviction decisions.
- This is a Python-level overhead measurement, so it is a conservative proxy for a production implementation in a lower-level runtime.
- The measured online overhead is more than two orders of magnitude smaller than the calibrated expert transfer time used by the replay simulator.
