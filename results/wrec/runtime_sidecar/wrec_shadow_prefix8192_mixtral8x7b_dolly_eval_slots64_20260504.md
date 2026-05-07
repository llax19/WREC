# WREC Runtime Shadow Report

## Setup

- Event stream: `/root/workspace/logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Train prior trace: `/root/workspace/logs/processed/wrec/mixtral8x7b_dolly_train_router_events_n512_mem48_20260501.jsonl`
- Model path: `/root/workspace/models/Mixtral-8x7B-Instruct-v0.1`
- Policy: `wrec_h2_shadow`
- Total slots: `64`
- History size: `8`
- Weights: recent `0.0`, request `1024.0`, cross-layer `1024.0`
- Bandwidth: `41.37220609315469` GB/s

## Online Shadow Counts

- Expert refs processed: `8192`
- Router events observed: `4096`
- Shadow hits: `7237`
- Shadow misses: `955`
- Shadow hit rate: `0.883422852`
- Shadow miss rate: `0.116577148`
- Would-admit count: `651`
- Would-bypass count: `304`
- Would-evict count: `651`
- Final resident experts: `64`

## Transfer Proxy

- Expert bytes: `352321536.0`
- Demand transfer bytes: `336467066880.0`
- Estimated stall: `8132.683718` ms
- Estimated stall per input token: `0.574870` ms

## Runtime Overhead

- Total shadow loop: `0.094906` s
- Mean overhead per expert ref: `11.585` us
- Mean overhead per router event: `23.170` us
- History update cost per expert ref: `0.587` us
- Decision cost per miss: `90.025` us

## Conclusion

- WREC consumed routed expert events sequentially and made each shadow cache decision without future accesses.
- This validates the policy/state interface as runtime-compatible, but it does not prove end-to-end serving latency improvement.
- The current shadow mode does not control real expert loading; it records what WREC would keep, admit, bypass, and evict.
