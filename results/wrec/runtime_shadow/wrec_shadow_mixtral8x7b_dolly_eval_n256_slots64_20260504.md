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

- Expert refs processed: `905408`
- Router events observed: `452704`
- Shadow hits: `757524`
- Shadow misses: `147884`
- Shadow hit rate: `0.836665901`
- Shadow miss rate: `0.163334099`
- Would-admit count: `112416`
- Would-bypass count: `35468`
- Would-evict count: `112416`
- Final resident experts: `64`

## Transfer Proxy

- Expert bytes: `352321536.0`
- Demand transfer bytes: `52102718029824.0`
- Estimated stall: `1259365.234537` ms
- Estimated stall per input token: `89.019950` ms

## Runtime Overhead

- Total shadow loop: `14.658765` s
- Mean overhead per expert ref: `16.190` us
- Mean overhead per router event: `32.380` us
- History update cost per expert ref: `0.614` us
- Decision cost per miss: `91.532` us

## Conclusion

- WREC consumed routed expert events sequentially and made each shadow cache decision without future accesses.
- This validates the policy/state interface as runtime-compatible, but it does not prove end-to-end serving latency improvement.
- The current shadow mode does not control real expert loading; it records what WREC would keep, admit, bypass, and evict.
