# WREC Runtime Shadow Report

## Setup

- Event stream: `logs/processed/wrec/3090ti_dual20_20260509/eval_n256.jsonl`
- Train prior trace: `logs/processed/wrec/3090ti_dual20_20260509/train_n512.jsonl`
- Model path: `models/Mixtral-8x7B-Instruct-v0.1`
- Policy: `wrec_h2_shadow`
- Total slots: `64`
- History size: `8`
- Weights: recent `0.0`, request `1024.0`, cross-layer `1024.0`
- Bandwidth: `41.37220609315469` GB/s

## Online Shadow Counts

- Expert refs processed: `905408`
- Router events observed: `452704`
- Shadow hits: `757395`
- Shadow misses: `148013`
- Shadow hit rate: `0.836523424`
- Shadow miss rate: `0.163476576`
- Would-admit count: `112458`
- Would-bypass count: `35555`
- Would-evict count: `112458`
- Final resident experts: `64`

## Transfer Proxy

- Expert bytes: `352321536.0`
- Demand transfer bytes: `52148167507968.0`
- Estimated stall: `1260463.785532` ms
- Estimated stall per input token: `89.097603` ms

## Runtime Overhead

- Total shadow loop: `27.645532` s
- Mean overhead per expert ref: `30.534` us
- Mean overhead per router event: `61.068` us
- History update cost per expert ref: `1.085` us
- Decision cost per miss: `173.289` us

## Conclusion

- WREC consumed routed expert events sequentially and made each shadow cache decision without future accesses.
- This validates the policy/state interface as runtime-compatible, but it does not prove end-to-end serving latency improvement.
- The current shadow mode does not control real expert loading; it records what WREC would keep, admit, bypass, and evict.
