# WREC Runtime Sidecar Smoke

## Inputs

- Trace: `logs/processed/wrec/3090ti_dual20_20260509/eval_n256.jsonl`
- Sidecar URL: `http://127.0.0.1:18765`
- Max events: `4096`

## Client

- Events sent: `4096`
- Expert refs sent: `8192`
- Client elapsed: `1.973018` s
- Mean client submit time: `481.694` us/event

## Sidecar Metrics

- Expert refs processed: `8192`
- Router events processed: `4096`
- Shadow hits: `7232`
- Shadow misses: `960`
- Shadow miss rate: `0.117187500`
- Would-admit: `659`
- Would-bypass: `301`
- Would-evict: `659`
- Sidecar loop overhead: `83.540` us/router event
- Decision overhead: `268.470` us/miss

## Conclusion

- The trace client successfully pushed routed expert events into the HTTP sidecar integration boundary.
- The sidecar returned online WREC decisions while maintaining shadow cache state.
- This is an external runtime integration smoke, not a vLLM internal expert-loading hook.
