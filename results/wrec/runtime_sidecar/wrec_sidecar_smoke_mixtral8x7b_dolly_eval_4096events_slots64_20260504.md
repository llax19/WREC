# WREC Runtime Sidecar Smoke

## Inputs

- Trace: `/root/workspace/logs/processed/wrec/mixtral8x7b_dolly_eval_router_events_n256_mem48_20260501.jsonl`
- Sidecar URL: `http://127.0.0.1:8765`
- Max events: `4096`

## Client

- Events sent: `4096`
- Expert refs sent: `8192`
- Client elapsed: `1.715408` s
- Mean client submit time: `418.801` us/event

## Sidecar Metrics

- Expert refs processed: `8192`
- Router events processed: `4096`
- Shadow hits: `7237`
- Shadow misses: `955`
- Shadow miss rate: `0.116577148`
- Would-admit: `651`
- Would-bypass: `304`
- Would-evict: `651`
- Sidecar loop overhead: `60.651` us/router event
- Decision overhead: `189.287` us/miss

## Conclusion

- The trace client successfully pushed routed expert events into the HTTP sidecar integration boundary.
- The sidecar returned online WREC decisions while maintaining shadow cache state.
- This is an external runtime integration smoke, not a vLLM internal expert-loading hook.
