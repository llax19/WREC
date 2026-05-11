# Qwen finite-slot short eval n=8 rerun after sidecar client fix

- Date: `2026-05-11`
- Model: `/root/WREC/models/qwen1.5-MoE-A2.7B`
- Runtime mode: `finite-slot`
- Slot capacity: `16`
- Max num batched tokens: `4`
- Sidecar URL: `http://127.0.0.1:18765/event`
- `WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0`

## Request Summary

- Requests completed: `8`
- Mean latency: `2.235494 s`
- Max latency: `3.565797 s`
- Input tokens: `117`
- Output tokens: `8`
- Per-request outputs: `results/wrec/runtime_finite_slot_qwen_single_gpu_20260511/qwen_finite_slot_short_eval_n8_rerun_after_sidecar_fix_20260511.json`

## Sidecar Metrics

- Router events: `1`
- Expert refs: `4`
- Shadow hits: `1`
- Shadow misses: `3`
- Would admit: `3`
- Would evict: `3`

## Observation

- Routed expert export count did not recover in this rerun; sidecar metrics still stopped at the first router event.
- This time the cause is explicit in the live vLLM log instead of being silent:
  - `WREC sidecar bridge failed`
  - `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`
- The failure happens after the first sidecar response reaches the finite-slot residency manager:
  - `record_sidecar_response()` calls `_evict_expert()` when `would_evict` is present.
  - `_evict_expert()` writes `state.expert_map[expert_id] = -1`.
- After that exception, the sidecar bridge follows fail-open behavior and disables itself, so later routed expert events are no longer exported.

## Conclusion

- The earlier sidecar client reconnect patch is not the active blocker for this Qwen finite-slot rerun.
- The current blocker is the finite-slot eviction path mutating `expert_map` as an inference tensor during sidecar response handling.
- Next step should fix `wrec_expert_residency.py` eviction/update semantics before using finite-slot sidecar counts as complete trace statistics.
