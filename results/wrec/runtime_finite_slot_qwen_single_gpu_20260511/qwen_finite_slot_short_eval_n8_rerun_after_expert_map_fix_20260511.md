# Qwen finite-slot short eval n=8 rerun after expert_map fix

- Date: `2026-05-11`
- Model: `/root/WREC/models/qwen1.5-MoE-A2.7B`
- Runtime mode: `finite-slot`
- Slot capacity: `16`
- Max num batched tokens: `4`
- Sidecar URL: `http://127.0.0.1:18765/event`
- `WREC_SIDECAR_MAX_EVENTS_PER_REQUEST=0`

## Request Summary

- Requests completed: `8`
- Mean latency: `3.669668 s`
- Max latency: `5.298327 s`
- Input tokens: `117`
- Output tokens: `8`
- Per-request outputs: `results/wrec/runtime_finite_slot_qwen_single_gpu_20260511/qwen_finite_slot_short_eval_n8_rerun_after_expert_map_fix_20260511.json`

## Sidecar Metrics

- Router events: `2808`
- Expert refs: `11232`
- Shadow hits: `2670`
- Shadow misses: `8562`
- Would admit: `8043`
- Would bypass: `519`
- Would evict: `8043`

## Observation

- After patching finite-slot `expert_map` creation and updates to execute outside `InferenceMode`, routed expert export no longer stops after the first sidecar response.
- The sidecar totals now match the earlier state-only short-eval run on the same 8 prompts:
  - `router_events=2808`
  - `expert_refs=11232`
- This means the finite-slot path can now complete the full `vLLM request -> routed expert export -> WREC sidecar decision` counting loop without being disabled by the earlier `InferenceMode` mutation error.

## Conclusion

- The blocking finite-slot undercount bug has been fixed for this guarded Qwen short-eval setting.
- For this exact configuration, finite-slot sidecar metrics are now complete enough to support larger-scale follow-up experiments.
- The remaining known issue is separate: WREC still falls back to an extra CPU copy instead of reusing vLLM offload storage.
