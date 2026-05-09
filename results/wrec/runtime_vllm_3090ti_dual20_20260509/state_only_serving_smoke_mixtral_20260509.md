# WREC state-only vLLM serving smoke

## Inputs

- Model: `/root/WREC/models/Mixtral-8x7B-Instruct-v0.1`
- vLLM env: `/root/miniconda3/envs/vllm_moe`, vLLM `0.19.1`
- Sidecar prior: `logs/processed/wrec/3090ti_dual20_20260509/train_n512.jsonl`
- Sidecar slots: `64`
- WREC mode: `WREC_SIDECAR_URL` only; `WREC_EXPERT_RESIDENCY` and finite-slot capacity were not enabled.

## Attempt 1

- Command shape: `vllm serve`, TP `2`, `max_model_len=512`, `gpu_memory_utilization=0.82`, `cpu_offload_gb=50`, `enable_return_routed_experts=True`, `enforce_eager=True`.
- Result: failed after all `19/19` Mixtral shards loaded.
- Error boundary: `Engine core initialization failed`, `WorkerProc initialization failed due to an exception in a background process`.
- Sidecar events: `0`.

## Attempt 2

- Command shape: `vllm serve`, TP `2`, `max_model_len=64`, `max_num_seqs=1`, `max_num_batched_tokens=64`, `gpu_memory_utilization=0.48`, `cpu_offload_gb=70`, `enable_return_routed_experts=True`, `enforce_eager=True`, `disable_custom_all_reduce=True`.
- Result: failed after all `19/19` Mixtral shards loaded.
- Error boundary: same `Engine core initialization failed` / `WorkerProc initialization failed`.
- Sidecar events: `0`.

## Observations

- Both failures happened before the HTTP server became ready and before any request was sent.
- GPU memory was released after failure: GPU0 `24 MiB`, GPU1 `10 MiB`.
- Host memory was available after failure: about `119 GiB` available; swap was fully used.
- `dmesg` could not be read in this container, so OOM-killer evidence could not be confirmed.
- The conservative retry reused the old successful mem48/offload smoke shape, but current vLLM `0.19.1` logged `Total CPU offloaded parameters: 43.25`, while the older record reported `70.28`.

## Conclusion

- This run did not complete the real Mixtral state-only serving smoke.
- The failure is a vLLM/Mixtral engine startup issue before sidecar integration is exercised.
- Current validated runtime evidence remains: HTTP sidecar smoke, installed `WrecSidecarClient` smoke, and finite-slot CPU fake-layer preflight.
- Next debugging step should isolate current site-packages vLLM startup from WREC sidecar by running the same conservative Mixtral command with all WREC env vars unset, then compare with the older overlay-based successful path.
