# Qwen UVA Backing Reuse Smoke

## Setup

- Model: `/root/WREC/models/qwen1.5-MoE-A2.7B`
- Server: vLLM OpenAI endpoint on `127.0.0.1:18002`
- Sidecar: WREC runtime sidecar on `127.0.0.1:8766`
- Offload: `--offload-backend uva`, `--cpu-offload-params experts.w13_weight experts.w2_weight`
- WREC: `WREC_EXPERT_RESIDENCY=1`, `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY=32`
- Request manifest: `results/wrec/runtime_selective_prefetch_smoke_short_20260515T/smoke_two_short_requests.jsonl`

## Outputs

- Request log: `logs/raw/qwen_uva_reuse_smoke_20260515T/smoke_two_short_requests.log.jsonl`
- WREC residency stats: `logs/processed/qwen_uva_reuse_smoke_20260515T/wrec_residency_stats.jsonl`
- Sidecar log: `logs/server/qwen_uva_reuse_smoke_20260515T/sidecar.log`

## Results

- Requests completed: `2`
- TTFT: `1211.44 ms`, `1196.10 ms`
- Output tokens: `1`, `1`
- WREC stats rows: `96`
- Layers observed: `24`
- Fallback rows: `0`
- Rows with prefetch hit: `24`
- Average prefetch hit rate: `0.0456`
- Max prefetch hit rate: `0.3333`
- Sidecar router events: `240`
- Sidecar expert refs: `960`
- Sidecar shadow hit rate: `0.1240`

## Conclusion

The smoke completed successfully. During vLLM startup, WREC logged reuse of
`_vllm_cpu_offload_storage` for both `w13_weight` and `w2_weight` across Qwen
MoE layers, and no WREC fallback rows were recorded in `wrec_residency_stats`.

One logging caveat: after a background launch failed due `vllm` not being on
the non-interactive shell `PATH`, the successful server was launched in the
foreground session, so the complete vLLM stdout was observed but not captured
in `server.log`.
