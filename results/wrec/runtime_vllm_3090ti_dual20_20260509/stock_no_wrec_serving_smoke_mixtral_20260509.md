# Mixtral vLLM serving smoke with WREC disabled

## Inputs

- Model: `/root/WREC/models/Mixtral-8x7B-Instruct-v0.1`
- vLLM env: `/root/miniconda3/envs/vllm_moe`, vLLM `0.19.1`
- WREC env: all tested `WREC_*` variables were unset via `env -u`.
- Command shape:
  - `tensor_parallel_size=2`
  - `max_model_len=64`
  - `max_num_seqs=1`
  - `max_num_batched_tokens=64`
  - `gpu_memory_utilization=0.48`
  - `cpu_offload_gb=70`
  - `enable_return_routed_experts=True`
  - `enforce_eager=True`
  - `disable_custom_all_reduce=True`

## Result

- vLLM started loading Mixtral with `UVAOffloader`.
- Runtime log reported `Total CPU offloaded parameters: 43.25`.
- Loading reached `19/19` checkpoint shards.
- Engine startup failed after weight loading:
  - `WorkerProc was terminated`
  - `Engine core initialization failed`
  - `WorkerProc initialization failed due to an exception in a background process`
- No HTTP server became ready and no request was sent.

## Resource notes

- During load, host memory pressure became high:
  - `available` memory dropped to about `14 GiB`.
  - swap was fully used: `2.0 GiB / 2.0 GiB`.
- After failure, resources were released:
  - GPU0 `24 MiB`, GPU1 `10 MiB`.
  - no residual vLLM process.
- `dmesg` was not readable in this container, so OOM-killer evidence could not be checked.

## Conclusion

- The current Mixtral serving startup failure reproduces with WREC disabled.
- Therefore the previous state-only WREC serving smoke failure is not caused by `WREC_SIDECAR_URL` or sidecar integration.
- The likely blocking issue is current vLLM `0.19.1` + Mixtral TP2 CPU-offload startup under this host-memory/swap condition, or a difference from the older overlay-based successful path.
- The `43.25` CPU-offload value is also present without WREC, so it is not caused by WREC environment variables.
