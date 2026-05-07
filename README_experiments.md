# Experiment Workspace Notes

This workspace is prepared for the MoE scheduling experiments described in
[experiment_guide.md](/root/workspace/experiment_guide.md).

## Directory layout

- `data/prompts/`: request inputs and prompt manifests
- `scripts/`: helper scripts grouped by experiment phase
- `logs/raw/`: raw per-request logs dumped by the serving pipeline
- `logs/processed/`: cleaned or merged logs ready for plotting
- `results/`: metric summaries and exported tables
- `figures/`: plots for the thesis
- `notes/`: ad hoc experiment notes

## Immediate next steps

1. Prepare a first `jsonl` prompt set in `data/prompts/`.
2. Keep request-level serving logs in `logs/raw/`.
3. Use `scripts/analysis/analyze_logs.py` to compute throughput, TTFT, and TPOT.
4. Move stable summaries into `results/` and plots into `figures/`.

## Scheduling implementation boundary

The current `length_only` mode in `scripts/runtime/log_vllm_requests.py` is a
client-side proxy scheduler: it reorders requests before submitting them to the
vLLM OpenAI-compatible server. This is useful for checking whether the workload,
logging path, and metrics are sensitive to request ordering, but it is not the
final expected system shape.

Final method results should come from a server-side implementation integrated
into vLLM, for example through a local package, patch, or scheduling module that
participates in request queueing, batching, or admission inside the serving
process. The current `vllm_moe` environment now includes a server-side
`--scheduling-policy length_only` path for that purpose. It also includes a
server-side `--scheduling-policy ltr` path that consumes external predictor
scores from `VLLM_LTR_SCORE_FILE`; this is the intended integration point for
paper-style learning-to-rank scheduling once predictor scores are available.

## Starter scripts

- `scripts/runtime/check_vllm_env.py`
  - Print Python, torch, transformers, and vLLM availability for the current
    environment.
- `scripts/runtime/vllm_smoke_load.py`
  - Minimal local-model generation check once `vLLM` is installed.
- `scripts/runtime/log_vllm_requests.py`
  - Send prompts from a JSONL manifest to a running vLLM OpenAI-compatible
    server and save request-level timing logs.
- `scripts/data_prep/check_requests.py`
  - Validate prompt manifests before running experiments.
- `scripts/analysis/analyze_logs.py`
  - Aggregate request logs into throughput, TTFT, and TPOT metrics.
- `scripts/runners/run_vllm_smoke_tp2.sh`
  - Conservative dual-GPU tensor-parallel smoke test wrapper.
- `scripts/runners/run_vllm_server_tp2.sh`
  - Conservative dual-GPU tensor-parallel vLLM server wrapper.
- `scripts/runners/run_tp2_fcfs_formal.sh`
  - Formal TP=2 FCFS batch experiment entrypoint.
- `scripts/runners/run_tp2_length_only_server_formal.sh`
  - Formal TP=2 server-side Length-only batch experiment entrypoint.
- `scripts/runners/run_tp2_ltr_server_formal.sh`
  - Formal TP=2 server-side LTR batch experiment entrypoint.
- `scripts/ltr/build_ltr_score_file.py`
  - Build a scheduler score file for LTR plumbing tests or predictor outputs.
- `scripts/ltr/train_ltr_lite_predictor.py`
  - Train a dependency-light, explainable LTR-lite ridge predictor from
    request manifests and existing raw FCFS logs, then export
    `VLLM_LTR_SCORE_FILE`-compatible scores.
- `scripts/runners/run_tp2_length_only_formal.sh`
  - Client-side proxy Length-only batch experiment entrypoint for preliminary checks.
- `scripts/runtime/scan_nvidia_users.py`
  - Best-effort `/proc`-based replacement for `fuser -v /dev/nvidia*`.

## Suggested command sequence after vLLM installation

1. Verify the environment:

```bash
conda activate vllm_moe
python scripts/runtime/check_vllm_env.py
```

2. Try a direct local-model smoke test:

```bash
python scripts/runtime/vllm_smoke_load.py --model /root/workspace/qwen1.5-MoE-A2.7B
```

If you want to start directly with dual-GPU tensor parallel and both cards are
available, use:

```bash
bash scripts/runners/run_vllm_smoke_tp2.sh
```

3. If you start the OpenAI-compatible server separately, log requests with:

```bash
python scripts/runtime/log_vllm_requests.py \
  --request-file data/prompts/requests_template.jsonl \
  --output-log logs/raw/debug_run.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model-name /root/workspace/qwen1.5-MoE-A2.7B \
  --strategy fcfs \
  --load-level debug \
  --tokenizer-path /root/workspace/qwen1.5-MoE-A2.7B
```

4. Summarize the resulting logs:

```bash
python scripts/analysis/analyze_logs.py logs/raw/debug_run.jsonl
```

5. If you need a quick container-local scan of visible NVIDIA device users and
   `fuser` is unavailable:

```bash
python scripts/runtime/scan_nvidia_users.py
```

## Notes for dual-GPU tensor parallel

- Dual-GPU tensor parallel requires both selected GPUs to be mostly free.
- In this workspace, `GPU 0` has previously shown heavy memory usage even when
  `nvidia-smi` did not list a visible process. If that condition remains, the
  TP=2 smoke test will still fail because rank 0 cannot reserve memory.
- The provided TP=2 wrappers use conservative defaults:
  - `tensor_parallel_size=2`
  - `gpu_memory_utilization=0.70`
  - `cpu_offload_gb=8`
  - `max_model_len=1024`
- Once the model loads successfully, these values can be increased step by
  step.

## Current TP=2 Load Levels

After the 2026-04-24 server-side scheduler calibration, the recommended TP=2
load levels are:

- `low`: `arrival_interval=8.0`, `max_concurrency=1`
- `medium`: `arrival_interval=0.5`, `max_concurrency=4`
- `high`: `arrival_interval=0.0`, `max_concurrency=6`

The previous `medium` setting (`arrival_interval=1.5`, `max_concurrency=2`)
was too light and did not create enough server-side queueing to expose
scheduling-policy differences.

## Expected raw log schema

Each line in a raw log file should be a JSON object with at least:

```json
{
  "request_id": "req-0001",
  "strategy": "length_only",
  "load_level": "medium",
  "submit_time": 0.0,
  "start_time": 0.01,
  "first_token_time": 0.25,
  "finish_time": 1.70,
  "input_tokens": 128,
  "output_tokens": 96
}
```

Field meanings:

- `submit_time`: request arrival timestamp
- `start_time`: request admitted into execution
- `first_token_time`: first output token timestamp
- `finish_time`: request completion timestamp
- `input_tokens`: prompt length after tokenization
- `output_tokens`: generated token count

The schema is intentionally simple so it can be produced by either a direct
Python client, a vLLM OpenAI-compatible client, or a custom scheduler wrapper.
