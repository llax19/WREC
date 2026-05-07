# vLLM Policy Scheduler Patch Bundle

This directory preserves the vLLM 0.19.1 files modified for the server-side
policy scheduler experiment on 2026-04-24.

## Target Environment

- Conda env: `vllm_moe`
- vLLM version: `0.19.1`
- Target package path:
  `/root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm`

## Modified Files

- `vllm/v1/core/sched/policy.py`
  - Adds `RequestFeatures`
  - Adds `FCFSPolicy`
  - Adds `PriorityPolicy`
  - Adds `LengthOnlyPolicy`
  - Adds `LTRPolicy`, which reads external predictor scores from
    `VLLM_LTR_SCORE_FILE`
- `vllm/v1/core/sched/request_queue.py`
  - Adds `SchedulingPolicy.LENGTH_ONLY`
  - Adds `SchedulingPolicy.LTR`
  - Adds a policy-driven heap queue
  - Adds server-side `LengthOnlyRequestQueue`
  - Adds server-side `LTRRequestQueue`
- `vllm/v1/core/sched/scheduler.py`
  - Loads the policy plugin in the scheduler
  - Compares queue heads by policy key when choosing between waiting queues
- `vllm/config/scheduler.py`
  - Allows `--scheduling-policy length_only`
  - Allows `--scheduling-policy ltr`

## Restore / Apply

From `/root/workspace`, copy this overlay into the active conda environment:

```bash
cp -r patches/vllm_policy_scheduler_20260424/site-packages/vllm/* \
  /root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/
```

Then verify:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_moe
python -m py_compile \
  /root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/policy.py \
  /root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/request_queue.py \
  /root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py \
  /root/miniconda3/envs/vllm_moe/lib/python3.10/site-packages/vllm/config/scheduler.py
vllm serve --help=SchedulerConfig | rg -A7 -- '--scheduling-policy'
```

Expected CLI option:

```text
--scheduling-policy {fcfs,length_only,ltr,priority}
```

## Experiment Entry Point

Use the server-side Length-only script:

```bash
bash scripts/runners/run_tp2_length_only_server_formal.sh
```

Use the server-side LTR script with a predictor score file:

```bash
VLLM_LTR_SCORE_FILE=/root/workspace/logs/processed/ltr_scores.jsonl \
  bash scripts/runners/run_tp2_ltr_server_formal.sh
```

The older client-side proxy script is still available as a preliminary check:

```bash
bash scripts/runners/run_tp2_length_only_formal.sh
```
