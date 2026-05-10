# Mixtral BNB WREC expert-residency state-only serving success

## Setup

- vLLM model id: `mixtral-bnb-wrec-residency-stateonly`
- Model root: `/root/WREC/models/Mixtral-8x7B-Instruct-v0.1`
- vLLM max model len: `64`
- Runtime mode: single-GPU Mixtral bitsandbytes serving with WREC sidecar export.
- `WREC_EXPERT_RESIDENCY=1`: enabled.
- `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY`: not set.
- `WREC_EXPERT_RESIDENCY_ROW_COPY`: not set.
- Sidecar prior: `logs/processed/wrec/3090ti_dual20_20260509/train_n512.jsonl`
- Sidecar total slots: `64`

## Evidence

- `/v1/models` returned the served model `mixtral-bnb-wrec-residency-stateonly`.
- Sidecar `/metrics` showed:
  - router events: `1472`
  - expert refs: `2944`
  - shadow hits: `614`
  - shadow misses: `2330`
  - shadow miss rate: `0.7914402173913043`
  - would admit: `1896`
  - would bypass: `434`
  - would evict: `1896`
  - final resident: `64`
  - online loop overhead: `417.76909578804344 us/router event`
  - decision overhead: `243.3888364806867 us/miss`

## Resource Snapshot

- GPU0: `19853 MiB / 24564 MiB`
- GPU1: `10 MiB / 24564 MiB`

## Conclusion

- Real vLLM Mixtral bitsandbytes serving works with `WREC_EXPERT_RESIDENCY=1` in state-only mode.
- This validates that the expert-residency manager can be enabled without slot capacity or row-copy and without breaking the serving path.
- This run still does not control real expert residency, does not replace expert tensors, and does not validate finite-slot serving.
