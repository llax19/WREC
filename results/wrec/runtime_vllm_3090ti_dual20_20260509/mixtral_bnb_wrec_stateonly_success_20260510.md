# Mixtral BNB WREC state-only serving success

## Setup

- vLLM model id: `mixtral-bnb-wrec-stateonly`
- Model root: `/root/WREC/models/Mixtral-8x7B-Instruct-v0.1`
- vLLM max model len: `64`
- Runtime mode: single-GPU Mixtral bitsandbytes serving with WREC sidecar export.
- WREC finite-slot: not enabled.
- WREC expert residency: not enabled.
- Sidecar prior: `logs/processed/wrec/3090ti_dual20_20260509/train_n512.jsonl`
- Sidecar total slots: `64`

## Evidence

- `/v1/models` returned the served model `mixtral-bnb-wrec-stateonly`.
- Sidecar `/metrics` showed:
  - router events: `736`
  - expert refs: `1472`
  - shadow hits: `307`
  - shadow misses: `1165`
  - shadow miss rate: `0.7914402173913043`
  - would admit: `948`
  - would bypass: `217`
  - would evict: `948`
  - final resident: `64`
  - online loop overhead: `417.62668342391305 us/router event`
  - decision overhead: `243.368382832618 us/miss`

## Resource Snapshot

- GPU0: `19853 MiB / 24564 MiB`
- GPU1: `10 MiB / 24564 MiB`

## Conclusion

- The real vLLM serving path successfully exported routed expert events to the WREC sidecar.
- This validates the state-only runtime chain:

```text
vLLM Mixtral bitsandbytes serving
-> routed_experts capture
-> WREC sidecar /event
-> online WREC shadow decisions
```

- This run does not control real expert residency and does not validate finite-slot serving.
