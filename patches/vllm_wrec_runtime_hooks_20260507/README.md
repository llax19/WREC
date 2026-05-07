# vLLM WREC Runtime Hooks

This patch package exports the local WREC changes made against vLLM `v0.19.0`.

## Contents

- `vllm_wrec_runtime_hooks.patch`: scheduler sidecar hook, layer-level residency bridge, and finite expert-slot residency files.

## Apply

```bash
git clone --branch v0.19.0 https://github.com/vllm-project/vllm.git vllm-v0.19.0
git -C vllm-v0.19.0 apply /path/to/repo/patches/vllm_wrec_runtime_hooks_20260507/vllm_wrec_runtime_hooks.patch
```

## Runtime Switches

- `WREC_SIDECAR_URL`: enables event export to the WREC sidecar.
- `WREC_RESIDENCY_MANAGER=1`: enables the layer-level offloader bridge.
- `WREC_EXPERT_RESIDENCY=1`: records expert-level WREC residency state.
- `WREC_EXPERT_RESIDENCY_SLOT_CAPACITY=<k>`: enables finite expert slots when `0 < k < global_num_experts`.
- `WREC_EXPERT_RESIDENCY_ROW_COPY=1`: opt-in full-shape row-copy skeleton; do not use as the default Mixtral CPU-offload path.

The current stable Mixtral runtime baseline is state-only sidecar integration. Finite-slot testing should start from that baseline.
