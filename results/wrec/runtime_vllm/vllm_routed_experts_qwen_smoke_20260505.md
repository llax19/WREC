# vLLM Routed Experts Capture

## Inputs

- Model: `/root/workspace/qwen1.5-MoE-A2.7B`
- Prompt: `请用一句话解释什么是混合专家模型。`
- Max tokens: `1`
- Max model len: `512`

## vLLM Output

- Finish reason: `length`
- Generated token count: `1`
- Generated text: `你`

## Routed Experts

- Shape: `[9, 24, 4]`
- Seq len: `9`
- Layers: `24`
- Top-k: `4`
- Router events exported: `216`
- Expert refs exported: `864`
- Events JSONL: `/root/workspace/results/wrec/runtime_vllm/vllm_routed_experts_qwen_events_20260505.jsonl`

## Conclusion

- vLLM internal routed expert capture is available through `enable_return_routed_experts=True`.
- The captured routed experts were converted into WREC runtime sidecar event format.
- This verifies the internal event-source path; it does not yet control vLLM expert residency or loading.
