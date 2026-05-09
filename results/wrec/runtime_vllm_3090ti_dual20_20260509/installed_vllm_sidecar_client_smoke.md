# Installed vLLM WREC Sidecar Client Smoke

- Routed experts shape: `[2, 2, 2]`
- Events delta: `4`
- Expert refs delta: `8`
- Client enabled after submit: `True`

Conclusion: installed vLLM WREC sidecar client can submit routed expert events to the state-only sidecar. This does not run model serving or control expert residency.
