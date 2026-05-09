# WREC Finite-Slot CPU Preflight

- Slot capacity: `2`
- Slot shape after install: `[2, 3, 2]`
- Resident experts: `[1, 2]`
- Logical-to-slot: `{1: 0, 2: 1}`
- Slot-to-logical: `[1, 2]`
- Expert map: `[-1, 0, 1, -1]`
- Overflow guard triggered: `True`

Conclusion: finite-slot mapping and overflow guard work on a minimal CPU fake layer. This is only a preflight before any real Mixtral/vLLM serving attempt.
