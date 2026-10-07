---
trigger: model_decision
description: Apply when creating or editing anything under kb/ or sp35/, or any code that computes condition states, CS_j, CI_i, BHI, star ratings, critical lists or MPI.
---

# SP:35 fidelity rules

Source: IRC:SP:35-2024 Chapter 7 (book pp. 101–144) and Table 5.1 (pp. 89–91).
Scan offsets in our PDF: PDF page = book page + 16 up to p.116; book p.117 is missing; PDF page = book page + 15 from p.118.

## Equations — implement exactly
- Eq. 3: `CS_j = Σ_k (100 − 100·S_k) / d` over the d distress types NOTICED on sub-component j. Nothing noticed, or element absent (NA) → `CS_j = 100`.
- Eq. 2: `CI_i = Σ_j W_j·CS_j / Σ_j W_j` (W_j = Table 7.1 column 5). Divide by the actual sum of the weights present (E2, E6).
- Eq. 1: `BHI = Σ_i W_i·CI_i / Σ_i W_i` (W_i = Table 7.1 column 3).
- Bridge BHI = minimum span BHI. End spans include the abutments; intermediate spans include the piers.
- Eq. 4: `MPI = Σ_i W_i·FR_i / Σ_i W_i`; Eq. 5: `FR_i = Σ_j W_j·PR_j / Σ_j W_j`; PR ∈ {0, 25, 50, 75, 100} per Table 7.8.
- S_k: I = 0, II = 0.25, III = 0.5, IV = 0.75, V = 1.0 (from `condition_states.yaml`).

## Rules
- Measurement → condition state uses the interval bands in `kb/sp35_2024/distress_*.yaml`. Keep `printed` strings for display. Rows with two criteria (`criteria:` list) take the WORSE state.
- A band of `null` means that state cannot occur for this distress (e.g. crushing has no state II).
- Star band and the MPI bridge-condition mapping come from `condition_states.yaml` and `mpi.yaml` (decisions E8, E9).
- Critical list = every sub-component with any distress in state V; urgent list = state IV. Both independent of BHI.
- Foundations not visible: require an engineer judgement entry (state + reason). Never default silently.
- Never use Table 7.1 column 6 (misprints, E5).
- Every number in a report must be reproducible from stored inputs plus the KB version hash.

## Tests that must stay green
`kb/sp35_2024/golden_examples.yaml`, cross-checked by `tools/sp35_reference_check.py`.
