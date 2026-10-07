---
trigger: model_decision
description: Apply when finishing a task, preparing a merge, or when the user mentions a gate (G0–G6), a demo, acceptance or a release.
---

# Testing and gates

- `uv run pytest -q` must pass. Changes in `sp35/` need unit tests and property tests (hypothesis):
  - BHI always within [0, 100];
  - making any distress state worse never raises CS_j, CI_i or BHI;
  - adding an NA element never lowers BHI;
  - the bridge BHI equals the minimum span BHI.
- Golden tests (values and tolerances in `kb/sp35_2024/golden_examples.yaml`):
  Table 7.6 BHI 95.06 ± 0.2 · pier cap 68.75 exact · p.130 deck slab 75.0 (book prints 50.0, E1) · Table 7.9 MPI 44.89 / 33.21 ± 0.2.
- Never change an expected value to make a test pass. If the book is wrong, add an E-entry to `docs/plan/SP35_CH7_INTERPRETATIONS.md` and ask for sign-off.
- Vision changes (from Phase 2): run the evaluation harness and attach the per-class report; block the change if any critical class drops by more than 5 points.
- Each gate needs the acceptance checklist from `docs/plan/IMPLEMENTATION_PLAN.md` with evidence links, open risks and decisions requested. Use the `phase-gate-review` skill.
