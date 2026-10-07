# SP:35 Bridge Inspection AI — Antigravity build pack

Everything Google Antigravity's agent needs to start building the one-click SAM 3.1 inspection tool scored per IRC:SP:35-2024 Chapter 7, with the Cosmos 3 Reasoner added in Phase 3.

- Stakeholder plan (diagrams, roadmap, risks): https://claude.ai/artifact/Jnuzxbqd6zDShjnGjfhUR8
- Agent-ready plan: `docs/plan/IMPLEMENTATION_PLAN.md`
- Interpretations for the bridge engineer to sign: `docs/plan/SP35_CH7_INTERPRETATIONS.md`

## What's inside

| Path | Purpose |
|---|---|
| `AGENTS.md` | Always-on rules: mission, non-negotiables, stack, layout |
| `.agents/rules/sp35-fidelity.md` | Equations and scoring rules (triggers on KB / engine work) |
| `.agents/rules/vision-gpu.md` | SAM 3.1 API, VRAM budget, prompt rules (triggers on `vision/**`) |
| `.agents/rules/testing-and-gates.md` | Tests, golden values, gate evidence |
| `.agents/skills/*/SKILL.md` | `/sp35-golden-tests`, `/vram-probe`, `/gpu-batch-day`, `/phase-gate-review` |
| `kb/sp35_2024/*.yaml` | Chapter 7 encoded: Table 7.1 weights, Tables 7.2/7.3 bands, condition states and stars, MPI Tables 7.7–7.9, golden examples |
| `kb/prompts/sam3_prompts.yaml` | SAM 3.1 prompt library v0 mapped to SP:35 codes |
| `tools/sp35_reference_check.py` | Independent oracle; reproduces the book's worked examples and checks the YAML |

Antigravity's legacy Workflows stop working on 19 Oct 2026, so repeatable procedures are Skills under `.agents/skills/`.

## Start

1. Open this folder as an Antigravity workspace.
2. Check the oracle runs: `uv run --with pyyaml python tools/sp35_reference_check.py` → `ALL PASS`.
3. Start an agent conversation in Planning mode with:

```
You are building "SP:35 Bridge Inspection AI".
Read AGENTS.md, docs/plan/IMPLEMENTATION_PLAN.md and docs/plan/SP35_CH7_INTERPRETATIONS.md first.
Execute Phase 0 only (tasks 0.1 to 0.6), in order.
For each task, produce an implementation plan artifact and wait for my approval before writing code.
Use SP:35 values only from kb/sp35_2024; if a value you need is missing, stop and ask me.
Never commit tokens or model weights.
Finish by running the sp35-golden-tests skill, then prepare the G0 review with the phase-gate-review skill.
```

4. Review each implementation-plan artifact before the agent writes code. Reject any plan that introduces an SP:35 value not in `kb/`.

## Things only a person can do
- Accept the SAM License for `facebook/sam3.1` on Hugging Face and create a read token (keep it out of the repo).
- Sign the interpretations register (bridge engineer).
- Book GPU days on the company RTX 3060 and RTX 5070 Ti for Phase 2.
