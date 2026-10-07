# SP:35 Bridge Inspection AI — agent rules

## Mission
A Windows desktop app that turns a folder of bridge inspection photos into a draft IRC:SP:35-2024 inspection report.
SAM 3.1 segments components and distresses. A deterministic engine applies SP:35 Chapter 7 (Tables 7.1–7.3, Eq. 1–5) to produce condition states, BHI per span and bridge, star rating, critical list and MPI.
A bridge engineer confirms every condition state and signs the Appendix V certificate.
NVIDIA Cosmos 3 Reasoner is added in Phase 3 behind the `Reasoner` contract (Phase 2 defines it).

## Non-negotiables
1. SP:35 numbers come only from `kb/sp35_2024/*.yaml`. Never type a weight, range, S_k value or star band into code. If a value is missing, stop and ask.
2. Follow the equations, not the book's printed arithmetic. Known misprints and decisions: `docs/plan/SP35_CH7_INTERPRETATIONS.md` (E1–E13).
3. The AI proposes; the engineer decides. Never finalise a condition state automatically. Every sub-component with a state-V distress appears in the critical list, whatever the BHI.
4. Never invent measurements. A mm/m/m² value exists only if measured (scale card, gauge, laser) or typed by the engineer. Every value records its source.
5. Keep inventory data and inspection data in separate tables (SP:35 §6.5).
6. Photos stay on the machine by default. Any cloud call (e.g. hosted NIM) needs an explicit per-project consent flag.
7. Never commit tokens, model weights, client photos or `.env` files.
8. Desk app on the RTX 2050 laptop: peak VRAM ≤ 3.4 GB.

## Stack
Python 3.12 · uv · PyTorch 2.7+ (CUDA 12.8 builds) · official `sam3` repo (`build_sam3_image_model`, `Sam3Processor`) with checkpoints from Hugging Face `facebook/sam3.1` · FastAPI · SQLite · pydantic · pytest + hypothesis.
GPU worker: WSL2 Ubuntu 24.04 on the laptop; Docker (CUDA 12.8) on the RTX 3060 and RTX 5070 Ti.

## Layout
- `kb/` — SP:35 knowledge base (YAML, versioned) and SAM 3.1 prompt library
- `sp35/` — scoring engine (pure Python, no GPU)
- `vision/` — SAM 3.1 worker
- `app/` — review app and reports
- `tools/sp35_reference_check.py` — independent oracle for the book's worked examples
- `docs/plan/` — the plan; read the current phase before starting

## Working agreement
- Planning mode: write an implementation plan artifact per work package and wait for approval.
- Tests first for anything in `sp35/`. Run the `sp35-golden-tests` skill before finishing any task touching `kb/` or `sp35/`.
- One work package per branch, small commits. Tick the WP checkboxes in `docs/plan/IMPLEMENTATION_PLAN.md` when done.
- Stop at each gate (G0, G1, G2 …) and run the `phase-gate-review` skill.
