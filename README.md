# SP:35 Bridge Inspection AI — Antigravity build pack

Everything Google Antigravity's agent needs to start building the one-click SAM 3.1 inspection tool scored per IRC:SP:35-2024 Chapter 7, with the Cosmos 3 Reasoner added in Phase 3.

- Stakeholder plan (diagrams, roadmap, risks): https://claude.ai/artifact/Jnuzxbqd6zDShjnGjfhUR8
- Agent-ready plan: `docs/plan/IMPLEMENTATION_PLAN.md`
- Interpretations for the bridge engineer to sign: `docs/plan/SP35_CH7_INTERPRETATIONS.md`
- Full technical onboarding and project handoff: [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md)

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
| `docs/KNOWLEDGE_BASE.md` | Architecture, safety rules, run commands, current evidence, risks, and roadmap for new contributors/agents |
| `app/backend/` | Local FastAPI upload API, persistent single-worker SAM job queue, and review-report serving |
| `app/web/` | Next.js upload dashboard, run history, detailed statistics, and embedded plotting reports |

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

## Run it (state after the 8 Oct 2026 audit fixes)

| Task | Command (repo root) |
|---|---|
| Rebuild the GPU environment | `.\tools\setup_vision_env.ps1` (add `-Force` to replace an existing `.venv-win`) |
| Check the environment | `.venv-win\Scripts\python.exe tools\env_check.py --full` |
| Download the weights | `uv run --with huggingface_hub python tools/fetch_weights.py` (needs `HF_TOKEN` set in the shell) |
| Build the pilot set | `uv run --group ingest python -m ingest.build_pilot --src ../yugmi_images` |
| Memory probe (G0) | `.venv-win\Scripts\python.exe -m vision.probe --photos data/pilot/probe10 --label rtx2050` |
| Run SAM 3.1 on a folder | `.venv-win\Scripts\python.exe -m vision.infer --folder data/pilot/probe10 --resume` (add `--tile` for native-resolution cracks and rebar) |
| Tests | `uv run --group dev pytest` |
| Phase 0 verification | `uv run --with pyyaml python tools/verify_phase0.py` (`--strict` at the gate) |

What exists now: the SP:35 Chapter 7 engine (`sp35/`), the SAM 3.1 loader, inferencer and probe (`vision/`), the pilot builder (`ingest/`), the saved-inference visual review tool (`tools/inference_review.py`), and a local upload/analysis dashboard (`app/`). The engineer decision workflow and formal Appendix IV/V report generation remain open.

## Local inspection dashboard

The Next.js dashboard and FastAPI backend run locally. Uploading images creates a stored run; SAM does **not** start until you select the run and press **Run analysis**. The GPU worker is serialized to one job at a time. Photos and generated reports stay under the ignored `data/app_runs/` directory.

1. Install/update project dependencies and the pinned GPU worker (if not already set up):

   ```powershell
   uv sync --group dev
   .\tools\setup_vision_env.ps1
   ```

2. In terminal 1, from the repository root, start the API:

   ```powershell
   uv run python -m app.backend
   ```

3. In terminal 2, start the web UI:

   ```powershell
   Set-Location .\app\web
   npm install
   $env:NEXT_TELEMETRY_DISABLED = "1"
   npm run dev
   ```

4. Open `http://127.0.0.1:3000`. The API is at `http://127.0.0.1:8000`; interactive API docs are at `http://127.0.0.1:8000/docs`.

Uploads accept JPG/JPEG and PNG, with per-image, total-size, file-count, and image-dimension limits. Use the optional high-detail tile pass only when appropriate; it increases processing time. Each completed run exposes the plotting-agent HTML report and summary statistics. Without ground-truth annotations, those statistics describe model output and processing—not accuracy or generalization.

To use a different SAM worker Python, set `SP35_WORKER_PYTHON` before starting the API. By default, the backend uses `.venv-win\Scripts\python.exe` when present and otherwise uses its own interpreter. Do not expose the API beyond localhost; this first version is a single-user local workstation tool, not a multi-user or network-hardened service.

The dashboard provides upload, queued inference, run history, processing/detection summaries, and plotting reports. It is not yet the full engineer decision workflow: manual condition-state approval, an audit trail, and formal Appendix IV/V report generation remain future work.

## Visual triage of distress proposals

The `Distress Bifurcation` agent reviews accepted distress masks by distress type only (not component or host) and separates them into `apt`, `misclassified`, and `uncertain`. It leaves `data/pilot/infer_all_v1` unchanged and writes per-proposal review cards and manifests to `data/pilot/infer_all_v1_bifurcated/`. The `uncertain` folder prevents ambiguous masks from being mislabeled as correct or incorrect. These are visual triage labels, not ground-truth accuracy or engineer-confirmed defects.

To prepare cards and the decisions template:

```powershell
.\.venv-win\Scripts\python.exe -m tools.bifurcate_distress prepare `
  --predictions "data\pilot\infer_all_v1" `
  --images "..\yugmi_images" `
  --outdir "data\pilot\infer_all_v1_bifurcated"
```

After the agent fills every row of `decisions.csv` with `apt`, `misclassified`, or `uncertain`, plus confidence and a visual rationale, apply the decisions:

```powershell
.\.venv-win\Scripts\python.exe -m tools.bifurcate_distress apply `
  --outdir "data\pilot\infer_all_v1_bifurcated"
```
