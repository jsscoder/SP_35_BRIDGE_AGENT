# SP:35 Bridge Inspection AI — Project Knowledge Base

**Purpose:** a durable onboarding and handoff document for engineers and AI agents joining this repository.  
**Snapshot:** 9 October 2026. Treat the status notes as a dated snapshot, not as a substitute for checking the current worktree, hardware, datasets, gates, and upstream model requirements.

## 1. Product and safety boundary

The project is a Windows-first, local-processing workflow that turns bridge inspection photographs into a **draft** IRC:SP:35-2024 inspection record:

1. Organize and deduplicate photographs.
2. Use SAM 3.1 to propose component and distress masks.
3. Let a reviewer inspect and correct proposals and enter measured values or engineer judgements.
4. Use the deterministic SP:35 engine to calculate condition scores, BHI, star rating, critical/urgent lists, and MPI.
5. Produce a report for an engineer to review and sign.

The AI proposes; the bridge engineer decides. The repository does not yet provide the complete desktop review-to-signed-report workflow. Inference detections are not condition states, and a generated overlay is not evidence of accuracy.

### Non-negotiable project rules

- All SP:35 weights, condition bands, state coefficients, star bands, MPI classes, and related numeric source values come from `kb/sp35_2024/*.yaml`. Never duplicate them in application code.
- Follow the decisions in [SP35_CH7_INTERPRETATIONS.md](./plan/SP35_CH7_INTERPRETATIONS.md), not erroneous printed arithmetic. Its draft/sign-off status must be checked before a gate.
- Do not automatically finalize a condition state. Every sub-component with a state-V distress belongs in the critical list, whatever the BHI.
- Do not invent measurements. A physical value must have been measured with an appropriate tool or entered by an engineer, and must retain its source/method.
- Keep inventory data separate from inspection data.
- Keep photographs on the machine by default. Any cloud processing requires explicit per-project consent.
- Never commit Hugging Face tokens, `.env` files, client photographs, model weights, or other credentials.
- The laptop GPU process is capped to stay within the documented 3.4 GB VRAM budget. Do not increase input sizes or alter precision without measurement.

## 2. Repository orientation

| Location | Responsibility |
|---|---|
| `AGENTS.md` | Short project rules and everyday commands; read before changing code. |
| `README.md` | Project entry point and setup overview. |
| `docs/plan/IMPLEMENTATION_PLAN.md` | Planned work packages, phase scope, gates, and acceptance criteria. |
| `docs/plan/SP35_CH7_INTERPRETATIONS.md` | Register of interpretation decisions E1 onward; source for arithmetic decisions. |
| `docs/reports/runtime_decision.md` | Windows GPU environment/model pairing decisions and measured runtime evidence. |
| `docs/gates/` | Dated gate evidence. Gate documents can become stale; verify their facts. |
| `kb/sp35_2024/` | Sole runtime authority for SP:35 tables, component weights, distress bands, state data, MPI data, and golden examples. |
| `kb/prompts/sam3_prompts.yaml` | SAM prompt names, SP:35 references, thresholds, area floors, tiles, and suppression settings. These prompt settings are hypotheses until evaluated against labels. |
| `sp35/` | CPU-only deterministic KB loader, condition-state mapping, scoring, bridge roll-up, and MPI logic. |
| `vision/model.py` | Supported checkpoint verification, SAM model loader, precision handling, CUDA cap, and detector helper. |
| `vision/infer.py` | Single-photo/folder/manifest inference, resume, tiling, per-photo serialization, and run summary. |
| `vision/postprocess.py` | Detection filtering, synonym/tile merging, host checks, suppression, overlap handling, RLE records, and inference overlay rendering. |
| `ingest/build_pilot.py` | Pilot photo scanning, deduplication, metadata, tags, and ingest manifest. |
| `tools/inference_review.py` | CPU-side visual report from saved inference JSON; optional COCO comparison. It does not run SAM. |
| `.github/agents/inference-review.agent.md` | Repository custom agent instructions for inspecting inference and producing visual reviews. |
| `tools/sp35_reference_check.py` | Independent worked-example oracle, separate from the engine/KB implementation. |
| `tests/` | Unit, golden, property, vision, ingest, and review-tool tests. |
| `requirements/vision-win.txt` | Pinned native Windows CUDA/SAM worker environment. |
| `tools/setup_vision_env.ps1`, `tools/env_check.py` | Build and verify `.venv-win`. |
| `Dockerfile` | CPU/reference-check and Linux CUDA worker image targets; the GPU target mounts data/weights at runtime. |
| `data/` | Local datasets, generated inference/review outputs, and model weights. Treat as local/private data, not distributable source. |
| `app/backend/` | Local FastAPI photo-upload API, persistent queued runs, SAM subprocess integration, and plotting-report serving. |
| `app/web/` | Next.js local upload dashboard, job history, analysis stats, and embedded plotting report. |

Use repository-relative paths in commands from the repository root. On Windows PowerShell, use backslashes in filesystem paths.

## 3. End-to-end pipeline

### 3.1 Photo ingest

`ingest/build_pilot.py` scans source folders, records hashes and metadata, identifies duplicate/conversion copies, retains a selected original, and writes a manifest and exclusions. In the pilot process recorded in the plan, 383 source rows were reduced to 167 kept photographs. Ingest tags such as component/span hints are not engineer-confirmed inventory records.

Still-open ingest work listed by the plan includes a complete EXIF/DJI XMP capture, thumbnails, app-backed span/support tagging, and blur checks for faces and number plates. Check the current code and plan before promising these are complete.

### 3.2 SAM 3.1 inference

`python -m vision.infer` loads the prompt library and model, processes each selected photo, and writes per-photo output. A normal run:

- Runs component prompts first to find host regions.
- Runs distress/suppressor prompts and associates distress proposals with detected or tagged hosts.
- Applies thresholds and minimum area, merges duplicate/synonym masks, removes suppression regions, rejects distress masks without a host, and resolves cross-class duplicates.
- Can optionally process selected classes in native-resolution tiles. A tiled result is still a model proposal.
- Saves masks as COCO-style RLE, boxes as XYXY bounds, scores, host/provenance details, rejection reasons, timings, and model/prompt metadata.

The tagging fallback can use a supplied `sp35_hint` host when SAM misses the component. It is a useful close-up-photo aid, not independent confirmation that the component or distress is correct.

### 3.3 Visual review and ground-truth comparison

`tools/inference_review.py` decodes saved masks and creates local review panels, an HTML index, and summary statistics. Accepted components and distress proposals use distinct boxes; suppressed/rejected detections are not rendered as accepted detections. Its SP:35 coverage section reads the Table 7.2/7.3 distress catalogs and prompt-to-row mappings from YAML, then reports prompted rows with accepted proposals and proposal counts by row. This is catalog/prompt coverage only: without labels, it cannot establish defect prevalence, recall, or accuracy. It reads inference files and images and does not load the GPU model.

When a compatible COCO annotation file is supplied, the utility performs score-ordered, greedy, one-to-one matching by image and class at a chosen mask-IoU threshold. Its precision/recall/matched-IoU summary is a limited diagnostic, **not** COCO mask AP, SP:35 condition-score validation, or a substitute for a held-out bridge evaluation.

Without ground truth the report is visual-only. It cannot show that SAM generalizes, is accurate, or is not overfitting.

### 3.4 Engineer review and scoring

The intended system flow is: detection proposal → engineer-confirmed component/distress → measurement or explicit engineer judgement → mapped condition state with a rule trace → score roll-up → engineer-approved report/certificate.

The scoring engine is deterministic and consumes approved condition-state observations. It must not infer a state directly from a SAM confidence score or mask area. A visible area estimate is not a crack width in millimetres. A physical measurement needs its actual measurement source and method.

## 4. SP:35 knowledge base and engine

### Source and validation

`sp35.kb.load_kb()` reads the five configured YAML files from `kb/sp35_2024/`, validates models, checks interval partitions and weight totals, cross-checks state coefficients across distress tables, resolves aliases, and computes a line-ending-independent version hash. Use this loader rather than parsing tables ad hoc.

Distress rows retain both their printed condition text and machine-usable interval bands. A row may have multiple criteria; condition mapping takes the worse criterion state. `null` means that state cannot occur for that row. An alias can point to an actual row or to `engineer_judgement`; the latter must not be converted into a fabricated measurement.

### Scoring model

For implementation details, read `sp35/condition.py`, `sp35/scores.py`, `sp35/mpi.py`, the YAML KB, and the interpretation register together.

- A sub-component score averages the state-derived scores across distress types actually noticed. No distress noticed or an absent element is NA and scores as defined by the engine/spec interpretation.
- Component and bridge scores are weighted roll-ups over the components/sub-components present, using the corresponding YAML weights and the actual included-weight denominator.
- The span model assigns observations to span, support, end-span, or bridge-level scope. The bridge BHI is the governing (lowest) span BHI.
- Star rating and MPI mappings are loaded from YAML. MPI ranks and rolls up the Table 7.7–7.9 factors/sub-factors.
- Critical and urgent lists are driven by distress states, independently of the overall BHI.

Do not recalculate SP:35 values in a report template or UI. Store the inputs, KB version, engine output, and engineer changes so a result can be reproduced.

### Interpretation and reference tests

The interpretation register currently records decisions E1–E14, including known printed-arithmetic discrepancies, weight-denominator behavior, star/MPI mapping, aliases, and the span model. It is marked draft in the source materials: seek the required bridge-engineer sign-off rather than treating an implementation decision as a signed standard interpretation.

The oracle in `tools/sp35_reference_check.py` independently reproduces worked examples and cross-checks selected KB data. It duplicates reference values intentionally to avoid sharing the engine's calculation path; this is the exception to the rule against duplicating runtime SP:35 source values. Do not “fix” a failing expected value to make tests pass. Investigate the interpretation and get sign-off.

## 5. SAM environment, checkpoint, and common issues

### Supported local GPU setup

The pinned Windows worker is Python 3.12, PyTorch 2.11.0+cu128, `sam3==0.1.4`, and the native-Windows Triton build listed in `requirements/vision-win.txt`. Rebuild only when appropriate; the setup script refuses to replace an existing `.venv-win` unless explicitly passed `-Force`.

```powershell
uv sync --group dev
.\tools\setup_vision_env.ps1
.\.venv-win\Scripts\python.exe .\tools\env_check.py
```

Use the GPU worker's Python executable for SAM commands. A common `ModuleNotFoundError: No module named 'torch'` means the command used a different Python environment, not that the project should install arbitrary system-wide torch.

### Checkpoint and loader

The expected local checkpoint is `data/weights/sam3.1/sam3.1_multiplex.pt`. `vision/model.py` contains its pinned expected size and SHA-256 and checks an explicitly allow-listed SAM package/checkpoint key mismatch. Download is gated by Hugging Face access and uses `HF_TOKEN` via `tools/fetch_weights.py`. Do not paste tokens into chat, source, logs, or this document. Set a token only in the user's environment, open a new terminal, and revoke any token that has been exposed.

**Load SAM only using `vision.model.load_model()`.** Its default path casts floating-point parameters in the vision trunk and text encoder to bf16 while preserving complex RoPE buffers, uses CUDA bf16 autocast, verifies expected checkpoint keys, and caps process memory. Casting the whole model to bf16 can break RoPE buffers. Unsupported missing/unexpected keys should raise rather than be accepted silently.

The current pinned `sam3==0.1.4` predates SAM 3.1 and is tolerated only for the documented still-image pairing. Do not extend this pairing to video/tracking. Video work requires a deliberately selected official code revision, pinned and validated separately.

### Example commands

Run a single photo:

```powershell
.\.venv-win\Scripts\python.exe -m vision.infer --photo "..\yugmi_images\IMG_4150.jpg"
```

Run a folder or resume it:

```powershell
.\.venv-win\Scripts\python.exe -m vision.infer --folder "..\yugmi_images" --outdir "data\pilot\infer_out" --resume
```

Run photos selected by the pilot manifest:

```powershell
.\.venv-win\Scripts\python.exe -m vision.infer --manifest "data\pilot\manifest.csv" --src "..\yugmi_images" --outdir "data\pilot\infer_all" --resume
```

Run the saved-output review without rerunning SAM:

```powershell
.\.venv-win\Scripts\python.exe -m tools.inference_review --predictions "data\pilot\infer_all" --images "..\yugmi_images" --outdir "data\pilot\review_all"
```

The CLI defaults output to `data/pilot/infer_out`; explicit `--outdir` changes it. Inspect the run's `run_summary.json` and review summary instead of assuming a command's output path or inferring success from the process starting.

### Inference output and resume caveats

An inference output directory typically has one `<photo-stem>.json` per photo, optional `<photo-stem>_overlay.png` images, an append-only `manifest.jsonl`, and `run_summary.json`. Per-photo JSON includes a status; image dimensions; accepted components/distresses; suppressors/rejected detections; mask RLEs; boxes, score and host metadata; prompt-library identity; model identity; time and peak-memory data.

`--resume` skips existing per-photo JSON files whose status is `ok` or `ok_reduced`. If prompts, code, thresholds, or post-processing behavior have changed, do not treat an old `--resume` directory as a clean new run. Use a distinct output directory or explicitly manage the old generated files after checking ownership and contents.

## 6. Current verified snapshot (9 October 2026)

The following was checked in the local workspace for this handoff:

- `data/pilot/infer_all` and its `run_summary.json` exist. It records 167/167 processed photos as `ok`, with no reduced-resolution retries or errors. The run records RTX 2050, PyTorch 2.11.0+cu128, SAM package 0.1.4, bf16, and the pinned checkpoint SHA prefix.
- `data/pilot/review_all` exists. Its report summary records 167 successful photos and 893 detections across 21 classes. No ground-truth annotations were supplied, and it correctly reports no accuracy metrics.
- The pinned SAM checkpoint file is present locally. Presence alone is not proof that a future copy is valid; run the repository verification if moving/restoring it.
- Eight test files were found. The most recent recorded full suite result in the project handoff/history is **82 passed**; rerun tests if working from a changed checkout.
- The implementation plan and dated gate documents contain older status. In particular, application, evaluation, signed KB, engineer approval, privacy/blur, and cloud-consent readiness must be verified from source/evidence rather than inferred from a green checkbox or historical review.
- At handoff creation, the only worktree changes were this document and its README link. Always recheck `git status --short` before continuing; preserve changes you did not make and do not reset, clean, stage, commit, or overwrite generated/user-owned files unless asked.

The current local data/weights/review artifacts are not portable project source. Their existence in this workspace does not imply that another PC has them or that they are tracked in Git.

## 7. Project status and roadmap

Consult [IMPLEMENTATION_PLAN.md](./plan/IMPLEMENTATION_PLAN.md) for authoritative work-package descriptions, dependencies, and acceptance criteria. The phase labels below describe the intended plan; they are not a declaration that the corresponding gate has passed.

### Existing foundations

- KB loader and deterministic SP:35 condition/scoring/MPI logic with unit, golden, and property coverage.
- Pilot ingestion and a 167-photo pilot set.
- Still-photo SAM inference, post-processing, optional tiling, JSON/RLE outputs, overlays, resume, timing, and VRAM logging.
- Inference visualization/review and an optional COCO mask comparison path.
- Native Windows GPU environment and a Linux CUDA Docker target.

### Important unfinished product work

- Complete ingest metadata, engineer-confirmed inventory/span/support organization, and export privacy/blur checks.
- Persistent bridge/inspection data model and engineer review workbench with audit history, manual correction/refinement, measured-value provenance, and required engineer judgement. The upload dashboard does not yet do these tasks.
- Measurement workflow for extents, counts, crack width, and other field values; no synthetic physical units.
- Appendix IV/V report generation and engineer signature.
- Validated labelled dataset, double-label agreement, bridge-held-out split, evaluation harness, and baseline per-class/critical-class metrics.
- Prompt and threshold tuning against labelled data; current prompt thresholds are starting guesses.
- Batch runner with interruption/restart evidence on the company GPUs.
- Signed interpretation register, engineer review, and phase gate acceptance evidence.

**Added 12 October 2026:** the dashboard/API can accept single or batch JPG/PNG uploads, queue local SAM analysis after an explicit user action, and show saved inference plots and processing statistics. It is localhost-only and single-user; it is not a multi-user/network-hardened service. See the README for launch commands.

### Cosmos Reasoner 3 / RTX 3060

**Team-reported update (9 October 2026):** Cosmos Reasoner 3 has been run on the other PC using 4-bit quantization. This is a reported feasibility milestone, not yet a reproducible benchmark in this repository: the exact GPU/VRAM, model revision, inference runtime, quantization implementation, peak VRAM, latency, and test inputs/results still need to be recorded. Cosmos is not currently wired into the inspection inference flow or an implemented Reasoner contract. Do not treat a successful quantized launch as evidence of inspection quality or add it to the critical path before the human review boundary, labelled evaluation set, and baseline are in place.

The project plan includes a Cosmos 3 spike primarily on an RTX 5070 Ti and considers Cosmos Reason 2 alternatives. Earlier planning research of the [NVIDIA Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) recorded an upstream reasoner-only Cosmos 3 Nano memory estimate around 16–17 GB, with larger requirements for the full framework path. This is time-sensitive and must be rechecked against the current official model/runtime docs. The reported 4-bit run indicates that a particular configuration may work on the other PC; it does not establish that every Cosmos 3 path fits an RTX 3060 or resolve the difference from upstream memory estimates. Identify the exact GPU/VRAM, driver, OS, RAM, disk, model and runtime configuration and capture peak-memory measurements before making a hardware recommendation.

For the 3060 machine, prioritize the Phase 2 labelled evaluation, SAM prompt sweeps, batch runner, and alternatives whose verified requirements fit its actual VRAM. If revisiting Cosmos 3:

1. Recheck current NVIDIA official docs/model card/cookbook and applicable license/terms.
2. Measure available VRAM and peak memory on the exact runtime and checkpoint; do not extrapolate from parameter count.
3. Define and test a narrow `Reasoner` contract plus deterministic stub before integrating any model.
4. Evaluate on engineer-labelled crops for false-positive reduction **and** critical-class recall loss.
5. Keep all reasoner outputs advisory and schema-validated; it must never assign a final condition state.
6. Hosted NIM sends photos off-machine and requires explicit project-level consent.
7. Record latency, VRAM, model/runtime versions, quality impact, and the hosting decision.

## 8. Development and verification workflow

### Before editing

1. Read `AGENTS.md`, the relevant source, related tests, and the current phase/work-package in the implementation plan.
2. Check `git status --short` and inspect existing modifications in any file you may touch. The worktree can contain user-owned changes.
3. For `kb/` or `sp35/`, follow the SP:35 fidelity rules, write tests first, and run the `sp35-golden-tests` skill before finishing.
4. For any phase gate/release, run the `phase-gate-review` skill and attach evidence.
5. For GPU memory/speed or OOM changes, use the `vram-probe` skill before adjusting inputs, precision, loader, or prompt workload.
6. For reserved GPU work, follow `gpu-batch-day`.

### Commands

```powershell
# CPU tests (project default)
uv run --group dev pytest

# Independent Chapter 7 worked-example oracle
uv run --with pyyaml python tools/sp35_reference_check.py

# Phase 0 verification; add --strict at the gate
uv run --with pyyaml python tools/verify_phase0.py

# Check the pinned GPU-worker environment
.\.venv-win\Scripts\python.exe .\tools\env_check.py --full

# GPU probe (native Windows environment)
.\.venv-win\Scripts\python.exe -m vision.probe
```

Prefer the narrowest relevant tests for routine edits, then use the golden/oracle checks where applicable. Vision behavior changes need labelled per-class evaluation once that harness/data exist; passing unit tests alone does not establish visual accuracy. Never modify golden expected values merely to obtain a pass.

### Change discipline

- One bounded work package per branch; do not commit unless asked.
- Update directly related documentation and tick the plan only when its evidence-based acceptance is met.
- Keep scoring deterministic and independent of GPU/model dependencies.
- Surface invalid inputs and failures explicitly; do not silently fall back to success-shaped results.
- Never change inference JSON or stored photos just to improve a plot.
- Do not run broad destructive cleanup against `data/`, `.venv-win`, or the repository.

## 9. Onboarding checklist for another model or engineer

Before taking action:

1. Read this file and `AGENTS.md`.
2. Read the current implementation plan section and interpretation register entries relevant to the task.
3. Inspect the current Git status, not just this snapshot.
4. Confirm what is present on this machine: Python environments, checkpoint, image set, inference outputs, annotations, GPU model/VRAM, and available storage.
5. Distinguish source-controlled implementation from local data and from planned work.
6. Confirm whether a task needs code, a report, an evaluation, or only explanation; do not rerun expensive GPU inference when reviewing stored output suffices.
7. State what was actually measured, what was only inspected, and what remains unknown.

If this document conflicts with current code, YAML, tests, a signed engineer decision, or a current official upstream requirement, stop and reconcile the source-of-truth discrepancy before changing behavior.
