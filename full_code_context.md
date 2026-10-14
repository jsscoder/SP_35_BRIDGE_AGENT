# SP:35 Bridge Inspection AI — Full Codebase Context & Architecture Reference

**Repository:** `sp35-bridge-ai`  
**Purpose:** Authoritative architectural, functional, and code-level reference for AI agents, developers, and distributed workstations (Laptop RTX 2050 / Desktop RTX 3060).  
**Standard:** Indian Roads Congress code **IRC:SP:35-2024 Chapter 7** (*Manual for Bridge Management, Inventory, Inspection and Maintenance*).  
**Current Test Status:** **144 tests passing** (`uv run pytest -v`).  

---

## 1. High-Level Mission & Non-Negotiable Rules

The application takes a folder of bridge inspection photographs (smartphones or DJI drones), segments components and distresses via **Meta SAM 3.1**, and computes official IRC:SP:35-2024 condition states, Bridge Health Index ($BHI$), Star Ratings (1★–5★), Critical Lists, and Major Prioritization Indices ($MPI$) for highway networks.

### Core Non-Negotiables (`AGENTS.md`)
1. **Zero Hardcoded Numbers:** All SP:35 weights, limits, and condition intervals come strictly from `kb/sp35_2024/*.yaml`. Never hardcode an $S_k$, weight, or star band in Python.
2. **Equations Over Misprints:** The codebase implements the official equations (Eq. 1–5), not the handbook's printed arithmetic misprints (recorded in `docs/plan/SP35_CH7_INTERPRETATIONS.md` items E1–E14).
3. **The AI Proposes; The Engineer Certifies:** Never automatically finalize a condition state. Every State V defect forces an entry on the Critical List regardless of overall bridge BHI.
4. **Never Invent Measurements:** Quantities in mm, m, or m² must be site-measured (calipers, crack gauges, scale cards) or typed by the engineer. Every value records its provenance.
5. **Data Privacy & Local Execution:** Photos stay on the local machine by default. No cloud calls without explicit user consent.
6. **VRAM Budget on Laptop (RTX 2050 4GB):** Peak VRAM must stay $\le 3.4\text{ GB}$.

---

## 2. Directory Tree & Architecture Overview

```
sp35-bridge-ai/
├── AGENTS.md                  # System mission, invariants, stack, and commands
├── README.md                  # Project entry point, setup overview, and command cheatsheet
├── pyproject.toml             # uv package manager config, dependencies, pytest settings
├── full_code_context.md       # THIS FILE — comprehensive codebase context reference
├── Dockerfile                 # Multi-stage container (CPU oracle / Linux CUDA 12.8 worker)
├── docker-compose.yml         # Local orchestration for app backend, web frontend & worker
│
├── app/                       # Desktop Review Application (WP1.6)
│   ├── backend/               # FastAPI REST API & Local Task Runner
│   │   ├── __init__.py
│   │   ├── __main__.py        # Backend CLI entrypoint (uvicorn app.backend.main:app)
│   │   └── main.py            # Upload API, SQLite jobs DB, background subprocess execution
│   └── web/                   # Next.js 15+ Interactive Dashboard
│       ├── package.json       # Dependencies (React 19, Lucide, Tailwind CSS)
│       ├── tsconfig.json      # TypeScript configuration
│       ├── next.config.ts     # Next.js configuration
│       └── app/
│           ├── layout.tsx     # Global HTML shell and fonts
│           ├── globals.css    # Clean modern UI styling & component tokens
│           └── page.tsx       # Photo batch upload, job queue, metrics, embedded report viewer
│
├── kb/                        # Knowledge Base (Versioned source of truth)
│   ├── sp35_2024/             # Pure Chapter 7 tables
│   │   ├── components.yaml    # Table 7.1: Components C1-C7, sub-components, AHP weights
│   │   ├── condition_states.yaml # States I-V, penalty weights S_k, 1-5 star rating bands
│   │   ├── distress_bridge.yaml  # Table 7.2: Bridge distress rows, units, state intervals
│   │   ├── distress_approach.yaml# Table 7.3: Approach distress rows, units, state intervals
│   │   ├── mpi.yaml           # Tables 7.7-7.9: Maintenance Prioritization factors F1-F8
│   │   └── golden_examples.yaml  # Golden test benchmarks (Yamuna Bridge, worked examples)
│   ├── platform/
│   │   └── anomaly_map.yaml   # Maps external inspection platform defect strings to SP:35 rows
│   ├── prompts/
│   │   └── sam3_prompts.yaml  # SAM 3.1 prompts mapped to SP:35 component & distress IDs
│   └── requirements/
│       └── vision-win.txt     # Pinned CUDA 12.8 PyTorch / SAM 3.1 Windows dependencies
│
├── sp35/                      # Deterministic IRC:SP:35-2024 Scoring Engine (Pure Python, CPU)
│   ├── __init__.py            # Public package exports
│   ├── intervals.py           # Mathematical interval parsing: [a,b], (a,b], check_partition
│   ├── kb.py                  # Pydantic schemas validating all kb/ YAML files on load
│   ├── condition.py           # Maps physical measurements -> Condition State I-V
│   ├── scores.py              # Equations 1-3, Span model, Bridge BHI, Star rating, Critical list
│   └── mpi.py                 # Equations 4-5: Major Prioritization Index rollup
│
├── vision/                    # SAM 3.1 Computer Vision & Condition Proposal Pipeline
│   ├── __init__.py            # Vision exports
│   ├── model.py               # Safe SAM 3.1 loader (bf16 trunk/text encoders, VRAM cap, key check)
│   ├── postprocess.py         # Pure NumPy postprocessing: suppression, synonym merge, host gating
│   ├── tiling.py              # Native-resolution overlapping tile generator for high-res images
│   ├── infer.py               # Main CLI/worker: folder/manifest inference, resume, overlay render
│   ├── probe.py               # VRAM & speed benchmark suite (generates machine-readable JSON/MD)
│   └── condition_proposals.py # Proposes SP:35 States I-V for SAM distresses (damage %, counts)
│
├── ingest/                    # Ingestion, Deduplication, and Platform Annotation Adapters
│   ├── __init__.py            # Ingest exports
│   ├── build_pilot.py         # Ingestion script: dHash dedupe, EXIF/DJI XMP extraction, probe10 builder
│   └── platform_annotations.py# Ingests platform JSON exports (polygons, anomaly strings, engineer states)
│
├── tools/                     # Operational, Verification & Quality Assurance Utilities
│   ├── sp35_reference_check.py# Standalone mathematical oracle (checks book math & YAMLs)
│   ├── verify_phase0.py       # Gate verification runner (checks oracle, probe, hygiene, tests)
│   ├── env_check.py           # Validates PyTorch, CUDA 12.8, Triton, and SAM 3.1 environment
│   ├── fetch_weights.py       # Downloads SAM 3.1 checkpoint safely via HF_TOKEN
│   ├── setup_vision_env.ps1   # PowerShell automation script building .venv-win
│   ├── inference_review.py    # Builds local standalone HTML visual review reports with overlays
│   ├── bifurcate_distress.py  # Generates side-by-side visual review cards for human triage (apt/misclassified)
│   └── state_scorecard.py     # Quantitative benchmark comparing model states vs engineer ground truth
│
├── scripts/                   # Container automation scripts
│   ├── docker-build.ps1       # Builds local Docker images
│   ├── docker-run.ps1         # Runs sp35-worker container
│   ├── docker-clean.ps1       # Cleans dangling containers and caches
│   ├── docker-publish.ps1     # Publishes Docker image
│   └── docker-test.ps1        # Runs test suite inside container
│
├── tests/                     # 144 Unit, Property, and Golden Tests (100% green)
│   ├── conftest.py            # Global fixtures: loads KB and golden YAMLs once
│   ├── golden/
│   │   └── test_ch7_golden.py # Recomputes Chapter 7 book examples against expected values
│   ├── property/
│   │   └── test_ch7_properties.py # Hypothesis property tests (monotonicity, bounds [0, 100])
│   └── unit/
│       ├── test_engine.py     # Tests for sp35/kb.py, intervals.py, condition.py, scores.py
│       ├── test_postprocess.py# Tests for vision/postprocess.py rules (CPU, no GPU)
│       ├── test_tiling.py     # Tests for tiling geometry and coverage
│       ├── test_build_pilot.py# Tests for pilot classifier and dHash deduplication
│       ├── test_condition_proposals.py # Tests for vision/condition_proposals.py
│       ├── test_inference_review.py    # Tests for tools/inference_review.py
│       ├── test_bifurcate_distress.py  # Tests for tools/bifurcate_distress.py
│       ├── test_state_scorecard.py     # Tests for tools/state_scorecard.py
│       └── test_app_api.py    # Tests for FastAPI backend endpoints & SQLite runner
│
├── docs/                      # Documentation, Plans, Audits & Gate Reviews
│   ├── KNOWLEDGE_BASE.md      # Durable onboarding and system reference
│   ├── plan/
│   │   ├── IMPLEMENTATION_PLAN.md      # Master 6-Phase development roadmap (Phases 0–6)
│   │   ├── SP35_CH7_INTERPRETATIONS.md # Register of IRC handbook misprints & code decisions (E1-E14)
│   │   ├── COSMOS_REASONER_AGENT_SPEC.md# Specs & guardrails for Cosmos Reasoner 3 integration
│   │   └── OPUS_DETECTION_FIX_PROMPT.md# Prompt for Claude Opus to fix false detections & dropped defects
│   ├── audit/
│   │   └── AUDIT_2026-10-08.md         # Comprehensive Phase 0 implementation audit
│   ├── gates/
│   │   ├── G0_2026-10-07.md            # Initial G0 review
│   │   └── G0_2026-10-08.md            # Re-issued G0 review with verified evidence
│   └── reports/
│       ├── runtime_decision.md         # Decision: Native Windows standard, WSL2 fallback
│       ├── vram_probe_rtx2050_2026-10-08.md   # Human-readable VRAM probe benchmark
│       ├── vram_probe_rtx2050_2026-10-08.json # Machine-readable probe benchmark data
│       ├── detection_fix_probe10_2026-10-12.md# Evaluation of prompt & postprocess fixes
│       └── scrum_review_2026-10-09.md         # Phase 0 closeout & team review
│
└── .agents/                   # Antigravity Agent Configuration
    ├── rules/
    │   ├── sp35-fidelity.md   # Triggers on kb/ or sp35/: strict equations & rules
    │   ├── vision-gpu.md      # Triggers on vision/: VRAM budget, prompt rules, loader constraints
    │   └── testing-and-gates.md# Triggers on gates/merges: golden tests & evidence policy
    └── skills/
        ├── sp35-golden-tests  # Skill to run oracle and golden test suite
        ├── vram-probe         # Skill to profile GPU VRAM allocation & time
        ├── gpu-batch-day      # Skill to orchestrate bulk inference on RTX 3060/5070 Ti
        ├── phase-gate-review  # Skill to prepare gate evidence pack
        └── sp35-condition-proposals # Skill to grade distress severity & condition states
```

---

## 3. Deep Dive: Modules & Code Walkthrough

---

### Module 3.1: The Review Desktop Application (`app/`)

A Windows-first desktop review application allowing engineers to drag-and-drop inspection photo sets, monitor local background SAM 3.1 runs, inspect overlays, and review proposed condition states.

#### 1. Backend (`app/backend/main.py`)
* **Technology:** FastAPI, SQLite (`jobs.sqlite3`), ThreadPoolExecutor, Python subprocess.
* **Responsibilities:**
  - Persists runs to SQLite database (`runs` table with ID, creation time, status, photo counts, log output).
  - Handles secure image uploads (validating image magic bytes via PIL, sanitized filenames, up to 200 images per batch).
  - Asynchronously spawns `vision.infer` as a managed subprocess using the active Python environment.
  - Automatically invokes `tools.inference_review` upon run completion to compile interactive visual HTML reports.
* **Key Endpoints:**
  - `POST /api/runs`: Uploads images and queues an inference run.
  - `GET /api/runs`: Lists past inspection runs with statuses and timestamps.
  - `GET /api/runs/{id}`: Detailed run status, timing, processed photos, and tail of process stdout logs.
  - `GET /api/runs/{id}/report`: Serves the generated visual HTML review report.
  - `GET /api/runs/{id}/proposals`: Returns computed SP:35 Condition State proposals generated by `vision.condition_proposals`.
  - `GET /api/system`: System health check (CPU, GPU info, disk availability).

#### 2. Frontend (`app/web/`)
* **Technology:** Next.js 15+ (App Router), React 19, TypeScript, Tailwind CSS, Lucide icons.
* **Files:**
  - `app/web/app/page.tsx`: Single-page cockpit dashboard featuring:
    - Drag-and-drop batch upload box.
    - Active job progress bar with real-time log stream.
    - Historical runs table with status badges (`running`, `completed`, `failed`).
    - Embedded full-screen iframe viewer for inspection overlay reports.
  - `app/web/app/globals.css`: Dark/light mode theme variables and modern layout styling.

---

### Module 3.2: Deterministic SP:35 Scoring Engine (`sp35/`)

Pure Python, zero GPU requirement. Implements the exact mathematics of IRC:SP:35-2024 Chapter 7.

#### 1. `sp35/intervals.py`
* **Purpose:** Handles mathematical range strings from Table 7.2/7.3 (e.g. `"=0"`, `"(0.3,1.0]"`, `"(5,inf)"`).
* **Key Classes & Functions:**
  - `Interval(lo, hi, left_closed, right_closed)`: Represents a 1D real interval.
  - `Interval.parse(text)`: Converts string notations into `Interval` objects.
  - `check_partition(intervals, domain)`: Verifies that intervals cover the domain without gaps or overlaps (used during startup validation).
  - `find(intervals, x)`: Returns the matching interval for value $x$.

#### 2. `sp35/kb.py`
* **Purpose:** Loads and validates `kb/sp35_2024/*.yaml` using **Pydantic v2**. If a weight doesn't sum to 100 or an interval has a gap, it raises `KBError` at startup.
* **Key Classes:**
  - `SubComponent`, `Component`, `ComponentsKB`: Encodes Table 7.1. Validates that sub-component weights sum to 100.
  - `DistressRow`: Encodes Table 7.2/7.3 rows. Handles single-metric rows and dual-criteria rows (`criteria: [width, length]`).
  - `KnowledgeBase`: Container holding all loaded KB files, weight dictionaries (`weights_main`, `weights_sub`), $S_k$ map, and a line-ending independent SHA256 version hash (`kb.version_hash`).
* **Function:** `load_kb(path=DEFAULT_KB_DIR) -> KnowledgeBase`.

#### 3. `sp35/condition.py`
* **Purpose:** Maps quantitative defect measurements to discrete condition states $I, II, III, IV, V$.
* **Key Function:**
  - `condition_state(kb, row_id, value_or_dict) -> ConditionResult`:
    - Looks up the row in `distress_bridge.yaml` or `distress_approach.yaml`.
    - Resolves aliases (e.g., `"Clogging"` $\to$ `T7.2-25 Inadequate drainage`).
    - If dual criteria (e.g., crack width and length), evaluates both and takes the **worse state**.
    - Returns `ConditionResult(state="III", explanation="Table 7.2 row 11: 0.6 in (0.3,1.0] -> III")`.

#### 4. `sp35/scores.py`
* **Purpose:** Calculates $CS_j, CI_i, BHI$, star rating, and the Critical List.
* **Core IRC:SP:35-2024 Equations:**
  - **Equation 3 (Sub-Component Score $CS_j$):**
    $$CS_j = \frac{\sum_{k=1}^d (100 - 100 \cdot S_k)}{d}$$
    where $d$ is the number of distress types noticed on sub-component $j$, and $S_k \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$. If nothing noticed or element absent (NA), $CS_j = 100$.
  - **Equation 2 (Component Index $CI_i$):**
    $$CI_i = \frac{\sum_j W_j \cdot CS_j}{\sum_j W_j}$$
    where $W_j$ are Table 7.1 column 5 weights. Divided by the sum of weights present (Decision E2, E6).
  - **Equation 1 (Span Bridge Health Index $BHI$):**
    $$BHI = \frac{\sum_i W_i \cdot CI_i}{\sum_i W_i}$$
    where $W_i$ are Table 7.1 column 3 weights.
* **Bridge Health Index Rule (§7.5.1, Decision E14):**
  $$\text{Bridge } BHI = \min_{\text{spans}} (BHI_{\text{span}})$$
  The bridge condition is governed by its weakest span.
* **Critical & Urgent Lists:**
  - **Critical List:** Every sub-component with any distress in **State V** (mandatory repair, regardless of BHI).
  - **Urgent List:** Every sub-component with any distress in **State IV**.
* **Key Functions:**
  - `assess(kb, states_by_sub, scope, cs_overrides) -> Assessment`: Computes $CS, CI, BHI$, stars, critical and urgent lists for a set of observations.
  - `assess_bridge(kb, spans) -> BridgeAssessment`: Implements the span model (Decision E14), combines support observations (taking the worse of the two supports), and rolls up to bridge BHI.

#### 5. `sp35/mpi.py`
* **Purpose:** Implements Equations 4 & 5 for network-level maintenance budget prioritization (Tables 7.7–7.9).
  - **Equation 5:** $FR_i = \frac{\sum_j W_j \cdot PR_j}{\sum_j W_j}$
  - **Equation 4:** $MPI = \frac{\sum_i W_i \cdot FR_i}{\sum_i W_i}$
  where $PR \in \{0, 25, 50, 75, 100\}$.
* **Function:** `f7_from_bhi(kb, bhi)` derives Factor F7 (Bridge Condition) from BHI via star bands.

---

### Module 3.3: Computer Vision & Condition Proposals (`vision/`)

#### 1. `vision/model.py`
* **Purpose:** Safely loads the `facebook/sam3.1` model (`sam3.1_multiplex.pt`) within the RTX 2050 4GB memory constraint.
* **Crucial Architectural Fix (Audit A2):**
  - Selectively casts only floating-point parameters of the vision trunk and text encoder to `torch.bfloat16`.
  - **Never** casts the whole model with `.to(bfloat16)`, preserving the 32 `complex64` RoPE rotary embedding buffers intact.
  - Caches load memory at **1.75 GiB**, ensuring peak VRAM stays at **2.57 GB** (limit $\le 3.4\text{ GB}$).
  - Verifies checkpoint keys against `EXPECTED_MISSING` (exactly 4 neck conv keys discarded by `scalp=1`).

#### 2. `vision/postprocess.py`
* **Purpose:** Pure NumPy post-processing rules.
* **Core Functions:**
  - Merges duplicate synonym prompts ($IoU \ge 0.60$).
  - Subtracts global/local suppressors (cables, ropes, posters, people, vehicles).
  - Enforces Component Conflict Resolution (NMS) to eliminate phantom `crash_barrier` masks overlapping pier caps and girders.
  - Soft-Host Orphan Rescue: If high-confidence distresses lack a segmented host, rescues them under `host: {"class": "unassigned_structural", "source": "orphan_rescue"}` rather than discarding safety-critical defects.

#### 3. `vision/condition_proposals.py`
* **Purpose:** Evaluates SAM 3.1 distress detections and proposes official IRC:SP:35-2024 Condition States $I$–$V$.
* **Core Logic:**
  - Includes a standalone pure Python COCO-RLE decoder (`decode_rle`) so it runs anywhere without native C++ compilation.
  - **Percentage-based rows (spalling, honeycombing, leaching, rust):**
    $$\text{extent \%} = \frac{\text{Area of distress mask}}{\text{Area of host member mask}} \times 100$$
    Passes extent % to `sp35.condition.condition_state` to obtain proposed state.
  - **Count-based rows (missing elements, drainage spots, potholes):** Evaluates count directly against Table 7.2 intervals.
  - **Site-measurement rows (crack width mm, scour depth m):** Explicitly marks `needs_site_measurement` (abstains from guessing millimeter widths without a physical gauge/scale card).
  - **CLI:** `uv run python -m vision.condition_proposals --input data/pilot/infer_all_v1`
  - Supports `--calibrator <path_to_calibrator.pkl>` to refine damage % / visual proposal states with safety flags (`ai_visual_proposal_requires_site_certification`).
  - Generates `condition_proposals.json`, `condition_proposals.csv`, and `images_condition_status.csv`.

#### 4. `vision/condition_calibrator.py`
* **Purpose:** SP:35-2024 Hybrid Condition-State Calibrator & Threshold Optimizer.
* **Core Mechanisms:**
  - `valid_state_mask_for_row(kb, row_id)`: Enforces permitted states and null bands (e.g. State II is null for T7.2-16 Crushing).
  - `compute_corrected_denominators`: Mitigates whole-frame distortion by subtracting suppressors (ropes, banners, vehicles) and computing structural foreground proxies.
  - `extract_medial_crack_geometry`: Computes medial-axis width (px) and skeleton length (px) via Euclidean Distance Transform (`scipy.ndimage.distance_transform_edt`).
  - `build_asymmetric_cost_matrix`: Minimizes expected Quadratic Weighted Kappa distance with asymmetric under-call penalty ($\ge 2.5\times$ penalty for predicting a healthier state than ground truth).
  - `SP35ConditionCalibrator`: Gradient boosted model (`HistGradientBoostingClassifier`) blended with deterministic SP:35 KB prior.

---

### Module 3.4: Ingestion, Ground Truth & Evaluation Tools (`ingest/` & `tools/`)

#### 1. `ingest/platform_annotations.py`
* **Purpose:** Parses inspection platform annotation exports (`annotations.json`).
* **Functionality:**
  - Extracts polygons, anomaly names, asset names, and engineer-assigned condition states.
  - Maps asset names to SP:35 Table 7.1 sub-components (`C1.1` to `C7.3`) and supports/spans (e.g. `"Pier protection works — P5"` $\to$ `C2.9, P5`).
  - Maps anomaly strings to SP:35 Table 7.2/7.3 distress rows using [`kb/platform/anomaly_map.yaml`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/kb/platform/anomaly_map.yaml).

#### 2. `tools/state_scorecard.py`
* **Purpose:** Quantitative evaluation harness comparing AI-proposed states against certified bridge engineer ground-truth annotations.
* **Metrics Reported:**
  - **Exact Accuracy:** Agreement rate between proposed and ground-truth states.
  - **Within-1 Accuracy:** Proportions within $\pm 1$ condition state.
  - **Under-calls (Safety-Critical):** Detections where the AI predicted a better condition than the engineer (e.g., predicting State II when engineer rated State IV).
  - **Confusion Matrix & Quadratic Weighted Kappa ($\kappa_w$):** Standard inter-rater agreement metric via `quadratic_weighted_kappa`.
  - **Oracle Diagnostic Mode:** Runs the scoring engine directly against the engineer's polygon to determine whether an error stems from SAM segmentation or IRC rule mapping.
  - **Calibrated Evaluation:** Supports `--calibrator <path_to_calibrator.pkl>` to evaluate trained calibrator side-by-side with baseline and oracle.

#### 3. `tools/bifurcate_distress.py`
* **Purpose:** Rapid triage of detections into three queues: `apt` (accurate), `misclassified`, and `uncertain`.
* **Output:** Generates high-resolution side-by-side review cards (`_review_card`) showing the original photo alongside the color-coded defect overlay.

#### 4. `tools/inference_review.py`
* **Purpose:** Generates a standalone, dependency-free interactive HTML inspection report (`inference_review.html`) directly from inference JSON outputs. Displays image galleries, filterable detection lists, and defect coverage summaries against Table 7.2.

#### 5. `tools/train_calibrator.py`
* **Purpose:** CLI Training, 5-Fold GroupKFold Cross-Validation & Threshold Grid-Search Runner.
* **Workflow:**
  1. Per-class SAM 3.1 detection threshold grid search (evaluating `distresses` + `rejected` rescue).
  2. 5-fold GroupKFold cross-validation grouped by `photo` (zero same-photo data leakage).
  3. Benchmarking Baseline A (uncalibrated), Baseline B (zero-shot structural proxy prior), and Model C (hybrid calibrator).
  4. Exports `calibrator.pkl`, `threshold_recommendations.yaml`, and `calibration_scorecard_report.md`.

---

## 4. Test Suite (`tests/`)

Ran via: `uv run pytest -v` (**153 passed in ~6.8s**).

| Test Directory | File | Count | What It Tests |
|---|---|---|---|
| `tests/golden/` | `test_ch7_golden.py` | 8 | Eq. 3 examples, Yamuna Bridge BHI (strict & printed CS), Table 7.9 MPI examples. |
| `tests/property/` | `test_ch7_properties.py` | 6 | Hypothesis property tests: $BHI \in [0, 100]$, monotonicity of defects, NA element invariance, bridge BHI = min span BHI. |
| `tests/unit/` | `test_engine.py` | 30 | Interval parsing, KB loading, gap/overlap rejection, condition state mapping, dual-criteria evaluation. |
| `tests/unit/` | `test_postprocess.py` | 18 | Synthetic mask tests: synonym merging, whole-component redetection rejection, suppressor subtraction, SP:35 code preservation. |
| `tests/unit/` | `test_tiling.py` | 5 | Tile bounding box generation, 100% pixel coverage, tile overlap geometry. |
| `tests/unit/` | `test_build_pilot.py` | 2 | dHash duplicate detection, file classification heuristics. |
| `tests/unit/` | `test_condition_proposals.py`| 28 | Tests for `vision/condition_proposals.py` damage %, count rows, RLE decoding, and proposal rollups. |
| `tests/unit/` | `test_condition_calibrator.py`| 9 | Tests for feature extraction, EDT medial crack morphology, null state bands, asymmetric loss, and calibrator fit/save/load. |
| `tests/unit/` | `test_inference_review.py` | 12 | Tests for `tools/inference_review.py` HTML generation and coverage math. |
| `tests/unit/` | `test_bifurcate_distress.py`| 8 | Tests for review card generation and triage queue serialization. |
| `tests/unit/` | `test_state_scorecard.py`  | 14 | Tests for IoU polygon matching, confusion matrix, and kappa calculation. |
| `tests/unit/` | `test_app_api.py`          | 13 | Tests for FastAPI `/api/runs`, validation, SQLite persistence, and endpoints. |

---

## 5. Everyday Operational Commands

```powershell
# 1. Run all 144 unit, golden, and property tests
uv run pytest

# 2. Run the independent SP:35 math oracle
uv run --with pyyaml python tools/sp35_reference_check.py

# 3. Run Phase 0 verification suite (strict mode for gate review)
uv run --with pyyaml python tools/verify_phase0.py --strict

# 4. Launch the local desktop review app
# Backend (FastAPI on port 8000):
uv run python -m app.backend
# Web Frontend (Next.js on port 3000):
cd app/web; npm run dev

# 5. Run SAM 3.1 inference on a photo folder
.venv-win\Scripts\python.exe -m vision.infer --folder data/pilot/probe10 --resume

# 6. Generate proposed SP:35 Condition States from inference JSONs
uv run python -m vision.condition_proposals --input data/pilot/infer_all_v1

# 7. Evaluate model condition states against engineer ground truth
uv run python -m tools.state_scorecard --labels data/labels/annotations.json --predictions data/pilot/infer_all_v1

# 8. Generate standalone visual HTML report
uv run python -m tools.inference_review --predictions data/pilot/infer_all_v1 --images data/pilot/probe10 --output-dir data/pilot/review_html
```

---

## 6. Next Steps & External Agent Handoff

1. **Cosmos Reasoner Integration (Phase 3):**
   - Specification: [`docs/plan/COSMOS_REASONER_AGENT_SPEC.md`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/docs/plan/COSMOS_REASONER_AGENT_SPEC.md).
   - Designed to run on the **RTX 3060 desktop** via vLLM / 4-bit quantisation to verify genuine distresses, filter noise, and draft Appendix IV narrative blocks (*Probable Reasons, Maintenance Actions, Urgency*).
2. **Vision Tuning Strategy:**
   - Specification: [`docs/plan/OPUS_DETECTION_FIX_PROMPT.md`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/docs/plan/OPUS_DETECTION_FIX_PROMPT.md).
   - Focus: Eliminating phantom `crash_barrier` triggers, boosting bearing/expansion joint detection, and avoiding suppressor over-subtraction on exposed rebar.
