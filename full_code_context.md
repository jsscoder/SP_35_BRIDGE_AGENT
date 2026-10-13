# SP:35 Bridge Inspection AI — Full Codebase Context & Architecture Reference

**Repository:** `sp35-bridge-ai`  
**Purpose:** Comprehensive architectural, functional, and code-level reference for agents and developers across workstations.  
**Standard:** Indian Roads Congress code **IRC:SP:35-2024 Chapter 7** (*Manual for Bridge Management, Inventory, Inspection and Maintenance*).  

---

## 1. High-Level Mission & Non-Negotiable Rules

The application takes a folder of bridge inspection photos (smartphones or DJI drones), segments components and distresses via **Meta SAM 3.1**, and computes official IRC:SP:35-2024 condition states, Bridge Health Index ($BHI$), Star Ratings (1★–5★), Critical Lists, and Major Prioritization Indices ($MPI$) for highway networks.

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
├── README.md                  # Quickstart, setup instructions, and execution cheatsheet
├── pyproject.toml             # uv package manager config, dependencies, pytest settings
├── full_code_context.md       # THIS FILE — complete codebase reference
│
├── requirements/
│   └── vision-win.txt         # 41 pinned dependencies for CUDA 12.8 GPU worker
│
├── kb/                        # Knowledge Base (Versioned source of truth)
│   ├── sp35_2024/
│   │   ├── components.yaml    # Table 7.1: Components C1-C7, sub-components, AHP weights
│   │   ├── condition_states.yaml # States I-V, penalty weights S_k, 1-5 star rating bands
│   │   ├── distress_bridge.yaml  # Table 7.2: Bridge distress rows, units, state intervals
│   │   ├── distress_approach.yaml# Table 7.3: Approach distress rows, units, state intervals
│   │   ├── mpi.yaml           # Tables 7.7-7.9: Maintenance Prioritization factors F1-F8
│   │   └── golden_examples.yaml  # Golden test benchmarks (Yamuna Bridge, worked examples)
│   └── prompts/
│       └── sam3_prompts.yaml  # 87 SAM 3.1 prompts mapped to SP:35 component & distress IDs
│
├── sp35/                      # Deterministic IRC:SP:35-2024 Scoring Engine (Pure Python, CPU)
│   ├── __init__.py            # Public package exports
│   ├── intervals.py           # Mathematical interval parsing: [a,b], (a,b], check_partition
│   ├── kb.py                  # Pydantic schemas validating all kb/ YAML files on load
│   ├── condition.py           # Maps physical measurements -> Condition State I-V
│   ├── scores.py              # Equations 1-3, Span model, Bridge BHI, Star rating, Critical list
│   └── mpi.py                 # Equations 4-5: Major Prioritization Index rollup
│
├── vision/                    # SAM 3.1 Computer Vision Pipeline
│   ├── __init__.py            # Vision exports
│   ├── model.py               # Safe SAM 3.1 loader (bf16 trunk/text encoders, VRAM cap, key check)
│   ├── postprocess.py         # Pure NumPy postprocessing: suppression, synonym merge, host gating
│   ├── tiling.py              # Native-resolution overlapping tile generator for high-res images
│   ├── infer.py               # Main CLI/worker: folder/manifest inference, resume, overlay render
│   └── probe.py               # VRAM & speed benchmark suite (generates machine-readable JSON/MD)
│
├── ingest/                    # Ingestion, Deduplication, and Dataset Organization
│   ├── __init__.py            # Ingest exports
│   └── build_pilot.py         # Ingestion script: dHash dedupe, EXIF/DJI XMP extraction, probe10 builder
│
├── tools/                     # Operational, Setup & Verification Utilities
│   ├── sp35_reference_check.py# Standalone mathematical oracle (checks book math & YAMLs)
│   ├── verify_phase0.py       # Gate verification runner (checks oracle, probe, hygiene, tests)
│   ├── env_check.py           # Validates PyTorch, CUDA 12.8, Triton, and SAM 3.1 environment
│   ├── fetch_weights.py       # Downloads SAM 3.1 checkpoint safely via HF_TOKEN
│   └── setup_vision_env.ps1   # PowerShell automation script building .venv-win
│
├── tests/                     # 69 Unit, Property, and Golden Tests (100% green)
│   ├── conftest.py            # Global fixtures: loads KB and golden YAMLs once
│   ├── golden/
│   │   └── test_ch7_golden.py # Recomputes Chapter 7 book examples against expected values
│   ├── property/
│   │   └── test_ch7_properties.py # Hypothesis property tests (monotonicity, bounds [0, 100])
│   └── unit/
│       ├── test_engine.py     # Unit tests for sp35/kb.py, intervals.py, condition.py, scores.py
│       ├── test_postprocess.py# Unit tests for vision/postprocess.py rules (CPU, no GPU)
│       ├── test_tiling.py     # Unit tests for tiling geometry and coverage
│       └── test_build_pilot.py# Unit tests for pilot classifier and dHash deduplication
│
├── docs/                      # Documentation, Plans, Audits & Gate Reviews
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
│       └── vram_probe_rtx2050_2026-10-08.json # Machine-readable probe benchmark data
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
        └── phase-gate-review  # Skill to prepare gate evidence pack
```

---

## 3. Deep Dive: Modules & Code Walkthrough

---

### Module 3.1: The SP:35 Deterministic Scoring Engine (`sp35/`)

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
* **Key Classes & Functions:**
  - `assess(kb, states_by_sub, scope, cs_overrides) -> Assessment`: Computes $CS, CI, BHI$, stars, critical and urgent lists for a set of observations.
  - `Span(id, left_support, right_support, observations)`: Represents a bridge span between two supports.
  - `Support(label, kind)`: Represents an abutment or pier.
  - `assess_bridge(kb, spans) -> BridgeAssessment`: Implements the span model (Decision E14), combines support observations (taking the worse of the two supports), and rolls up to bridge BHI.

#### 5. `sp35/mpi.py`
* **Purpose:** Implements Equations 4 & 5 for network-level maintenance budget prioritization (Tables 7.7–7.9).
  - **Equation 5:** $FR_i = \frac{\sum_j W_j \cdot PR_j}{\sum_j W_j}$
  - **Equation 4:** $MPI = \frac{\sum_i W_i \cdot FR_i}{\sum_i W_i}$
  where $PR \in \{0, 25, 50, 75, 100\}$.
* **Function:** `f7_from_bhi(kb, bhi)` derives Factor F7 (Bridge Condition) from BHI via star bands.

---

### Module 3.2: Computer Vision & SAM 3.1 Engine (`vision/`)

Handles promptable segmentation of bridge structural members and defects.

#### 1. `vision/model.py`
* **Purpose:** Safely loads the `facebook/sam3.1` model (`sam3.1_multiplex.pt`) within the RTX 2050 4GB memory constraint.
* **Crucial Architectural Fix (Audit A2):**
  - **NEVER** call `model.to(torch.bfloat16)` on the entire model: doing so strips the imaginary part of the 32 `complex64` RoPE rotary embedding buffers and breaks segmentation completely!
  - Instead, `vision/model.py` selectively iterates through the floating-point parameters of the vision trunk (`model.backbone.vision_backbone.trunk`) and language encoder (`model.backbone.language_backbone.encoder`) and casts only them to `torch.bfloat16`.
  - **Result:** Halves model memory to **1.75 GiB load**, reducing peak VRAM from 4.18 GB to **2.57 GB** with zero system RAM paging.
* **Key Guardrails:**
  - `EXPECTED_MISSING`: Asserts that exactly 4 neck convolutional tensors (`convs.3.*`, discarded by `scalp=1`) are missing. Any other key mismatch raises `CheckpointMismatch`.
  - Memory Capping: Uses `torch.cuda.set_per_process_memory_fraction()` to cap memory below physical free VRAM, ensuring OOM errors raise cleanly instead of crawling into Windows shared memory.

#### 2. `vision/postprocess.py`
* **Purpose:** Pure NumPy/Python post-processing pipeline turning raw SAM masks into structured, audit-ready detections. Runs on CPU without GPU dependencies.
* **Key Classes & Data Structures:**
  - `PromptClass`: Dataclass holding class ID, kind (`component`, `distress`, `suppression`), prompts, SP:35 code reference, threshold, min area, and suppressor list.
  - `Detection`: Holds mask array, score, bounding box, prompt, and assigned host.
  - `PostResult`: Container with `components`, `distresses`, `suppressors`, and `rejected`.
* **The 5 Post-Processing Rules:**
  1. **Threshold & Area Filtering:** Drops detections below class threshold or min area.
  2. **Synonym Merging:** If two synonymous prompts (e.g., `"spalled concrete"` and `"broken concrete surface"`) find the same region ($IoU \ge 0.60$), merges them into a single detection.
  3. **Suppression Subtraction:** Subtracts suppression masks (cables, ropes, posters, people, vehicles) from distresses. If $> 50\%$ covered by suppressors, rejects the distress.
  4. **Host Component Gating:** A distress must be $\ge 50\%$ inside a detected component. If it re-detects the entire component ($IoU > 0.80$), rejects it. If no component was detected, checks `photo_host` fallback from tags.
  5. **Cross-Class Disambiguation:** If two different distress classes fire on the same patch ($IoU \ge 0.90$), keeps only the higher-scoring class.

#### 3. `vision/tiling.py`
* **Purpose:** Generates overlapping bounding boxes for high-resolution images (e.g. 12 MP / 24 MP) to detect hairline cracks and projecting rebar that disappear when downscaled.
* **Function:** `tile_boxes(w, h, size=1008, overlap=0.20) -> List[Tuple[int, int, int, int]]`.

#### 4. `vision/infer.py`
* **Purpose:** Production CLI and batch runner.
* **Usage:**
  - Single photo: `python -m vision.infer --photo path/to/img.jpg --tile`
  - Folder run: `python -m vision.infer --folder data/pilot/probe10 --resume`
  - Full manifest: `python -m vision.infer --manifest data/pilot/manifest.csv --src ../yugmi_images --outdir data/pilot/infer_all --resume`
* **Outputs per photo:**
  - `<stem>.json`: Detailed JSON containing component masks (COCO-RLE), distress masks, rejected list with reasons, and memory/time stats.
  - `<stem>_overlay.png`: Color-coded inspection overlay (Blue = Components, Red = Distresses, Yellow = Suppressors).
  - Appends progress to `manifest.jsonl`.

#### 5. `vision/probe.py`
* **Purpose:** Automated hardware benchmarking and memory profiling. Tests 10 representative photos across 87 prompts in `bf16` plus one native tiled pass. Generates `docs/reports/vram_probe_<label>_<date>.json` and `.md`.

---

### Module 3.3: Ingestion & Dataset Preparation (`ingest/`)

#### `ingest/build_pilot.py`
* **Purpose:** Turns raw inspection folders into cleaned, auditable datasets.
* **Processing Steps:**
  1. Computes SHA256 checksums and perceptual hashes (`dHash` 16×16 via Lanczos grayscale thumbnails).
  2. Identifies and excludes HEIC-conversion twins, resaved `_1` copies, and near-duplicates (Hamming distance $\le 20/256$).
  3. Inspects EXIF and DJI XMP tags (`drone-dji:ImageSource="InfraredCamera"`) to exclude thermal non-bridge frames and screenshots.
  4. Output files:
     - `data/pilot/manifest.csv`: Lists every file, its keep/drop status, and the explicit reason.
     - `data/pilot/tags.csv`: Links photos to span/component codes.
     - `data/pilot/probe10/`: Copies 10 representative probe photos across 12 MP and 24 MP frames covering piers, pier caps, bearings, soffits, and decks.

---

### Module 3.4: Verification & Tools (`tools/`)

#### 1. `tools/sp35_reference_check.py`
* **Purpose:** The **independent mathematical oracle**.
* **Zero Project Imports:** Contains its own isolated hardcoded copy of Table 7.1 and worked examples.
* **Validations Performed:**
  - Deck slab example under Eq. 3 ($75.0$ vs printed $50.0$, Decision E1).
  - Pier cap example ($68.75$).
  - Yamuna Bridge Table 7.6 BHI ($95.16$ with printed CS, $95.01$ strictly by state, tolerance $\pm 0.2$).
  - MPI worked examples (Bridge 1: $44.77$, Bridge 2: $33.19$).
  - Cross-checks that `components.yaml`, `mpi.yaml`, `distress_bridge.yaml`, and `distress_approach.yaml` have valid weights and non-overlapping partitions.

#### 2. `tools/verify_phase0.py`
* **Purpose:** Phase 0 verification suite.
* **Exit Codes & Modes:**
  - Day-to-day mode: `uv run python tools/verify_phase0.py` (passes with `PENDING` on human sign-off).
  - Gate review mode: `uv run python tools/verify_phase0.py --strict` (fails on `PENDING`).
* **Checks:**
  - `oracle`: Runs `sp35_reference_check.py`.
  - `kb-signed`: Checks whether E1–E14 register is signed by the bridge engineer.
  - `probe-report`: Verifies that `vram_probe_*.json` exists and reports `pass_3_4gb: true`.
  - `pilot-set`: Asserts 383 files $\to$ 167 kept photos, with reasons for every exclusion.
  - `inference-outputs`: Asserts all detections carry valid SP:35 codes.
  - `repo-hygiene`: Scans git tracked and untracked files for regex `hf_[A-Za-z0-9]{30,}` tokens, `.pt` weights, and private photos.

#### 3. `tools/fetch_weights.py` & `tools/env_check.py`
* Safe checkpoint downloader reading `HF_TOKEN` from the shell environment without placing tokens in command-line arguments or git configs. Checks the downloaded file against published SHA256 (`0567debeec80...`).

---

## 4. Test Suite (`tests/`)

Ran via: `uv run pytest -v` (69 passed in 2.38s).

| Test Directory | File | Items | What It Tests |
|---|---|---|---|
| `tests/golden/` | `test_ch7_golden.py` | 8 | Eq. 3 examples, Yamuna Bridge BHI (strict & printed CS), Table 7.9 MPI examples. |
| `tests/property/` | `test_ch7_properties.py` | 6 | Hypothesis property tests: $BHI \in [0, 100]$, monotonicity of defects, NA element invariance, bridge BHI = min span BHI. |
| `tests/unit/` | `test_engine.py` | 30 | Interval parsing, KB loading, gap/overlap rejection, condition state mapping, dual-criteria evaluation. |
| `tests/unit/` | `test_postprocess.py` | 18 | Synthetic mask tests: synonym merging, whole-component redetection rejection, suppressor subtraction, SP:35 code preservation. |
| `tests/unit/` | `test_tiling.py` | 5 | Tile bounding box generation, 100% pixel coverage, tile overlap geometry. |
| `tests/unit/` | `test_build_pilot.py` | 2 | dHash duplicate detection, file classification heuristics. |

---

## 5. Current Data Artifacts & Benchmarks

1. **Pilot Dataset (`data/pilot/`):**
   - 383 raw images $\to$ 167 kept flyover photos.
   - `probe10/`: 10 benchmark photos used for rapid testing.
2. **Inference Outputs (`data/pilot/infer_all/`):**
   - Full 167-photo inference completed on RTX 2050 laptop (`ok: 167`, zero errors, ~16.3s per photo).
3. **Hardware Performance (RTX 2050 4GB):**
   - Peak Allocated VRAM: **2.57 GB** (Target: $\le 3.4\text{ GB}$).
   - Peak Reserved VRAM: **2.89 GB** (Zero shared-memory spill).
   - Speed: **16.5 s / photo** across 87 prompts at 1024px.
   - Native-Resolution Tiled Pass: 20 tiles in **47.6 s** (peak 2.57 GB).

---

## 6. How to Run Everyday Commands

```powershell
# 1. Run all unit, golden, and property tests
uv run pytest

# 2. Run the independent SP:35 math oracle
uv run --with pyyaml python tools/sp35_reference_check.py

# 3. Run Phase 0 verification suite (strict mode for gate review)
uv run --with pyyaml python tools/verify_phase0.py --strict

# 4. Check CUDA, GPU, and Python environment
.venv-win\Scripts\python.exe tools\env_check.py --full

# 5. Run SAM 3.1 inference on a photo folder
.venv-win\Scripts\python.exe -m vision.infer --folder data/pilot/probe10 --resume

# 6. Run native-resolution tiled inference (detects hairline cracks and rebar)
.venv-win\Scripts\python.exe -m vision.infer --photo data/pilot/probe10/IMG_4231.jpg --tile

# 7. Run automated hardware VRAM & speed benchmark
.venv-win\Scripts\python.exe -m vision.probe --photos data/pilot/probe10 --label rtx2050
```

---

## 7. Next Milestones & Open Roadmaps

1. **Cosmos Reasoner Integration (Phase 3):**
   - Full specification and guardrails available in [`docs/plan/COSMOS_REASONER_AGENT_SPEC.md`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/docs/plan/COSMOS_REASONER_AGENT_SPEC.md).
   - Target: Connect SAM crops to Cosmos 3 on the RTX 3060 desktop (via vLLM 4-bit) to verify genuine distresses, filter noise, and draft Appendix IV narrative.
2. **Vision Detection Fine-Tuning:**
   - Full implementation prompt available in [`docs/plan/OPUS_DETECTION_FIX_PROMPT.md`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/docs/plan/OPUS_DETECTION_FIX_PROMPT.md).
   - Target: Remove generic `"concrete parapet"` prompt causing false `crash_barrier` triggers, implement soft-host orphan rescue, and boost bearing/joint detections.
3. **Closing Gate G0:**
   - Requires Bridge Engineer sign-off on `docs/plan/SP35_CH7_INTERPRETATIONS.md` (E1–E14) and rotating the Hugging Face access token.

