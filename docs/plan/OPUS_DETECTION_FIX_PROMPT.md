# Master Implementation Prompt for Claude Opus: Fixing SAM 3.1 Bridge Detections & Missed Defects

**Target Agent:** Claude Opus (Master Coding Agent)  
**Repository:** `sp35-bridge-ai` (`c:\Users\omkar\Desktop\sp-35 reasoner\sp35-bridge-ai`)  
**Mission:** Fix component misclassifications (especially spurious `crash_barrier` detections), recover missed/dropped defects, refine `kb/prompts/sam3_prompts.yaml`, and enhance `vision/postprocess.py` and `vision/infer.py` to achieve high-fidelity bridge inspection segmentation.

---

## 1. Executive Context & The Problem

During full-pass and probe evaluations of the Patna flyover pilot dataset (`data/pilot/probe10` and `data/pilot/infer_out`), inspection of the segmentation masks revealed three critical defects:

1. **Massive Over-Classification of `crash_barrier` (`C6.9`):**
   - Pier caps, girder flanges, retaining wall tops, and bearing pedestals (e.g., in `IMG_4227.jpg`) are constantly misclassified as `crash_barrier`.
   - **Root Cause:** In `kb/prompts/sam3_prompts.yaml`, `crash_barrier` includes the generic prompt `"concrete parapet"`. In urban bridge photography, almost every horizontal concrete ledge triggers this prompt with confidence $> 0.50$.
   - **Downstream Consequence:** Because `crash_barrier` falsely consumes the region, actual defects on pier caps and girders are rejected by `vision/postprocess.py` with:  
     `"re-detects the whole crash_barrier (IoU 0.99)"` or falsely assigned to host `crash_barrier`.

2. **Genuine Defects Not Being Masked At All (Dropped Detections):**
   - High-severity cracks, efflorescence, and rebar are completely missing from the final JSON outputs on several images.
   - **Root Cause A (Hard Host Gating):** In `vision/postprocess.py` (lines 262–268), any distress with $< 50\%$ area inside a detected component is completely rejected (`only X% inside any component`). If SAM fails to segment a challenging girder or pier cap, *all distresses on it are discarded*.
   - **Root Cause B (Fixed 0.50 Threshold):** Fine hairline cracks, subtle damp patches, and rebar often score between $0.35$ and $0.48$ in zero-shot SAM 3.1 and get dropped at the entry threshold.
   - **Root Cause C (Suppression Over-Subtraction):** Shadows and ropes subtract from defect masks, dropping their area below `min_area_px: 400`, triggering silent rejection.

3. **No Component-Level Mutual Exclusion (NMS):**
   - In `vision/postprocess.py`, components are kept without cross-class competition. The same physical pixel region can simultaneously be tagged as a `girder`, a `pier`, and a `crash_barrier`.

---

## 2. Scope of Work & Implementation Plan

You must implement the following 5 work packages without breaking existing SP:35 scoring or CI tests.

### Work Package 1: Prompt Library Refactoring (`kb/prompts/sam3_prompts.yaml`)
1. **Disambiguate `crash_barrier`:**
   - Remove ambiguous prompts like `"concrete parapet"` and `"bridge railing"`.
   - Replace with distinct, unambiguous classes:
     - `crash_barrier` (`C6.9`): `["concrete crash barrier", "Jersey barrier", "highway safety barrier", "W-beam guardrail"]`
     - `bridge_railing` (`C1.4` / `C6.9` depending on position): `["pedestrian steel railing", "bridge parapet railing"]`
2. **Add Missing Substructure & Superstructure Prompts:**
   - Add specific prompts for bearing shelves and pedestals:
     - `bearing_seat` (`C7.2`): `["bearing pedestal", "concrete bearing seat", "bearing shelf"]`
     - `pier_cap` (`C2.8`): `["pier cap beam", "hammerhead pier cap", "pier cantilever"]`
     - `girder` (`C5.1`): `["concrete I-girder", "prestressed concrete box girder", "girder bottom flange", "girder web"]`
3. **Per-Class Thresholds & Minimum Areas:**
   - Move away from the global $0.50$ default. Configure sensible per-class thresholds:
     - `crack`: threshold `0.35`, `min_area_px: 150` (cracks are thin and low-confidence in zero-shot)
     - `exposed_reinforcement`: threshold `0.38`, `min_area_px: 200`
     - `spalling`: threshold `0.45`, `min_area_px: 350`
     - `crash_barrier`: threshold `0.60`, `min_area_px: 2000` (must be high confidence to prevent spurious triggers)
     - `pier` / `pier_cap`: threshold `0.50`

### Work Package 2: Cross-Class Component Disambiguation / NMS (`vision/postprocess.py`)
1. Implement **Component Conflict Resolution**:
   - If two different component classes overlap with $IoU \ge 0.50$, resolve the conflict using:
     - Score difference (if score delta $> 0.15$).
     - Structural hierarchy heuristics (e.g., a photo tagged `C2.x` substructure should favor `pier` or `pier_cap` over `crash_barrier`).
   - Eliminate phantom `crash_barrier` masks overlapping pier caps and girders.

### Work Package 3: Soft-Host Fallback for Critical Distresses (`vision/postprocess.py`)
1. **Never Silently Discard High-Confidence Distresses:**
   - If a distress has high confidence (e.g., score $\ge 0.50$ for crack, rebar, or spalling) but fails the `host_inside_min` check:
     - Do **NOT** reject it.
     - Check the photo's tagged component (`photo_host`) from `data/pilot/tags.csv`.
     - If still unhosted, assign `host: {"class": "unassigned_structural", "sp35": "UNKNOWN", "source": "orphan_rescue"}` and preserve it in `distresses` marked `review_required: true`.
2. **Relax `redetect_iou` for Heterogeneous Masks:**
   - Ensure a spalling or crack detection is only rejected for "re-detecting whole component" if the bounding box and mask aspect ratio truly mirror the entire component, not when localized damage covers a small pedestal.

### Work Package 4: Tiled Pass & Scale Integration (`vision/infer.py` & `vision/tiling.py`)
1. Verify that `--tile` works seamlessly for fine distresses (`crack`, `exposed_reinforcement`, `honeycombing`).
2. Ensure that tile-level detections stitched back to the full frame do not get eliminated by aggressive global suppressor masks.

### Work Package 5: Evaluation & Regression Testing
1. Re-run inference across the 10 probe photos (`data/pilot/probe10`):
   ```powershell
   .venv-win\Scripts\python.exe -m vision.infer --folder data/pilot/probe10 --resume
   ```
2. Inspect the generated `<photo>.json` files and `<photo>_overlay.png`:
   - Verify that `IMG_4227.jpg` correctly identifies the bearing seat/pedestal (`C7.2`) or pier cap (`C2.8`), and does NOT label the bearing shelf as a `crash_barrier`.
   - Verify that genuine spalling and cracks on `IMG_4227` and `IMG_4231` are preserved.
   - Verify that the total number of valid distress detections increases while false `crash_barrier` counts decrease.

---

## 3. Strict Guardrails & Invariants

1. **Do NOT touch `kb/sp35_2024/*.yaml` or `sp35/scores.py`:**
   - Scoring formulas, weights, and intervals are signed and validated. Do not alter them.
2. **Deterministic & CPU-Testable:**
   - All filtering rules in `vision/postprocess.py` must remain pure NumPy/Python so they can be unit-tested on CPU without a GPU.
3. **Tests Must Remain Green:**
   - Run `uv run pytest` $\to$ all 69+ tests must pass.
   - Run `uv run --with pyyaml python tools/verify_phase0.py` $\to$ must report `VERIFY ALL PASS`.
4. **Memory Constraint:**
   - The vision loader in `vision/model.py` must retain its `bf16` encoder casting and VRAM cap ($< 3.4\text{ GB}$ peak allocation on RTX 2050).

---

## 4. Expected Deliverables

1. Updated `kb/prompts/sam3_prompts.yaml` with clean, non-conflicting prompts and per-class thresholds.
2. Updated `vision/postprocess.py` with component conflict resolution and orphan-rescue soft host gating.
3. New/updated unit tests in `tests/unit/test_postprocess.py` specifically asserting:
   - A pier cap is not suppressed by a parapet prompt.
   - An unhosted high-confidence crack is rescued rather than discarded.
   - Component NMS resolves overlapping `girder` and `crash_barrier` correctly.
4. Summary table of detection counts on `data/pilot/probe10` before vs. after your fix.

