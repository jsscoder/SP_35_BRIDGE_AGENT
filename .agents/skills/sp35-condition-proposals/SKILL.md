---
name: sp35-condition-proposals
description: Propose IRC:SP:35-2024 condition states I-V (1-5) with severity text for distresses that SAM 3.1 has already segmented, for one photo or a whole batch. Use after `vision.infer` when asked to grade, classify or mark distress severity/condition state, or to prepare states for engineer review before BHI. Never use it to finalise a state or to compute BHI directly.
---

# SP:35 condition-state proposals (I–V) from SAM 3.1 output

**Ground rule (SP:35 §7.4, p. 106):** "The method of inspection may be visual or NDT but the extent and severity are to be measured at the site." Everything this skill produces is a *proposal* from a photo. `final_state` is always `null`; the engineer confirms each state (AGENTS.md rule 3) before anything reaches `sp35.scores.assess` / BHI.

## Inputs
SAM 3.1 per-photo JSON from `vision.infer` (needs `distresses[].sp35` row, `host`, `score`, `bbox`; tiled distresses are included).
If there is no inference yet, run it first (GPU, `.venv-win`):
- one photo: `.venv-win\Scripts\python.exe -m vision.infer --photo <photo.jpg> --outdir <dir>`
- a batch:   `.venv-win\Scripts\python.exe -m vision.infer --manifest data/pilot/manifest.csv --src ../yugmi_images --outdir <dir> --resume`

## Run (CPU, no GPU needed)
- Single photo: `uv run python -m vision.condition_proposals --input <dir>\IMG_4227.json`
- Batch (folder): `uv run python -m vision.condition_proposals --input <dir>`
- Output (default `<dir>\conditions\`): `<photo>.conditions.json` per photo, `conditions.csv` (one row per distress row per host per photo), `summary.json` (counts by status and state, KB version and signed flag).
- **Mode.** Default `max`: every detection the KB can rate gets a state, whatever SAM knows about the bridge part (the part comes later from a part classifier). Each proposal says how its value was obtained in `basis` (`bridge-part outline`, `whole photo`, `count`) and lists what makes it weaker in `flags` (no part found, frame used as the part, mask covering the whole part, Table 7.3 twin used on the wearing course). `--strict` gives only states measured against an outlined part.
- Never in either mode: a mm / m / gradient value (crack width, scour, deflection) — those stay `needs_site_measurement`; and no "I" for photos where SAM found nothing (SAM missed 22 of 34 engineer-rated defects).
- **Known weakness (13 Oct 2026, 167-photo run):** max mode rates 97 photos (strict: 21), but on the 12 engineer-rated defects it reaches it said V every time where the engineer said I–III, because SAM's "spalling" masks cover a median 47 % of the frame. Treat `whole photo` states as triage, not BHI input, until SAM's masks are fixed; check with `tools/state_scorecard.py`.
- `--frame-as-member` (strict mode only) for close-ups whose whole frame is the tagged member; the method text says so.

## How a state is proposed (all numbers from `kb/sp35_2024/*.yaml` via `sp35.condition.condition_state`)
| Row unit (Table 7.2 / 7.3) | Value used | Status |
|---|---|---|
| `%` (spalling, honeycombing, rust, efflorescence, exposed rebar, …) | share of the *detected* host members' masks covered by the union of that distress's masks (overlaps counted once) | `proposed_visual_estimate` |
| `No.` (inadequate drainage spots, missing elements, potholes) | count in this photo (lower bound for the member) | `proposed_visual_estimate` |
| `No.` + `%` (vegetation) | both; the worse state wins (engine rule) | `proposed_visual_estimate` |
| `mm`, `m`, `m2`, gradient (crack width, scour, deflection) | none — measure at site with gauge/calliper/level | `needs_site_measurement` |
| host is only the photo tag (no member mask) | none unless `--frame-as-member` | `needs_member_extent` |
| distress masks cover more than `redetect_iou` of the host | none — usually the member itself | `needs_mask_check` |
| orphan host (`unassigned_structural`) | none — engineer assigns the member first | `needs_host` |
| Table 7.2 row on a wearing-course/approach host | none — reviewer picks the Table 7.3 row | `needs_table_7_3_row` |

Each proposal carries: distress class, SP:35 row and name, host member and its code, detections (score, bbox), value and method, proposed state with its name and the Table 5.1 severity summary, the engine's explanation (printed range), review notes, and `requires_engineer: true`.

Extent uses only the member masks the damage touches (three piers in a photo, spall on one: rated against that one), and overlapping damage masks are counted once.

## Reasoning trace (JSON `reasoning`, CSV `reasoning`)
Every proposal explains itself in five steps, each naming who owns it and where to fix it if wrong (`if_wrong`):
1. **SAM** — distress type, prompts that matched, score vs class threshold, mask area (SAM 3.1 inference).
2. **Member** — host member and code, SAM member score, % of the damage inside it; or photo tag / orphan rescue.
3. **Extent** — the arithmetic: damaged member px / member px = %, or the count, or why it can't be measured.
4. **KB** — the row, the machine band that holds the value, the resulting state, and the book's printed range.
5. **Result** — status and proposed state; the final state is the engineer's.

There is no model in steps 3–5: the state is a KB lookup. If a reviewer sees "15 % spalling -> II", step 4 shows the band used, so the fault is in the KB band or the engine (golden tests), never "the model".

## Manual review and where to fix
The CSV has empty columns for the engineer: `review_verdict`, `error_stage`, `correct_distress`, `correct_member`, `correct_value`, `correct_state`, `review_notes`. `error_stage` codes (also in `summary.json`):

| Code | Fix in |
|---|---|
| `ok` | nothing |
| `not_damage` | SAM 3.1: prompts / suppressors / thresholds, or fine-tuning |
| `wrong_distress_type` (e.g. debris clogging called spalling) | SAM 3.1: prompt library, or fine-tuning |
| `wrong_mask_extent` (outline too big/small) | SAM 3.1: prompts, tiling, fine-tuning |
| `wrong_member` (pier cap called crash barrier) | SAM 3.1 component prompts / `vision/postprocess.py` rules |
| `wrong_measurement` (masks right, value wrong) | this skill: `vision/condition_proposals.py` |
| `wrong_state_for_value` | KB band `kb/sp35_2024` (engineer sign-off) or `sp35/condition.py` |
| `needs_site_value` | not an error: measure on site |

After review: `uv run python -m vision.condition_proposals --tally <dir>\conditions\conditions.csv` writes `review_tally.json` with counts per code, where each is fixed, and the SAM->engineer relabels (e.g. spalling -> debris_clogging) — the labelled examples for later fine-tuning.

## Never
- Type an SP:35 range, weight or S_k into code or a prompt; add rows/aliases only in the KB with engineer sign-off.
- Estimate a crack width, scour depth or deflection from a photo, or fill `final_state`.
- Feed proposals into BHI without engineer confirmation; while `kb_signed` is false no score is released.
- Send photos or JSON to any external service.

## Report back
Give the output folder, `summary.json` counts by status and by state I–V, any state IV/V proposals by photo (these go to the urgent/critical lists only after confirmation), and the list of `needs_site_measurement` items as the site-measurement checklist. Say plainly that accuracy is unmeasured until the Phase 2 labelled baseline exists. In the 167-photo pilot run (v1 library) most state-V proposals came from SAM "spalling" masks covering whole painted girder/soffit/median faces, and some hosts were misclassified (crash_barrier on pier caps), so check the overlay before trusting any IV/V.

Tests: `uv run --group dev pytest tests/unit/test_condition_proposals.py`.
