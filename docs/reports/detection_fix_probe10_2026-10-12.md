# Detection fix — prompt library v1 on probe10 (12 Oct 2026)

Spec: `docs/plan/OPUS_DETECTION_FIX_PROMPT.md`. RTX 2050, bf16 loader unchanged.
Before = `data/pilot/infer_all` (library v0, same 10 photos, same tags). After = `data/pilot/infer_fix_2026-10-12` (v1).
**No ground truth exists**, so this measures what survives, not accuracy. Every distress is still "proposed, unverified".

## Counts (10 probe photos)

| | Before (v0) | After (v1) |
|---|---|---|
| Components | 33 | 28 |
| crash_barrier | 8 | 7 (IMG_4160 3, 4231 1, 4238 1, 4297 2) |
| bearing_seat | 0 | 1 |
| approach_wall / kerb | 3 / 2 | 1 / 0 (lost conflicts to real barriers on road photo 4297) |
| Distresses | 33 | 88 |
| crack | 12 | 53 (27 scored 0.35–0.45; 35 on IMG_4185 alone) |
| spalling | 17 | 28 |
| exposed_reinforcement | 0 | 2 (both look false: a banner on 4231, a thin rod on 4238) |
| Hosted by detected component / photo tag / orphan rescue | — | 23 / 64 / 1 |
| Peak alloc / reserved (max) | 2.38 / 2.89 GiB | 2.45 / 2.78 GiB (≤ 3.4 GB) |
| Seconds per photo | 16.1–16.5 (87 prompts) | 17.4–23.4 (93 prompts) |

## What changed and what it did
- **crash_barrier prompts** ("concrete parapet" led 83 of 172 barriers in the 167-photo run): the replacement "concrete crash barrier" scores *higher* on the same pier caps (IMG_4231 0.56 → 0.74, IMG_4195 0.54 → 0.84). The phantom is gone from IMG_4227 and 4290, but **not** from IMG_4231, where no pier_cap mask competes. Text prompts alone do not separate a pier cap from a barrier here; exemplar boxes (WP2.5) or fine-tuning (Phase 4) are needed.
- **Component conflicts** (mask IoU ≥ 0.5): photo tag first, then a score gap > 0.15, then `component_group_rank`. Tag-first was chosen after the first run, where score-first let a 0.84 barrier take the region of the real pier and bearing seat on pier-tagged IMG_4195; with tag-first the pier is kept.
- **Orphan rescue** (crack, spalling, rebar at score ≥ 0.5): fired once on probe10, because all 10 probe photos are tagged and the photo-tag fallback runs first. It matters on the 157 untagged pilot photos (e.g. the real projecting bars on IMG_4226 and 4266 were lost only for want of a host). Host `unassigned_structural`, `sp35: null` (not "UNKNOWN", so nothing can mistake it for a code), `review_required: true`.
- **Per-class thresholds**: SAM's own floor is now 0.35 for every prompt. The crack threshold of 0.35 adds real-looking road cracks (IMG_4297) and a diagonal crack (4160), but also paint edges, a construction joint and, on IMG_4185, 35 "cracks" that are cables or bars crossing the frame. **Recommendation: crack 0.45** (drops the 27 lowest) until WP2.5 measures it.
- **Co-occurring distresses** (crack + leaching, spalling + rebar, …): both kept, the lower one flagged for review.

## Not changed, with reason
- `metal_railing` still suppresses rebar: of the 5 rebar masks it removed in the 167-photo run, 3 were ornamental fences, 1 a cable, 1 a construction cage.
- Whole-component re-detection rule: the rejected "spalling" masks are near-whole members (41 of 47 merged into them were ≥ 0.6 of the giant's area), and a bbox/aspect test cannot tell a crushed small pedestal from a re-detection.
- Bearings: no bearing mask reached 0.5; the highest seen was 0.57 (bearing_seat, IMG_4195, lost a conflict to the tagged pier). Needs exemplar prompts and labels.
- `kb/sp35_2024/*.yaml`, `sp35/`, the loader and VRAM cap: untouched.
