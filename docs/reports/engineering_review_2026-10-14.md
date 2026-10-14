# Engineering review: labels, bands, calibrator (14 Oct 2026)

## Scope and limits
This is an AI desk review of repository files and data. It is **not** a certification. It does not certify, sign off, or declare IRC:SP:35-2024 or MoRTH conformance; only the certified bridge engineer can. I did not have the book ("printed" means the KB's stored text) and viewed no photos. Checks ran as read-only Python on KB `215716e83928` (draft-for-signoff).

## Finding 1: Label mapping (High)
**Evidence.** Of 49 records, 46 are usable. All 31 asset names map to the correct Table 7.1 code and group. Two records are rightly rejected: "Appurtenances — Span 3" (a group) and one map-only record. All 48 polygons lie inside the `imagePayload` width and height. All are 4-corner boxes (one covers 64% of its photo).
- **14 labels cannot result from any KB value.** Six are state II where II is null: 3 Crushing/Damaged (T7.2-16, `distress_bridge.yaml:120`) and 3 Undesired Restraints (T7.2-23, `:164`). Eight are visible defects marked I where I = Nil (spalling, staining ×3, efflorescence, degradation, patching, pop-out). Table 7.6 itself records "crushing/damaged" as II (`golden_examples.yaml:71`), so alias E13 is doubtful.
- **Bearings:** every "…Bearing…" asset goes to C7.1 (weight 50.0) (`ingest/platform_annotations.py:62-63`). Exposed rebar, pop-out and crushing there are pedestal defects (C7.2, 25.4).
- **Silent cross-table fallback** (`platform_annotations.py:92`): pop-out on a bearing → T7.3-27 (pavement, m²); deformation and unevenness on Alignment → Table 7.2 rows.
- **Joint damage → T7.3-24** (`anomaly_map.yaml:21`) borrows a pavement row for deck expansion joints. **Clogging** on a bearing → T7.2-25 (drainage-spot count).
- **Correct:** silting → T7.2-26 on pier/pier protection (T7.3-20 is for pavement joints); corner spalling → T7.2-08. Efflorescence/Leaching → T7.2-15 is acceptable; T7.2-09 (names leaching) is stricter at exactly 10%.

**Recommendation.** Get engineer rulings on the 14 labels before training or scoring. Add a C7.2 asset. Flag cross-table fallbacks as a `problem`. Use boxes to locate defects, not to measure extent.

## Finding 2: Band sanity (Medium)
**Evidence.** The loader rejects gaps, overlaps and non-ascending bands (`sp35/intervals.py:46-74`), and the KB loads. Every band matches its printed range. Gaps go to the worse state (E7), except four boundaries that go to the **better** state:
- T7.2-11 cracking 0.3 mm → II, though printed in II and III; 0.3 is a common gauge reading (`:90`).
- T7.2-15 efflorescence 10% → II, though printed "<10"/"11-25" (`:114`).
- T7.2-03 scour 0.25 m → II, though printed "<0.25"/"0.26-0.50" (`:37`).
- T7.3-13 ravelling 25% → II (`distress_approach.yaml:56`).

T7.4-TILT rounding puts exactly 1 in 75 in IV, not III (`:199`; errs safe).

**Recommendation.** The engineer confirms these boundaries, the vegetation reading (`:47`) and the pop-out and D-cracking m² units.

## Finding 3: Calibrator numbers not from the KB (High)
I ran the calibrator's KB-only path on synthetic masks. Line numbers are in `vision/condition_calibrator.py`.

| Line | Rule | Effect |
|---|---|---|
| 184-186, 198 | Extent ×0.05 for a "close-up", no host mask, or damage >40% of host | **Under-call.** 50% spall → 2.5% → II (KB: V); 40% rust → II |
| 205 | Rebar `max(6.0, …)` | An SP:35 band picked in code. Over-call below 5%; never above III (30% → III; KB: V) |
| 117, 206-208 | Staining set to 0% without an image; production never passes one (`condition_proposals.py:405`) | **Under-call.** Reports 0% |
| 238 | 0.12 mm per pixel crack width | Invented mm (AGENTS rule 4); ignores camera distance |
| 220 | mm/m rows default to state II | Under-call |
| 403 | 0.65/0.35 blend | Labels have no IV/V, so the trained model gives IV/V zero probability |
| 288 | crc32 row-ID feature | Learns each row's usual label, not severity |
| 84, 169, 215, 294, 390 | Floors, 0.55/0.45 blend (line 215 never runs), cost and prior constants | Minor |

Also: `tools/train_calibrator.py:615` hard-codes S_k. `CHAT_CONTEXT_FOR_CALIBRATION_AGENT.md:245-430` holds a wrong copy of Table 7.2. `condition_proposals.py:411-424` replaces "measure at site" with a calibrator state, which conflicts with §7.4. The scorecard's "0 IV/V under-calls" cannot be tested because no label is IV or V.

**Recommendation.** Don't use the calibrator for state proposals until the ×0.05, the 6.0 floor, the staining zero and the 0.12 mm/px are removed. Site rows should output "needs site measurement", with a visual hint at most.

## Finding 4: Proposed State IV gate (High; do not adopt)
It contradicts Table 7.2, which rates each row on its own measure:
- **Spalling (T7.2-08)** is IV above 10% and V above 20% on extent alone. Under the rule, a 30% spall would stay III and **leave the critical list** (`condition_states.yaml:37`).
- **Rust (T7.2-10)** is III above 5%, IV above 10%, V above 25%. A cap at II contradicts all three.
- **Exposed reinforcement (T7.2-12)** is IV above 10% and V above 15%, with no section-loss condition.
- **Rows without an area** (cracking in mm, deformation, scour, restraints) could never reach IV. A 3 mm girder crack would stay III.
- **Crushing (T7.2-16)** is IV from 5%. Rebar loss has nothing to do with bearing crushing.
- **Rebar loss** is its own row (T7.2-13). A "10%" in code would break rule 1.

The real concern, photos mistaking surface stains for corrosion, belongs in a review flag or the T7.2-09 vs T7.2-10 mapping, never in a cap.

## Finding 5: Under-call safety (High)
**Most at risk:**
- **Hidden or depth rows:** T7.2-07, T7.2-13, T7.2-14, T7.4-CORR, scour T7.2-03, marine borers T7.2-24, foundations C4.
- **Width or displacement rows:** cracking T7.2-11, deformation T7.2-05, leaning T7.2-21, tilt, settlement T7.3-09, lane drop, faulting.
- **Bearing function:** T7.2-16 and T7.2-23.
- **Counts:** one photo is a lower bound.
- **% rows:** a photo shows only part of a member.

**Recommendation.** Always measure on site every mm, m, m² or gradient row, plus T7.2-07, T7.2-13, T7.2-14, T7.4-CORR, C4 and bearing function. The AI flags but never lowers a state. Exposed rebar, crushing, or any crack on a girder, pier cap or bearing seat goes to the engineer, with the AI state shown as a lower bound.

## Questions for the certified bridge engineer
1. Is "Crushing / Damaged" crushing (T7.2-16, no II) or general damage? Can restraints be II?
2. Can a visible, insignificant defect be state I?
3. Should the four Finding 2 boundaries go to the worse state?
4. Which row rates expansion-joint damage and clogged joints or bearings?
5. Should bearing-pedestal defects go to C7.2? Should joints go to C5.6 or C6.3?
6. Which rows may carry an AI-proposed state without a site value?
