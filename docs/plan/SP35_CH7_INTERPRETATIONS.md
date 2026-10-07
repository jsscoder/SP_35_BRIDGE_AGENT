# SP:35 Chapter 7 — interpretations register

IRC:SP:35-2024, *Manual for Bridge Management, Inventory, Inspection and Maintenance (First Revision)*.
Reading Chapter 7 closely turned up the items below, where the book is ambiguous or its printed arithmetic disagrees with its own equations.
Each item records how the software behaves. The bridge engineer signs this register at gate G0. Every report states which KB version (and therefore which decisions) it used.

Status: **draft for sign-off** (prepared 7 Oct 2026). Values were recomputed with `tools/sp35_reference_check.py`.

| ID | Where (book page) | Issue | Decision in the software | Effect |
|---|---|---|---|---|
| E1 | p.130, example under Eq. 3 | Deck slab with distresses in states II, III, II, I is printed as (150+50+0)/4 = 50.0. Eq. 3 gives (150+50+100)/4 = 75.0; the book scores state I as 0 instead of 100. | Follow Eq. 3. | Golden test expects 75.0. |
| E2 | Table 7.6, C1 and C6 (pp.133-135) | Component indices appear divided by 100 instead of ΣW_j = 99.9. | Follow Eq. 2: divide by the actual sum of weights. | C1 75.73 vs printed 75.65; C6 89.86 vs 89.77. |
| E3 | Table 7.6, C2 | CI printed 95.03; recomputed from the printed inputs, 95.44. | Follow Eq. 2. | Bridge BHI 95.16 vs printed 95.06; inside the ±0.2 golden tolerance. |
| E4 | Table 7.6 entries | Guide bunds list "vegetation, II" but CS 100; light posts list "II" but CS 50; the narrative says no component is in state V although the railing (C1.4) is rated V. | Score strictly by the recorded states. Always list state-V items as critical. | Strict-state BHI 95.01 (book 95.06). Critical list = [C1.4]. |
| E5 | Table 7.1, column 6 | C1.7 printed 0.31 (3.6 × 10.3 / 100 = 0.37); C3.3 printed 1.24 (5.8 × 8.4 / 100 = 0.49). | Never use column 6. Use columns 3 and 5. | None on BHI. |
| E6 | Table 7.1 | Main weights total 99.9; groups C1 and C6 total 99.9. | Normalise by the sum, as Eq. 1 and Eq. 2 already do. | None. |
| E7 | Tables 7.2, 7.3 | Printed ranges overlap or leave gaps (cracking 0.3 mm sits in II and III; 1.0–1.1 mm uncovered; loose joints and drainage spots "2" uncovered; integer-style bands such as 10/11 applied to decimals). | Explicit interval per state in `kb/sp35_2024/distress_*.yaml`, printed text kept beside it. Uncovered counts go to the worse state. Dual criteria take the worse state. | Every value maps to exactly one state (checked by the reference script). |
| E8 | Star table, p.132 | Bands printed <20, 21–40, 41–60, 61–80, 81–100 leave gaps; text on p.135 says ">80 is excellent". | (80,100] 5★, (60,80] 4★, (40,60] 3★, (20,40] 2★, [0,20] 1★. | Boundary values (e.g. 20.5, 40.0) have a defined star. |
| E9 | Table 7.9, factor F7 | Bridge condition classed 50 and 25 for BHI 75.69 and 85.66. Under the star bands these BHIs are Good (25) and Excellent (0). | Default: derive F7 from the star bands. Golden test keeps the book's 50/25 to reproduce 44.89 / 33.21. | With star bands the example MPIs would be 37.35 / 25.77. |
| E10 | Tables 7.7 vs 7.9 | F3.3 diversion cost 8.3 vs 8.29; F3.5 optimal use of funds 32 vs 31.69. | Use the Table 7.9 values (sub-factors then total 100). | Negligible. |
| E11 | Table 7.4 photo plates | Two rows not in Table 7.2: corrosion depth with pitting excluded (mm: <0.1, 0.1–0.3, 0.31–0.5, 0.51–1.0, >1) and tilting (1 in 100, 1 in 75, 1 in 60). | Added as rows T7.4-CORR and T7.4-TILT. | Two more measurable distresses. |
| E12 | Our scan of the book | Book page 117 (part of the Table 7.4 photo plates) is missing from the PDF; page offsets also drift near the end. | Copy p.117 from the printed book into the reference set. | Reference photo library complete. |
| E13 | Table 7.6 | Uses names that are not Table 7.2 rows: clogging, missing, rusting, crushing/damaged, staining, not visible. | Alias list in `distress_bridge.yaml` (clogging → inadequate drainage; missing → missing elements; rusting → rust; crushing/damaged → crushing; staining → staining/scaling/leaching; not visible → engineer judgement). | Example data and field names map to measurable rows. |
| E14 | §7.5.1, p.131 (span rule) | SP:35 says BHI is computed per span (end spans with abutment, intermediate spans with piers) and the bridge takes the lowest, but not how the two supports of a span combine or where approaches and waterway belong. | Per sub-component, a span takes the worse of its two supports (C2, C4, C7). Approaches (C1) belong to end spans only. Waterway (C3) applies to every span. Components outside a span are excluded from that span's Eq. 1 sums. | Bridge BHI is conservative; intermediate spans are not inflated by approach scores. |

## Sign-off

| Role | Name | Decision (accept / change) | Date | Signature |
|---|---|---|---|---|
| Bridge engineer (domain owner) | | | | |
| Co-founder (product) | | | | |
