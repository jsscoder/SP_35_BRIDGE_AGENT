# Context Bundle for Training/Calibration Module Agent

This document bundles the 6 items requested by the calibration agent:
1. Root cause, log & resolution for the 3 errored images (`IMG_4190`, `IMG_4253`, `IMG_4277`).
2. Sample inference JSON (`IMG_4227.json`).
3. `kb/sp35_2024/condition_states.yaml`
4. `kb/sp35_2024/distress_bridge.yaml`
5. `vision/condition_proposals.py`
6. `tools/state_scorecard.py`

---

## 1. Traceback & Log for the 3 Errored Images

In the initial inference run across 170 images (`data/pilot/infer_all_v1/manifest.jsonl`), **164 succeeded, 3 had non-ok status**:

```json
[
  {
    "photo": "IMG_4190.jpg",
    "status": "error",
    "detail": "ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
  },
  {
    "photo": "IMG_4253.jpg",
    "status": "error",
    "detail": "ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
  },
  {
    "photo": "IMG_4277.jpg",
    "status": "error",
    "detail": "ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
  }
]
```

### Root Cause Analysis & Fix
- **Where:** `vision/postprocess.py` inside `resolve_components()`.
- **Cause:** When filtering out the losing component detection from `kept`, the code previously executed `kept = [k for k in kept if k != loser]`. Because `k` and `loser` are `Detection` dataclasses with NumPy boolean masks (`mask: np.ndarray`), Python attempted to evaluate `k.mask != loser.mask`, which returns a boolean array. In a list comprehension filter `if array:`, NumPy raises `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`.
- **Resolution:** Replaced with identity comparison:
  ```python
  kept = [k for k in kept if k is not loser]  # compare by identity, not numpy mask equality
  ```
- **Verification:** Re-running inference on `IMG_4190.jpg`, `IMG_4253.jpg`, and `IMG_4277.jpg` with this fix results in `status: ok` for all three.

---

## 2. Sample Inference JSON (`IMG_4227.json`)

```json
{
  "photo": "IMG_4227.jpg",
  "native_size": [3024, 4032],
  "status": "ok",
  "processed_size": [768, 1024],
  "scale": 0.25397,
  "sec": 17.3,
  "peak_alloc_gib": 2.34,
  "peak_reserved_gib": 2.86,
  "counts": {
    "components": 1,
    "distresses": 5,
    "suppressors": 2,
    "rejected": 21
  },
  "components": [
    {
      "class": "pier",
      "kind": "component",
      "sp35": "C2.7",
      "prompts": ["concrete pier column"],
      "score": 0.539,
      "bbox": [229, 0, 526, 592],
      "area_px": 159525,
      "frame": "photo",
      "rle": {"size": [1024, 768], "counts": "ePU7k01YO\\n0W3VN8J5K4L6J6J6J9Gl0PWOlK\\d0..."}
    }
  ],
  "distresses": [
    {
      "class": "spalling",
      "kind": "distress",
      "sp35": "T7.2-08",
      "prompts": ["broken concrete surface", "spalled concrete"],
      "score": 0.715,
      "bbox": [147, 584, 649, 730],
      "area_px": 54634,
      "frame": "photo",
      "host": {
        "class": "bearing_seat",
        "sp35": "C7.2",
        "source": "photo tag (component not detected)",
        "inside_fraction": 1.0,
        "host_coverage": 0.069
      },
      "rle": {"size": [1024, 768], "counts": "Zcc48fo04K4M4M2M4dQOZOXm0j0^ROKTm0..."}
    },
    {
      "class": "crack",
      "kind": "distress",
      "sp35": "T7.2-11",
      "prompts": ["concrete crack", "crack in concrete"],
      "score": 0.613,
      "bbox": [591, 152, 673, 376],
      "area_px": 1521,
      "frame": "photo",
      "host": {
        "class": "bearing_seat",
        "sp35": "C7.2",
        "source": "photo tag (component not detected)",
        "inside_fraction": 1.0,
        "host_coverage": 0.002
      },
      "rle": {"size": [1024, 768], "counts": "b[_b05jo01N3N1N2O1O0O10O1O001O1O1O01..."}
    },
    {
      "class": "crushing",
      "kind": "distress",
      "sp35": "T7.2-16",
      "prompts": ["crushed concrete edge"],
      "score": 0.559,
      "bbox": [0, 708, 768, 746],
      "area_px": 14805,
      "frame": "photo",
      "host": {
        "class": "bearing_seat",
        "sp35": "C7.2",
        "source": "photo tag (component not detected)",
        "inside_fraction": 1.0,
        "host_coverage": 0.019
      },
      "rle": {"size": [1024, 768], "counts": "Zf0d0\\o0001O000O2O000000O1000000000..."}
    }
  ],
  "suppressors": [
    {
      "class": "poster",
      "kind": "suppression",
      "sp35": null,
      "prompts": ["signboard", "poster", "banner"],
      "score": 0.848,
      "bbox": [0, 737, 768, 1024],
      "area_px": 215076,
      "frame": "photo"
    },
    {
      "class": "rope",
      "kind": "suppression",
      "sp35": null,
      "prompts": ["rope"],
      "score": 0.512,
      "bbox": [0, 404, 768, 522],
      "area_px": 9565,
      "frame": "photo"
    }
  ],
  "rejected": [
    {
      "class": "spalling",
      "kind": "distress",
      "sp35": "T7.2-08",
      "prompts": ["spalled concrete"],
      "score": 0.375,
      "bbox": [0, 0, 257, 573],
      "area_px": 110691,
      "frame": "photo",
      "reject_reason": "score 0.38 below class threshold 0.45"
    },
    {
      "class": "honeycombing",
      "kind": "distress",
      "sp35": "T7.2-06",
      "prompts": ["porous voids in concrete surface"],
      "score": 0.535,
      "bbox": [147, 584, 649, 730],
      "area_px": 54824,
      "frame": "photo",
      "reject_reason": "same region as spalling (higher score)"
    }
  ],
  "distress_status": "proposed, unverified (no evaluation baseline yet)",
  "photo_tag": {"label": "bearing_seat", "sp35": "C7.2"},
  "prompt_library": {"version": "v1", "sha12": "fc947068c9e0"}
}
```

---

## 3. `kb/sp35_2024/condition_states.yaml`

```yaml
# IRC:SP:35-2024 condition states (Table 5.1, pp. 89-91), reduction coefficients (§7.5, p.130)
# and BHI star rating (p.132). Descriptions below are short summaries; see Table 5.1 for the full text.

kb_version: sp35_2024-kb0
status: draft-for-signoff

states:
  - id: I
    name: Excellent
    s_k: 0.0
    summary: "Sound; only constructional deficiencies; extent nil or insignificant; routine maintenance only."
  - id: II
    name: Good
    s_k: 0.25
    summary: "More than satisfactory; minor extent, low severity; low-cost regular maintenance; specialised repairs at convenience."
  - id: III
    name: Fair
    s_k: 0.5
    summary: "Satisfactory but durability reduced; major extent, medium severity; specialised maintenance and repairs needed soon; some monitoring."
  - id: IV
    name: Poor
    s_k: 0.75
    summary: "Structurally deficient; strength, stability or traffic safety no longer assured; restriction of use may be needed; rehabilitation on a programme basis."
  - id: V
    name: Critical
    s_k: 1.0
    summary: "Weak; risk of partial or total failure; immediate propping or closure may be required; repair or replacement immediately."

# Eq. 3 contribution of one distress: 100 - 100 * s_k  ->  I 100, II 75, III 50, IV 25, V 0

rules:
  no_distress_noticed: {cs_j: 100, report_as: "Nil / I"}          # p.131
  element_absent: {cs_j: 100, report_as: "NA"}                     # p.131
  foundations_not_visible: engineer_judgement_required             # p.131, from shift/tilt/deformation above
  other_distress_types: map_with_table_5_1_descriptions            # p.131: types not in Tables 7.2/7.3
  ndt_results: map_to_same_five_states                             # p.91 and p.131
  critical_list: any_distress_in_state_V                           # p.130-131, independent of BHI
  urgent_list: any_distress_in_state_IV

# Star rating — printed bands (p.132): <20 Critical/1; 21-40 Poor/2; 41-60 Fair/3; 61-80 Good/4; 81-100 Excellent/5.
# Text p.135: BHI "more than 80" -> excellent, five star. Gaps resolved by decision E8.
star_rating:
  decision: E8
  bands:
    - {stars: 5, condition: Excellent, interval: "(80,100]"}
    - {stars: 4, condition: Good,      interval: "(60,80]"}
    - {stars: 3, condition: Fair,      interval: "(40,60]"}
    - {stars: 2, condition: Poor,      interval: "(20,40]"}
    - {stars: 1, condition: Critical,  interval: "[0,20]"}
```

---

## 4. `kb/sp35_2024/distress_bridge.yaml`

```yaml
# IRC:SP:35-2024 Table 7.2 — distress types and ranges for condition states (concrete bridges)
# Source: book pp. 106-107; reference photos in Table 7.4 (pp. 111-117; p.117 missing from our scan, E12).
#
# `printed`  = the book's text for states I..V, kept for display and audit.
# `bands`    = machine intervals per state (decision E7, needs engineer sign-off):
#              "=0" exact zero (printed "Nil"); "(a,b]", "[a,b)", "[a,b]", "(a,b)"; "inf" = no upper bound;
#              null = state cannot occur for this distress.
# Gaps and overlaps in the printed ranges are resolved so every value maps to exactly one state:
#   printed "<a" for II followed by "a-b" for III  -> II "(0,a)",  III "[a,b]"
#   printed "a-b" followed by "(b+1)-c"            -> next band starts "(b,"
# `criteria` (instead of unit/bands) = two measures; the WORSE resulting state applies.
# Units: % = share of the member's area/length affected; mm, m = measured; No. = count.

kb_version: sp35_2024-kb0
source: "IRC:SP:35-2024, Table 7.2, pp. 106-107; Table 7.4, pp. 111-117"
table: "7.2"
status: draft-for-signoff
s_k: {I: 0.0, II: 0.25, III: 0.5, IV: 0.75, V: 1.0}

distresses:
  - id: T7.2-01
    name: Erosion
    unit: "%"
    printed: {I: Nil, II: "0-10", III: "11-20", IV: "21-30", V: ">30"}
    bands:   {I: "=0", II: "(0,10]", III: "(10,20]", IV: "(20,30]", V: "(30,inf)"}

  - id: T7.2-02
    name: Abrasion
    unit: "%"
    printed: {I: "<10", II: "10-20", III: "21-40", IV: "41-50", V: ">50"}
    bands:   {I: "[0,10)", II: "[10,20]", III: "(20,40]", IV: "(40,50]", V: "(50,inf)"}

  - id: T7.2-03
    name: Scour (depth from ground level)
    unit: m
    printed: {I: Nil, II: "<0.25", III: "0.26-0.50", IV: "0.51-1.0", V: ">1.0"}
    bands:   {I: "=0", II: "(0,0.25]", III: "(0.25,0.5]", IV: "(0.5,1.0]", V: "(1.0,inf)"}

  - id: T7.2-04
    name: Vegetation
    unit: "%"
    printed: {I: Nil, II: "<10", III: "10-25", IV: "26-50", V: ">50"}
    bands:   {I: "=0", II: "(0,10]", III: "(10,25]", IV: "(25,50]", V: "(50,inf)"}

  - id: T7.2-05
    name: Deflection
    unit: mm
    printed: {I: "<1", II: "1-5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "[0,1)", II: "[1,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-06
    name: Honeycombing
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-07
    name: Delamination
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-08
    name: Spalling
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-09
    name: Leaching
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-10
    name: Rust staining
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-11
    name: Cracking
    criteria:
      - name: width
        unit: mm
        printed: {I: "<0.1", II: "0.1-0.3", III: "0.3-1.0", IV: "1.1-2.0", V: ">2.0"}
        bands:   {I: "[0,0.1)", II: "[0.1,0.3]", III: "(0.3,1.0]", IV: "(1.0,2.0]", V: "(2.0,inf)"}
      - name: length
        unit: m
        printed: {I: "<0.5", II: "0.5-1.0", III: "1.1-2.0", IV: "2.1-3.0", V: ">3.0"}
        bands:   {I: "[0,0.5)", II: "[0.5,1.0]", III: "(1.0,2.0]", IV: "(2.0,3.0]", V: "(3.0,inf)"}

  - id: T7.2-12
    name: Exposed reinforcement
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-13
    name: Reduction in rebar area
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-14
    name: Concrete disintegration
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-15
    name: Efflorescence
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-16
    name: Crushing
    unit: "%"
    printed: {I: Nil, II: null, III: "<5", IV: "6-10", V: ">10"}
    bands:   {I: "=0", II: null, III: "(0,5]", IV: "(5,10]", V: "(10,inf)"}

  - id: T7.2-17
    name: Cavities
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-18
    name: Distortion
    unit: mm
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-19
    name: Loosening of joint / connection
    unit: No.
    printed: {I: Nil, II: "1", III: "2-5", IV: "6-10", V: ">10"}
    bands:   {I: "=0", II: "=1", III: "[2,5]", IV: "[6,10]", V: "(10,inf)"}

  - id: T7.2-20
    name: Missing elements
    unit: No.
    printed: {I: Nil, II: "1", III: "2-3", IV: "4-5", V: ">5"}
    bands:   {I: "=0", II: "=1", III: "[2,3]", IV: "[4,5]", V: "(5,inf)"}

  - id: T7.2-21
    name: Cracks in weld
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-22
    name: Corrosion
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-23
    name: Movement of joint
    unit: mm
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-24
    name: Deformation of pad
    unit: "%"
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  - id: T7.2-25
    name: Inadequate drainage
    unit: No.
    printed: {I: Nil, II: "1", III: "2-5", IV: "6-10", V: ">10"}
    bands:   {I: "=0", II: "=1", III: "[2,5]", IV: "[6,10]", V: "(10,inf)"}

  - id: T7.2-26
    name: Failure of slope protection
    unit: "%"
    printed: {I: Nil, II: "<10", III: "11-25", IV: "26-50", V: ">50"}
    bands:   {I: "=0", II: "(0,10]", III: "(10,25]", IV: "(25,50]", V: "(50,inf)"}

  - id: T7.2-27
    name: Water-logging
    unit: "%"
    printed: {I: Nil, II: "<10", III: "11-25", IV: "26-50", V: ">50"}
    bands:   {I: "=0", II: "(0,10]", III: "(10,25]", IV: "(25,50]", V: "(50,inf)"}

  - id: T7.2-28
    name: Settlement
    unit: mm
    printed: {I: Nil, II: "<5", III: "6-10", IV: "11-20", V: ">20"}
    bands:   {I: "=0", II: "(0,5]", III: "(5,10]", IV: "(10,20]", V: "(20,inf)"}

  # Added rows from Table 7.4 reference photo plates (Decision E11)
  - id: T7.4-CORR
    name: Corrosion depth (pitting excluded)
    unit: mm
    printed: {I: "<0.1", II: "0.1-0.3", III: "0.31-0.5", IV: "0.51-1.0", V: ">1.0"}
    bands:   {I: "[0,0.1)", II: "[0.1,0.3]", III: "(0.3,0.5]", IV: "(0.5,1.0]", V: "(1.0,inf)"}

  - id: T7.4-TILT
    name: Tilting
    unit: gradient
    printed: {I: Nil, II: null, III: "< 1 in 100", IV: "1 in 75", V: "> 1 in 60"}
    bands:   {I: "=0", II: null, III: "(0, 0.010]", IV: "(0.010, 0.0133]", V: "(0.0133, inf)"}

# Aliases used in Table 7.6 or inspector notes (Decision E13)
aliases:
  clogging: T7.2-25
  missing: T7.2-20
  rusting: T7.2-22
  crushing/damaged: T7.2-16
  staining: T7.2-10
  corrosion: T7.2-22
  vegetation: T7.2-04
  rutting: T7.3-04
  erosion: T7.2-01
  silting: T7.2-25
```

---

## 5. `vision/condition_proposals.py` (Core Logic Overview)

```python
"""Proposed SP:35 condition states (I-V) for SAM 3.1 distress detections.
Evaluates damage % vs host member area, handles count rows, and flags site-measurement rows.
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np

from sp35.condition import condition_state
from sp35.kb import KnowledgeBase, load_kb
from .postprocess import load_prompt_library

COUNT_UNIT = "No."
SITE_UNITS = ("mm", "m", "m2", "gradient")
SCORED_STATUSES = ("ok", "ok_reduced")

def decode_rle(rle: dict) -> np.ndarray:
    """COCO RLE -> bool mask in pure Python/NumPy without pycocotools."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        runs, pos = [], 0
        while pos < len(counts):
            x, k, more = 0, 0, True
            while more:
                c = ord(counts[pos]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                pos += 1
                k += 1
                if not more and c & 0x10:
                    x |= -1 << (5 * k)
            if len(runs) > 2:
                x += runs[-2]
            runs.append(x)
        counts = runs
    mask = np.zeros(h * w, dtype=bool)
    cursor = 0
    val = False
    for c in counts:
        if val:
            mask[cursor:cursor + c] = True
        cursor += c
        val = not val
    return mask.reshape((w, h)).T

def propose_condition_for_group(detections: List[dict], host_mask: Optional[np.ndarray],
                                row_id: str, kb: KnowledgeBase, lib) -> dict:
    """Evaluates union mask of distresses on a single host and calls sp35.condition.condition_state."""
    row = kb.distress_row(row_id)
    unit = row.unit
    
    # 1. Units requiring physical scale
    if unit in SITE_UNITS or row.criteria:
        return {
            "proposed_state": None,
            "reason": "needs_site_measurement",
            "explanation": f"{row.name} ({unit or 'criteria'}) requires site measurement"
        }
    
    # 2. Count-based distresses
    if unit == COUNT_UNIT:
        count = len(detections)
        res = condition_state(kb, row_id, count)
        return {
            "proposed_state": res.state,
            "value": count,
            "unit": COUNT_UNIT,
            "explanation": res.explanation
        }
        
    # 3. Area percentage distresses
    if unit == "%":
        dist_masks = [decode_rle(d["rle"]) for d in detections if "rle" in d]
        dist_union = np.logical_or.reduce(dist_masks)
        
        if host_mask is not None:
            host_area = int(host_mask.sum())
            dist_area = int(np.logical_and(dist_union, host_mask).sum())
            pct = round((dist_area / max(1, host_area)) * 100.0, 2)
            basis = "host_member_mask"
        else:
            frame_area = dist_union.size
            dist_area = int(dist_union.sum())
            pct = round((dist_area / frame_area) * 100.0, 2)
            basis = "whole_photo_frame"
            
        res = condition_state(kb, row_id, pct)
        return {
            "proposed_state": res.state,
            "value": pct,
            "unit": "%",
            "basis": basis,
            "explanation": res.explanation
        }
```

---

## 6. `tools/state_scorecard.py` (Evaluation Harness Overview)

```python
"""Condition-state scorecard: predicted states vs the bridge engineer's labels.
Finds SAM distress covering the ground-truth polygon best, evaluates exact and within-1 accuracy,
under-calls, and quadratic weighted kappa.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from ingest.platform_annotations import Label, load_labels
from sp35.condition import condition_state
from sp35.kb import STATES, KnowledgeBase, load_kb
from vision.condition_proposals import decode_rle

RANK = {s: i for i, s in enumerate(STATES)}

def polygon_mask(label: Label, result: dict) -> np.ndarray:
    w, h = result["processed_size"]
    s = result["scale"]
    img = Image.new("1", (w, h))
    ImageDraw.Draw(img).polygon([(x * s, y * s) for x, y in label.polygon], fill=1)
    return np.asarray(img, dtype=bool)

def quadratic_weighted_kappa(matrix: np.ndarray) -> float:
    n = len(STATES)
    total = matrix.sum()
    if total == 0:
        return 0.0
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / total
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    observed_disagreement = (weights * matrix).sum() / total
    expected_disagreement = (weights * expected).sum() / total
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected_disagreement)
```

