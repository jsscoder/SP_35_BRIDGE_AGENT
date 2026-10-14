"""Proposed SP:35 condition states (I-V) for SAM 3.1 distress detections.

Reads the per-photo JSON written by `vision.infer` (one file or a folder) and,
for each distress row on each host component in a photo, proposes a condition
state through the deterministic engine (`sp35.condition.condition_state`) and
the bands in `kb/sp35_2024/*.yaml`. No SP:35 number lives in this file.

SP:35 §7.4: "the extent and severity are to be measured at the site". So every
state here is a PROPOSAL from a photo, never final (`final_state` is always
null) and always for engineer review:
- % rows (spalling, honeycombing, rust, ...): extent = share of the detected
  host members' masks covered by the union of that distress's masks ("visual
  estimate, % of the host visible in this photo"); damage over more than the
  prompt library's `redetect_iou` of the member is sent for a mask check;
- No. rows (drainage spots, missing elements): count in this photo (a lower
  bound for the member);
- mm / m / gradient rows (crack width, scour, deflection): no state, "measure
  at site" — a photo has no scale until Phase 2;
- no state either when the host is an orphan (`unassigned_structural`), when a
  Table 7.2 row sits on a wearing-course host (Table 7.3 applies), or when the
  host is only the photo's tag (no member mask; `--frame-as-member` treats a
  close-up frame as the member and labels it so).

    uv run python -m vision.condition_proposals --input data/pilot/infer_all_v1
    uv run python -m vision.condition_proposals --input data/pilot/infer_all_v1/IMG_4227.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from sp35.condition import condition_state
from sp35.kb import KnowledgeBase, load_kb

from .postprocess import load_prompt_library

PROMPTS = Path(__file__).resolve().parents[1] / "kb/prompts/sam3_prompts.yaml"

COUNT_UNIT = "No."
SITE_UNITS = ("mm", "m", "m2", "gradient")
SCORED_STATUSES = ("ok", "ok_reduced")


def decode_rle(rle: dict) -> np.ndarray:
    """COCO RLE (compressed string or list counts) -> bool mask, without pycocotools
    (same algorithm as pycocotools' rleFrString), so this module runs on CPU anywhere."""
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
    flat = np.zeros(h * w, dtype=bool)
    ends = np.cumsum(counts)
    for start, end in zip(ends[0::2], ends[1::2]):
        flat[start:end] = True
    return flat.reshape((h, w), order="F")


def _extent(result: dict, dets: List[dict]) -> Optional[dict]:
    """Pixels of the damaged member(s) covered by the union of these distress masks
    (overlapping detections counted once). Only member masks the damage touches count:
    with three piers in a photo, a spall on one is rated against that pier, not all three.
    None if a mask is missing or in another frame."""
    host = dets[0]["host"]["class"]
    hosts = [c for c in result.get("components", []) if c["class"] == host and c.get("rle")]
    if not hosts or any(d.get("frame", "photo") != "photo" or not d.get("rle") for d in dets):
        return None
    masks = [decode_rle(c["rle"]) for c in hosts]
    if any(d["rle"]["size"] != list(masks[0].shape) for d in dets):
        return None
    damage = np.logical_or.reduce([decode_rle(d["rle"]) for d in dets])
    touched = [m for m in masks if np.logical_and(m, damage).any()]
    if not touched:
        return None
    host_mask = np.logical_or.reduce(touched)
    on, member = int(np.logical_and(damage, host_mask).sum()), int(host_mask.sum())
    return {"pct": round(100.0 * on / member, 2), "damage_on_member_px": on, "member_px": member,
            "damage_px": int(damage.sum()), "member_masks": len(touched), "member_masks_in_photo": len(hosts)}


def _frame_extent(result: dict, dets: List[dict]) -> dict:
    """% of the whole photo covered by the union of these distress masks, for damage
    whose bridge part SAM did not outline (max mode): the frame stands in for the part."""
    by_frame = defaultdict(list)
    for d in dets:
        if d.get("rle"):
            by_frame[tuple(d["rle"]["size"])].append(decode_rle(d["rle"]))
    if by_frame:
        size, masks = max(by_frame.items(), key=lambda kv: len(kv[1]))
        damage = np.logical_or.reduce(masks)
        frame = int(size[0] * size[1])
        return {"how": "photo frame as member", "pct": round(100.0 * float(damage.sum()) / frame, 2),
                "damage_px": int(damage.sum()), "frame_px": frame}
    w, h = result.get("processed_size") or (0, 0)
    if w and h:
        area = sum(int(d.get("area_px", 0)) for d in dets)
        return {"how": "photo frame as member", "pct": round(min(100.0, 100.0 * area / (w * h)), 2),
                "damage_px": area, "frame_px": w * h}
    return {"how": "photo frame as member",
            "pct": round(min(100.0, 100.0 * sum(d["host"].get("host_coverage", 0.0) for d in dets)), 2)}


def _table_7_3_twin(kb: KnowledgeBase, row):
    """The Table 7.3 row of the same name (Vegetation, Erosion, Missing elements...), if any."""
    if kb.table_of(row) == "7.3":
        return row
    return next((r for r in kb.rows if kb.table_of(r) == "7.3" and r.name.casefold() == row.name.casefold()), None)


def _host_key(d: dict) -> tuple:
    h = d.get("host") or {}
    return h.get("class"), h.get("sp35"), h.get("source")


def _group(distresses: List[dict]) -> Dict[tuple, List[dict]]:
    """One SP:35 rating per distress row per host in a photo: SP:35 rates the
    member, so every spall on the same pier counts towards one extent."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for i, d in enumerate(distresses):
        groups[(d["sp35"], d["class"]) + _host_key(d)].append(dict(d, _index=i))
    return groups


def _measure(row, dets: List[dict], source: str, frame_as_member: bool, whole_member: float,
             result: Optional[dict] = None, mode: str = "strict") -> tuple:
    """(value for condition_state or None, method text or reason, facts for the reasoning trace).
    `facts["flags"]` lists what makes a max-mode value weaker than a measured one."""
    units = [m.unit for m in row.measures()]
    count = len(dets)
    pct, pct_method, facts = None, None, {}
    flags: List[str] = []
    if "%" in units and source == "detected":
        ext = _extent(result or {}, dets)
        if ext is not None:
            pct, facts = ext["pct"], dict(ext, how="union of masks")
            pct_method = "visual estimate: % of the host mask covered in this photo (union of the masks)"
        else:
            cover = [round(100.0 * d["host"].get("host_coverage", 0.0), 2) for d in dets]
            pct = round(min(100.0, sum(cover)), 2)
            facts = {"how": "sum of per-mask coverage", "per_mask_pct": cover, "pct": pct}
            pct_method = ("visual estimate: sum of per-detection % of the host mask in this photo "
                          "(masks unavailable; overlaps may be counted twice)")
        biggest = max(100.0 * d["host"].get("host_coverage", 0.0) for d in dets)
        facts["largest_single_mask_pct"] = round(biggest, 2)
        if max(pct, biggest) > 100.0 * whole_member:
            # same test as postprocess `redetect_iou`: damage over most of the member is usually the member itself
            msg = (f"distress masks cover {max(pct, biggest):.0f}% of the {dets[0]['host']['class']} mask: "
                   "likely the member itself, not a defect on it - engineer checks the mask")
            if mode == "strict":
                return None, msg, facts
            flags.append(f"damage mask covers {max(pct, biggest):.0f}% of the {dets[0]['host']['class']}: "
                         "may be the whole part, not damage on it")
    elif "%" in units and (frame_as_member or mode == "max"):
        # no outline of the bridge part (photo tag or no part found): the whole photo stands in for it
        facts = _frame_extent(result or {}, dets)
        pct = facts["pct"]
        what = "the tagged member" if source.startswith("photo tag") else "the part (no bridge part found)"
        pct_method = f"visual estimate: % of the photo frame, frame taken as {what}"
        flags.append("no outline of the bridge part: extent taken over the whole photo")
    facts["flags"] = flags
    values, methods = {}, []
    for unit in units:
        if unit == COUNT_UNIT:
            values[unit] = count
            facts["count"] = count
            methods.append(f"{count} counted in this photo (lower bound for the member)")
        elif unit == "%" and pct is not None:
            values[unit] = pct
            methods.append(pct_method)
        elif unit == "%":
            return None, ("extent needs the member's outline: the host is only the photo tag "
                          "(see --frame-as-member)"), facts
        else:
            return None, f"{row.name} is in {unit}: measure at site (SP:35 §7.4); a photo has no scale yet", facts
    value = values[units[0]] if len(units) == 1 else values
    return value, "; ".join(methods), facts


# Where a reviewer's verdict sends the fix. The skill has no model: the state is a KB
# lookup, so a wrong state for a correct value can only come from the KB or the engine.
ERROR_STAGES = {
    "ok": "nothing to fix",
    "not_damage": "SAM 3.1 inference: false positive (paint, shadow, poster, joint) - prompts, suppressors, "
                  "thresholds in kb/prompts/sam3_prompts.yaml, or fine-tuning",
    "wrong_distress_type": "SAM 3.1 inference: e.g. debris clogging detected as spalling - prompt library "
                           "or fine-tuning",
    "wrong_mask_extent": "SAM 3.1 inference: outline too big or too small (whole member painted as damage) - "
                         "prompts, tiling, fine-tuning",
    "wrong_member": "SAM 3.1 component prompts or vision/postprocess.py host and conflict rules "
                    "(e.g. pier cap called crash barrier)",
    "wrong_measurement": "condition skill (vision/condition_proposals.py): masks right but the value wrong "
                         "(partial view, perspective, formula)",
    "wrong_state_for_value": "KB band in kb/sp35_2024 (engineer sign-off, ruling E7) or the engine "
                             "sp35/condition.py (golden tests) - no model chooses the state",
    "needs_site_value": "not an error: the value has to be measured on site",
}
REVIEW_FIELDS = ("review_verdict", "error_stage", "correct_distress", "correct_member", "correct_value",
                 "correct_state", "review_notes")

_IF_WRONG = {
    "detection": "wrong type or not damage -> SAM 3.1 inference (error_stage wrong_distress_type / not_damage)",
    "member": "wrong member -> SAM 3.1 components / post-processing (error_stage wrong_member)",
    "measurement": "outline wrong -> SAM 3.1 (wrong_mask_extent); outline right but value wrong -> skill "
                   "(wrong_measurement)",
    "kb_band": "value right but state wrong for it -> KB band or engine (wrong_state_for_value)",
}
_STAGE_NAMES = {"detection": "SAM", "member": "Member", "measurement": "Extent", "kb_band": "KB", "result": "Result"}


def _step(n: int, stage: str, owner: str, finding: str, **data) -> dict:
    out = {"step": n, "stage": stage, "owner": owner, "finding": finding}
    if stage in _IF_WRONG:
        out["if_wrong"] = _IF_WRONG[stage]
    if data:
        out["data"] = data
    return out


def _reasoning(p: dict, dets: List[dict], result: dict, lib, row, facts: dict, res=None) -> List[dict]:
    """Why this proposal says what it says: one step per stage that could be wrong."""
    cls, host = p["distress_class"], p["host"]
    prompts = list(dict.fromkeys(q for d in dets for q in d.get("prompts", [])))
    try:
        thr = lib.by_id(cls).threshold
    except StopIteration:            # class no longer in the prompt library
        thr = None
    area = sum(int(d.get("area_px", 0)) for d in dets)
    tiles = sum(d.get("frame", "photo") == "tile" for d in dets)
    found = (f"SAM 3.1 found {cls} ({p['sp35_row']}) with {', '.join(repr(q) for q in prompts) or 'no prompt recorded'}; "
             f"best score {p['max_score']:.2f}" + (f" (class threshold {thr:.2f})" if thr is not None else "")
             + f"; {len(dets)} mask(s), {area:,} px" + (f"; {tiles} from the tiled pass" if tiles else ""))
    steps = [_step(1, "detection", "SAM 3.1 inference", found, prompts=prompts, scores=[d["score"] for d in dets],
                   threshold=thr, area_px=area,
                   prompt_library=(result.get("prompt_library") or {}).get("version"))]

    src = host["source"] or ""
    if src == "detected":
        scores = [c["score"] for c in result.get("components", []) if c["class"] == host["class"] and "score" in c]
        inside = [d["host"].get("inside_fraction") for d in dets]
        finding = f"on {host['class']} ({host['sp35']}), outlined by SAM"
        if scores:
            finding += f" (member score {max(scores):.2f})"
        if any(x is not None for x in inside):
            finding += "; damage inside the member: " + ", ".join(f"{x:.0%}" for x in inside if x is not None)
        steps.append(_step(2, "member", "SAM 3.1 + post-processing", finding, member_scores=scores,
                           inside_fraction=inside))
    elif src == "orphan_rescue":
        steps.append(_step(2, "member", "SAM 3.1 + post-processing",
                           "no member found under the damage and the photo has no tag; kept by orphan rescue "
                           "(score >= orphan_rescue_min) instead of being discarded"))
    else:
        steps.append(_step(2, "member", "photo tag (tags.csv)",
                           f"SAM did not outline the member; the photo's tag says {host['class']} ({host['sp35']})"))
    note = next((d["host"].get("note") for d in dets if d["host"].get("note")), None)
    if note:
        steps[-1]["finding"] += f"; {note}"

    how = facts.get("how")
    if how == "union of masks":
        finding = (f"extent = damage on member / member = {facts['damage_on_member_px']:,} / {facts['member_px']:,} px"
                   f" = {facts['pct']:.2f} % (union of {len(dets)} damage mask(s) over {facts['member_masks']} member "
                   f"mask(s) touched, of {facts.get('member_masks_in_photo', facts['member_masks'])} in the photo; largest single "
                   f"mask {facts['largest_single_mask_pct']:.1f} % of its member)")
    elif how == "sum of per-mask coverage":
        finding = (f"extent = sum of per-mask coverage {facts['per_mask_pct']} = {facts['pct']:.2f} % "
                   "(masks unavailable; overlaps may be counted twice)")
    elif how == "photo frame as member":
        px = (f"{facts['damage_px']:,} / {facts['frame_px']:,} px = " if facts.get("frame_px") else "")
        finding = (f"extent = damage / whole photo = {px}{facts['pct']:.2f} % (no outline of the bridge part, "
                   "so the frame is taken as the part)")
    elif "count" in facts:
        finding = f"{facts['count']} counted in this photo (lower bound for the member)"
    else:
        finding = p.get("reason", "not measured")
    if how and "count" in facts:
        finding += f"; count {facts['count']}"
    if how and p["status"] == "needs_mask_check":
        finding += f" -> {p['reason']}"
    steps.append(_step(3, "measurement", "condition skill", finding, **facts))

    if res is not None:
        parts = []
        for m in row.measures():
            v = p["value"].get(m.unit) if isinstance(p["value"], dict) else p["value"]
            if v is None:
                continue
            band = next(s for s, b in m.bands.items() if b is not None and float(v) in b)
            parts.append(f"{v:g} {m.unit} in {m.bands[band].text} -> {band}")
        kb_file = "distress_bridge.yaml" if p["table"] == "7.2" else "distress_approach.yaml"
        steps.append(_step(4, "kb_band", f"KB kb/sp35_2024/{kb_file} (draft until signed)",
                           f"Table {p['table']} {p.get('rated_row', p['sp35_row'])} {p['row_name']}: " + "; ".join(parts)
                           + f" => {res.state} ({p['state_name']}); the book prints {res.state} as "
                           f"'{p['printed'][res.state]}' (gaps between printed ranges closed by ruling E7)",
                           bands={m.unit: {s: (b.text if b else None) for s, b in m.bands.items()}
                                  for m in row.measures()}, printed=p["printed"]))
    else:
        steps.append(_step(4, "kb_band", "KB", "not reached: no value to look up"))
    outcome = (f"proposed {p['proposed_state']} ({p['state_name']}); the final state is the engineer's"
               if p["proposed_state"] else p.get("reason", ""))
    if p.get("flags"):
        outcome += "; flags: " + "; ".join(p["flags"])
    steps.append(_step(5, "result", "condition skill", f"{p['status']}: {outcome}"))
    return steps


def _summary_text(steps: List[dict]) -> str:
    return " | ".join(f"{s['step']} {_STAGE_NAMES[s['stage']]}: {s['finding']}" for s in steps)


MODES = ("max", "strict")
_BASIS = {"union of masks": "bridge-part outline", "sum of per-mask coverage": "bridge-part outline",
          "photo frame as member": "whole photo"}


def propose_photo(result: dict, kb: KnowledgeBase, frame_as_member: bool = False, lib=None,
                  mode: str = "max") -> dict:
    """Proposals for one inference JSON (already parsed), each with its reasoning trace.

    mode "max" (default): give a state to every detection the KB can rate, whatever SAM
    knows about the bridge part; what makes it weaker is listed in `flags`.
    mode "strict": only states measured against an outlined bridge part."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    lib = lib or load_prompt_library(PROMPTS)
    whole_member = lib.rules.redetect_iou
    out = {"photo": result.get("photo"), "inference_status": result.get("status"),
           "prompt_library": result.get("prompt_library"), "proposals": []}
    if result.get("status") not in SCORED_STATUSES:
        return out
    distresses = list(result.get("distresses", [])) + list((result.get("tiled") or {}).get("distresses", []))
    for (row_id, cls, host_class, host_sp35, source), dets in sorted(_group(distresses).items(),
                                                                     key=lambda kv: tuple(map(str, kv[0]))):
        p = {"distress_class": cls, "sp35_row": row_id, "host": {"class": host_class, "sp35": host_sp35,
             "source": source}, "detections": [{"index": d["_index"], "score": d["score"], "bbox": d["bbox"],
             "frame": d.get("frame", "photo")} for d in dets],
             "max_score": max(d["score"] for d in dets),
             "proposed_state": None, "final_state": None, "requires_engineer": True}
        notes = [d["review_note"] for d in dets if d.get("review_note")]
        review = {f: None for f in REVIEW_FIELDS}     # filled in by the engineer
        try:
            row = kb.row(row_id)
        except KeyError as e:
            p.update(status="unknown_row", reason=str(e), reasoning=[], reasoning_text=str(e), review=review)
            out["proposals"].append(p)
            continue
        p.update(row_name=row.name, table=kb.table_of(row), units=[m.unit for m in row.measures()],
                 printed=row.printed)
        host_note = next((d["host"].get("note") for d in dets if d["host"].get("note")), None)
        facts, res, flags, rate_row = {}, None, [], row
        if source == "orphan_rescue" and mode == "strict":
            p.update(status="needs_host", reason="SAM found no host component: engineer assigns the member first")
        elif host_note and mode == "strict":
            p.update(status="needs_table_7_3_row", reason=host_note)
        else:
            if source == "orphan_rescue":
                flags.append("no bridge part found under the damage: part to be assigned later")
            if host_note:                       # wearing course / approach: SP:35 rates these with Table 7.3
                twin = _table_7_3_twin(kb, row)
                if twin is not None and twin is not row:
                    rate_row = twin
                    flags.append(f"wearing course / approach: rated with Table 7.3 {twin.id} ({twin.name})")
                elif twin is None:
                    flags.append("wearing course / approach: no Table 7.3 row of this name, rated with the Table 7.2 row")
            if rate_row is not row:
                p.update(rated_row=rate_row.id, row_name=rate_row.name, table=kb.table_of(rate_row),
                         units=[m.unit for m in rate_row.measures()], printed=rate_row.printed)
            value, method, facts = _measure(rate_row, dets, source or "", frame_as_member, whole_member, result, mode)
            flags += facts.get("flags", [])
            if value is None:
                status = ("needs_site_measurement" if "measure at site" in method else
                          "needs_mask_check" if "engineer checks the mask" in method else "needs_member_extent")
                p.update(status=status, reason=method)
            else:
                res = condition_state(kb, rate_row.id, value)
                state = next(s for s in kb.states.states if s.id == res.state)
                p.update(status="proposed_visual_estimate", value=value, method=method,
                         proposed_state=res.state, state_name=state.name,
                         severity=state.summary, explanation=res.explanation)
        p["basis"] = _BASIS.get(facts.get("how"), "count" if "count" in facts else None) if res else None
        p["flags"] = flags
        if notes:
            p["review_notes"] = notes
        p["reasoning"] = _reasoning(p, dets, result, lib, rate_row, facts, res)
        p["reasoning_text"] = _summary_text(p["reasoning"])
        p["review"] = review
        out["proposals"].append(p)
    return out


def _inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("*.json")):
        if f.name not in ("run_summary.json", "summary.json") and not f.name.endswith(".conditions.json"):
            yield f


def run(input_path: Path, outdir: Path, frame_as_member: bool = False, kb: Optional[KnowledgeBase] = None,
        mode: str = "max") -> dict:
    kb = kb or load_kb()
    lib = load_prompt_library(PROMPTS)
    outdir.mkdir(parents=True, exist_ok=True)
    header = {"kb_version_hash": kb.version_hash, "kb_status": kb.status, "kb_signed": kb.is_signed,
              "frame_as_member": frame_as_member, "mode": mode,
              "note": "Proposed states from photos (SP:35 §7.4: extent and severity are measured at site). "
                      "Not for scoring until an engineer confirms each one."}
    rows, statuses, states, photos = [], Counter(), Counter(), 0
    bases, photo_groups = Counter(), Counter()
    for f in _inputs(input_path):
        result = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or "photo" not in result:
            continue
        photos += 1
        proposal = dict(header, **propose_photo(result, kb, frame_as_member, lib, mode))
        (outdir / f"{Path(result['photo']).stem}.conditions.json").write_text(json.dumps(proposal, indent=1),
                                                                              encoding="utf-8")
        props = proposal["proposals"]
        photo_groups["no distress found by SAM" if not props else
                     "at least one state" if any(p["proposed_state"] for p in props) else
                     "distress found, no state"] += 1
        for p in props:
            statuses[p["status"]] += 1
            if p["proposed_state"]:
                states[p["proposed_state"]] += 1
                bases[p["basis"]] += 1
            rows.append({"photo": proposal["photo"], "distress": p["distress_class"], "sp35_row": p["sp35_row"],
                         "rated_row": p.get("rated_row", p["sp35_row"]),
                         "host": p["host"]["class"], "host_sp35": p["host"]["sp35"], "host_source": p["host"]["source"],
                         "detections": len(p["detections"]), "max_score": p["max_score"],
                         "value": json.dumps(p.get("value")) if "value" in p else "",
                         "proposed_state": p["proposed_state"] or "", "status": p["status"],
                         "basis": p.get("basis") or "", "flags": "; ".join(p.get("flags", [])),
                         "reason_or_method": p.get("method") or p.get("reason", ""),
                         "reasoning": p["reasoning_text"], **{f: "" for f in REVIEW_FIELDS}})
    if photos == 0:
        raise SystemExit(f"no inference JSON found at {input_path}")
    fields = ["photo", "distress", "sp35_row", "rated_row", "host", "host_sp35", "host_source", "detections",
              "max_score", "value", "proposed_state", "status", "basis", "flags", "reason_or_method", "reasoning",
              *REVIEW_FIELDS]
    with open(outdir / "conditions.csv", "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel reads it as UTF-8
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(header, photos=photos, proposals=len(rows), by_status=dict(statuses),
                   proposed_states={s: states[s] for s in ("I", "II", "III", "IV", "V")},
                   states_by_basis=dict(bases), photos_by_outcome=dict(photo_groups),
                   review_columns=list(REVIEW_FIELDS), error_stages=ERROR_STAGES)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def tally(reviewed_csv: Path) -> dict:
    """Read a conditions.csv after manual review and say where the errors are:
    each `error_stage` maps to the part of the pipeline that has to change."""
    with open(reviewed_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    reviewed = [r for r in rows if (r.get("error_stage") or "").strip()]
    stages = Counter(r["error_stage"].strip() for r in reviewed)
    unknown = sorted(set(stages) - set(ERROR_STAGES))
    relabels = Counter((r["distress"], r["correct_distress"].strip()) for r in reviewed
                       if r["error_stage"].strip() == "wrong_distress_type" and (r.get("correct_distress") or "").strip())
    members = Counter((r["host"], r["correct_member"].strip()) for r in reviewed
                      if r["error_stage"].strip() == "wrong_member" and (r.get("correct_member") or "").strip())
    return {"rows": len(rows), "reviewed": len(reviewed), "unknown_error_stages": unknown,
            "by_error_stage": {k: {"count": n, "fix_in": ERROR_STAGES.get(k, "unknown code")}
                               for k, n in stages.most_common()},
            "distress_relabels": [{"sam_said": a, "engineer_said": b, "count": n} for (a, b), n in relabels.most_common()],
            "member_relabels": [{"sam_said": a, "engineer_said": b, "count": n} for (a, b), n in members.most_common()]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose SP:35 condition states (I-V) from SAM 3.1 inference JSON")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="one <photo>.json or a folder from vision.infer")
    src.add_argument("--tally", type=Path, help="a reviewed conditions.csv: count errors by stage and where to fix them")
    ap.add_argument("--outdir", type=Path, help="default: <input folder>/conditions")
    ap.add_argument("--strict", action="store_true",
                    help="only states measured against an outlined bridge part (default: max, rate every "
                         "detection the KB can rate and flag the weaker ones)")
    ap.add_argument("--frame-as-member", action="store_true",
                    help="strict mode only: for photo-tag hosts, treat the frame as the member (close-ups)")
    args = ap.parse_args(argv)
    if args.tally:
        result = tally(args.tally)
        (args.tally.parent / "review_tally.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
        return 0
    base = args.input.parent if args.input.is_file() else args.input
    summary = run(args.input, args.outdir or base / "conditions", args.frame_as_member,
                  mode="strict" if args.strict else "max")
    print(json.dumps({k: summary[k] for k in ("mode", "photos", "photos_by_outcome", "proposals", "by_status",
                                               "proposed_states", "states_by_basis", "kb_signed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
