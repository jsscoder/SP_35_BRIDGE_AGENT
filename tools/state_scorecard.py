"""Condition-state scorecard: predicted states vs the bridge engineer's labels.

    uv run python -m tools.state_scorecard --labels data/labels/yugmi_engineer_annotations_2026-10-13.json \
        --predictions data/pilot/infer_all_v1

For every usable engineer label (polygon + member + distress row + state) on a
photo that was inferred, it finds the SAM 3.1 distress mask that covers the
polygon best and takes the state the condition skill proposed for it (none =
abstained). It reports exact and within-one accuracy, abstentions, under-calls
(predicted better than the engineer: the dangerous direction), the confusion
matrix and quadratic weighted kappa, against the majority-class baseline.

It also runs an ORACLE diagnostic: the measurement + KB route applied to the
engineer's own polygon and distress row. That separates the two error sources —
if the oracle is right and the pipeline wrong, fix detection (SAM 3.1); if the
oracle is wrong too, the measurement or the KB route disagrees with the engineer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def pipeline_state(label: Label, result: dict, cond: Optional[dict], min_overlap: float) -> dict:
    """The skill's state for the SAM distress that covers the engineer's polygon best."""
    gt = polygon_mask(label, result)
    if not gt.any():
        return {"state": None, "why": "polygon outside the processed frame"}
    best, idx = 0.0, None
    for i, det in enumerate(result.get("distresses", [])):
        share = float((decode_rle(det["rle"]) & gt).sum()) / float(gt.sum())
        if share > best:
            best, idx = share, i
    if idx is None or best < min_overlap:
        return {"state": None, "why": "SAM found no distress on the polygon", "overlap": round(best, 3)}
    det = result["distresses"][idx]
    prop = next((p for p in (cond or {}).get("proposals", [])
                 if any(x["index"] == idx and x.get("frame", "photo") == "photo" for x in p["detections"])), None)
    out = {"state": prop["proposed_state"] if prop else None, "sam_class": det["class"], "sam_row": det["sp35"],
           "overlap": round(best, 3), "status": prop["status"] if prop else "no proposal", "sam_det_idx": idx}
    out["why"] = f"SAM {det['class']} covers {best:.0%} of the polygon; skill: {out['status']}"
    return out


def oracle_state(label: Label, result: dict, kb: KnowledgeBase, same_member_count: int) -> dict:
    """Measurement + KB route on the engineer's own polygon and row (perfect detection)."""
    row = kb.row(label.row_id)
    units = [m.unit for m in row.measures()]
    values, notes = {}, []
    for unit in units:
        if unit == "No.":
            values[unit] = same_member_count
            notes.append(f"{same_member_count} labelled on this member in this photo")
        elif unit == "%":
            gt = polygon_mask(label, result)
            members = [decode_rle(c["rle"]) for c in result.get("components", []) if c.get("sp35") == label.sub_code]
            touched = [m for m in members if (m & gt).any()]
            if not touched:
                notes.append(f"SAM has no mask of {label.sub_code} under the polygon: no member area")
                continue
            member = np.logical_or.reduce(touched)
            values[unit] = round(100.0 * float((gt & member).sum()) / float(member.sum()), 2)
            notes.append(f"polygon covers {values[unit]} % of the {label.sub_code} mask")
        else:
            notes.append(f"{row.name} is in {unit}: site measurement")
    if not values:
        return {"state": None, "why": "; ".join(notes)}
    value = values[units[0]] if len(units) == 1 else values
    if len(units) > 1 and len(values) < len(units):
        notes.append("one of two measures only")
    res = condition_state(kb, row.id, value)
    return {"state": res.state, "value": value, "why": "; ".join(notes)}


def metrics(pairs: List[Tuple[str, Optional[str]]]) -> dict:
    n = len(pairs)
    answered = [(g, p) for g, p in pairs if p]
    exact = sum(g == p for g, p in answered)
    within = sum(abs(RANK[g] - RANK[p]) <= 1 for g, p in answered)
    under = sum(RANK[p] < RANK[g] for g, p in answered)
    over = sum(RANK[p] > RANK[g] for g, p in answered)
    conf = {g: {**{p: 0 for p in STATES}, "none": 0} for g in STATES}
    for g, p in pairs:
        conf[g][p or "none"] += 1
    return {"labels": n, "answered": len(answered), "abstained": n - len(answered),
            "exact_all": round(exact / n, 3) if n else None,
            "exact_answered": round(exact / len(answered), 3) if answered else None,
            "within_one_answered": round(within / len(answered), 3) if answered else None,
            "under_calls": under, "over_calls": over, "qwk_answered": qwk(answered), "confusion": conf}


def quadratic_weighted_kappa(data) -> Optional[float]:
    """Computes Quadratic Weighted Kappa from either a list of (gt, pred) pairs or a KxK confusion matrix."""
    if isinstance(data, np.ndarray) and data.ndim == 2:
        k = data.shape[0]
        o = data.astype(float)
        if o.sum() < 2:
            return 0.0
    else:
        pairs = [(g, p) for g, p in data if p is not None]
        if len(pairs) < 2:
            return None
        k = len(STATES)
        o = np.zeros((k, k), dtype=float)
        for g, p in pairs:
            if g in RANK and p in RANK:
                o[RANK[g], RANK[p]] += 1
    total = o.sum()
    if total <= 0:
        return 0.0 if isinstance(data, np.ndarray) else None
    e = np.outer(o.sum(1), o.sum(0)) / total
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (w * e).sum()
    return round(1.0 - (w * o).sum() / den, 3) if den else None


qwk = quadratic_weighted_kappa


def score(labels_path: Path, predictions: Path, conditions: Optional[Path] = None, min_overlap: float = 0.1,
          kb: Optional[KnowledgeBase] = None, calibrator: Optional[Path] = None) -> dict:
    kb = kb or load_kb()
    conditions = conditions or predictions / "conditions"
    labels = load_labels(labels_path, kb)
    cal_model = None
    if calibrator:
        from vision.condition_calibrator import SP35ConditionCalibrator
        cal_path = Path(calibrator)
        if cal_path.exists():
            cal_model = SP35ConditionCalibrator.load(cal_path, kb=kb)
    per_member = Counter((l.photo, l.sub_code, l.support, l.span, l.row_id) for l in labels if l.usable)
    rows, skipped = [], Counter()
    for lab in labels:
        if not lab.usable:
            skipped["; ".join(lab.problems)] += 1
            continue
        stem = Path(lab.photo).stem
        f = predictions / f"{stem}.json"
        result = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        if not result or result.get("status") not in ("ok", "ok_reduced"):
            skipped["photo not in the inference run"] += 1
            continue
        cf = conditions / f"{stem}.conditions.json"
        cond = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None
        pred = pipeline_state(lab, result, cond, min_overlap)
        orc = oracle_state(lab, result, kb, per_member[(lab.photo, lab.sub_code, lab.support, lab.span, lab.row_id)])
        cal_res = None
        if cal_model and pred.get("sam_det_idx") is not None and lab.row_id:
            dets = result.get("distresses", [])
            idx = pred["sam_det_idx"]
            if idx < len(dets):
                best_det = dets[idx]
                target_sp35 = best_det.get("host", {}).get("sp35")
                comp_masks = [decode_rle(c["rle"]) for c in result.get("components", [])
                              if "rle" in c and (c.get("sp35") == target_sp35 or c.get("sp35") == lab.sub_code)]
                host_m = np.logical_or.reduce(comp_masks) if comp_masks else None
                try:
                    pdata = cal_model.predict_proposal([best_det], result, host_m, lab.row_id)
                    cal_res = {"state": pdata["proposed_state"], "why": pdata["explanation"]}
                except Exception as ex:
                    cal_res = {"state": None, "why": f"calibrator error: {ex}"}
        row_entry = {"annotation_id": lab.annotation_id, "photo": lab.photo, "member": lab.sub_code,
                     "member_name": kb.sub_names.get(lab.sub_code), "support": lab.support, "span": lab.span,
                     "anomaly": lab.anomaly, "row": lab.row_id, "engineer_state": lab.state,
                     "pipeline": pred, "oracle": orc}
        if cal_res is not None:
            row_entry["calibrated"] = cal_res
        rows.append(row_entry)
    gts = [r["engineer_state"] for r in rows]
    majority = Counter(gts).most_common(1)[0][0] if gts else None
    report = {
        "labels_file": str(labels_path), "predictions": str(predictions), "min_overlap": min_overlap,
        "kb_signed": kb.is_signed, "scored": len(rows), "skipped": dict(skipped),
        "engineer_states": dict(Counter(gts)),
        "pipeline": metrics([(r["engineer_state"], r["pipeline"]["state"]) for r in rows]),
        "oracle_measurement_route": metrics([(r["engineer_state"], r["oracle"]["state"]) for r in rows]),
        "baseline_majority": dict(metrics([(g, majority) for g in gts]), state=majority,
                                  note="always predicting the most common engineer state (chosen on this same set)"),
        "pipeline_miss_reasons": dict(Counter(r["pipeline"]["why"].split(";")[0].split(" covers")[0]
                                              for r in rows if not r["pipeline"]["state"])),
        "oracle_abstain_reasons": dict(Counter(r["oracle"]["why"].split(":")[0] for r in rows
                                               if not r["oracle"]["state"])),
        "rows": rows,
    }
    if cal_model:
        report["calibrated"] = metrics([(r["engineer_state"], (r.get("calibrated") or {}).get("state")) for r in rows])
    return report


def write_markdown(report: dict, path: Path) -> None:
    def line(name, m):
        return (f"| {name} | {m['answered']}/{m['labels']} | {m['exact_all']} | {m['exact_answered']} | "
                f"{m['within_one_answered']} | {m['under_calls']} | {m['over_calls']} | {m['qwk_answered']} |")
    out = [f"# Condition-state scorecard", "",
           f"Labels: `{report['labels_file']}` · predictions: `{report['predictions']}` · "
           f"scored {report['scored']} · KB signed: {report['kb_signed']}", "",
           f"Engineer states: {report['engineer_states']}", "",
           "| Method | Answered | Exact (all labels) | Exact (answered) | Within one | Under-calls | Over-calls | QWK |",
           "|---|---|---|---|---|---|---|---|",
           line("Pipeline (SAM 3.1 + skill)", report["pipeline"])]
    if "calibrated" in report:
        out.append(line("Calibrated (SP35ConditionCalibrator)", report["calibrated"]))
    out.extend([
           line("Oracle: measurement + KB on the engineer's polygon", report["oracle_measurement_route"]),
           line(f"Baseline: always {report['baseline_majority']['state']}", report["baseline_majority"]), "",
           f"Pipeline abstained because: {report['pipeline_miss_reasons']}", "",
           f"Oracle abstained because: {report['oracle_abstain_reasons']}", "",
           f"Skipped labels: {report['skipped']}", "",
           "| Photo | Member | Anomaly → row | Engineer | Pipeline | Oracle | Why (pipeline / oracle) |", "|---|---|---|---|---|---|---|"])
    for r in report["rows"]:
        where = r["support"] or (f"span {r['span']}" if r["span"] else "")
        out.append(f"| {r['photo']} | {r['member']} {where} | {r['anomaly']} → {r['row']} | {r['engineer_state']} | "
                   f"{r['pipeline']['state'] or '—'} | {r['oracle']['state'] or '—'} | "
                   f"{r['pipeline']['why']} / {r['oracle']['why']} |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="vision.infer output folder")
    ap.add_argument("--conditions", type=Path, help="default: <predictions>/conditions")
    ap.add_argument("--calibrator", type=Path, help="optional path to calibrator.pkl")
    ap.add_argument("--outdir", type=Path, help="default: <predictions>/scorecard")
    ap.add_argument("--min-overlap", type=float, default=0.1, help="share of the polygon a SAM mask must cover")
    args = ap.parse_args(argv)
    report = score(args.labels, args.predictions, args.conditions, args.min_overlap, calibrator=args.calibrator)
    outdir = args.outdir or args.predictions / "scorecard"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "state_scorecard.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, outdir / "state_scorecard.md")
    print(json.dumps({k: report[k] for k in ("scored", "skipped", "engineer_states")}, indent=1))
    methods_to_print = ["pipeline"]
    if "calibrated" in report:
        methods_to_print.append("calibrated")
    methods_to_print.extend(["oracle_measurement_route", "baseline_majority"])
    for k in methods_to_print:
        m = report[k]
        print(f"{k:26s} answered {m['answered']}/{m['labels']}  exact(all) {m['exact_all']}  "
              f"exact(answered) {m['exact_answered']}  under {m['under_calls']}  over {m['over_calls']}  qwk {m['qwk_answered']}")
    print("pipeline abstained:", report["pipeline_miss_reasons"])
    print("oracle abstained:", report["oracle_abstain_reasons"])
    print(f"report: {outdir / 'state_scorecard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
