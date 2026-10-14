import json
from pathlib import Path
from collections import Counter

def diagnose_undetected():
    gt_path = Path(r"c:\Users\omkar\Downloads\annotations.json")
    if not gt_path.exists():
        gt_path = Path("data/labels/yugmi_engineer_annotations_2026-10-13.json")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_items = json.load(f)["data"]

    infer_dir = Path("data/pilot/infer_all_v1")
    infer_jsons = {p.stem: p for p in infer_dir.glob("*.json") 
                   if p.name not in ("run_summary.json", "condition_proposals.json", "calibration_scorecard_report.json")}

    print(f"Total inference JSONs in {infer_dir}: {len(infer_jsons)}")

    undetected = []
    reasons_summary = Counter()

    for idx, gt in enumerate(gt_items):
        cond = gt.get("condition")
        p_str = gt.get("imagePayload")
        if not p_str or not cond:
            continue
        try:
            p_json = json.loads(p_str)
            img_name = p_json.get("name")
        except Exception:
            continue

        stem = Path(img_name).stem
        anom = gt.get("anomalyName", "Unknown")
        asset = gt.get("assetName", "Unknown")

        in_infer_run = stem in infer_jsons
        dets_count = 0
        rej_count = 0
        rejs = []

        if in_infer_run:
            try:
                with open(infer_jsons[stem], "r", encoding="utf-8") as jf:
                    inf = json.load(jf)
                    dets = inf.get("distresses", [])
                    rejs = inf.get("rejected", [])
                    dets_count = len(dets)
                    rej_count = len(rejs)
            except Exception:
                pass

        if dets_count == 0:
            if not in_infer_run:
                cat = "PHOTO_NOT_IN_INFERENCE_DATASET"
                detail = "Photo file was not provided in input pilot directory"
            elif rej_count > 0:
                cat = "REJECTED_BY_VISION_FILTER"
                detail = f"{rej_count} candidate(s) rejected: {[(r.get('class'), r.get('reject_reason'), round(r.get('score', 0), 2)) for r in rejs]}"
            else:
                cat = "ZERO_PREDICTIONS_FROM_SAM"
                detail = "SAM 3.1 prompt yielded no candidates above raw threshold"

            reasons_summary[cat] += 1
            undetected.append({
                "photo": img_name,
                "stem": stem,
                "condition": cond,
                "anomaly": anom,
                "asset": asset,
                "category": cat,
                "detail": detail
            })

    print(f"\nTotal Undetected Annotations: {len(undetected)} / 48")
    print("\n--- SUMMARY OF ROOT CAUSES ---")
    for cat, count in reasons_summary.items():
        print(f" * {cat}: {count} annotations")

    print("\n--- DETAILED ANNOTATION BREAKDOWN ---")
    for u in undetected:
        print(f"[{u['category']}] {u['photo']} (GT: State {u['condition']})")
        print(f"    Anomaly: '{u['anomaly']}' on Asset: '{u['asset']}'")
        print(f"    Detail:  {u['detail']}\n")

if __name__ == "__main__":
    diagnose_undetected()

