import json
import csv
from pathlib import Path

def analyze_gt_matching():
    # Load GT from Downloads or fallback to local
    # local copy first (data/ is gitignored); the Downloads file is only a fallback
    gt_path = Path("data/labels/yugmi_engineer_annotations_2026-10-13.json")
    if not gt_path.exists():
        gt_path = Path(r"c:\Users\omkar\Downloads\annotations.json")
    
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_items = json.load(f)["data"]

    # Load Calibrated proposals from CSV
    csv_path = Path("data/pilot/calibrated_v1/conditions_comparison_audit.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    csv_by_photo = {}
    for r in csv_rows:
        p = r["photo"]
        csv_by_photo.setdefault(p, []).append(r)

    results = []
    for gt in gt_items:
        cond = gt.get("condition")
        p_str = gt.get("imagePayload")
        if not p_str or not cond:
            continue
        try:
            p_json = json.loads(p_str)
            img_name = p_json.get("name")
        except Exception:
            continue

        csv_data = csv_by_photo.get(img_name, [])
        calib_state = csv_data[0]["calibrated_proposed_state"] if csv_data else None
        base_state = csv_data[0]["baseline_proposed_state"] if csv_data else None
        base_status = csv_data[0]["baseline_status"] if csv_data else "NO_DETECTION"

        results.append({
            "photo": img_name,
            "gt_condition": cond,
            "gt_anomaly": gt.get("anomalyName", "Unknown"),
            "gt_asset": gt.get("assetName", "Unknown"),
            "calib_state": calib_state,
            "base_state": base_state,
            "base_status": base_status,
            "detected": bool(csv_data)
        })

    print(f"Total Ground Truth items evaluated: {len(results)}")
    detected = [r for r in results if r["detected"]]
    print(f"Photos detected by AI: {len(detected)} / {len(results)}")

    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    exact = sum(1 for r in detected if r["calib_state"] == r["gt_condition"])
    within_1 = sum(1 for r in detected if abs(roman_map.get(r["calib_state"], 0) - roman_map.get(r["gt_condition"], 0)) <= 1)
    base_exact = sum(1 for r in detected if r["base_state"] == r["gt_condition"] and r["base_state"] != "NONE")
    base_scored = sum(1 for r in detected if r["base_state"] and r["base_state"] != "NONE")

    print("\n--- RESULTS ---")
    print(f"Baseline Scored: {base_scored} / {len(detected)}")
    print(f"Baseline Exact Matches: {base_exact} / {len(detected)} ({base_exact/len(detected)*100:.1f}%)")
    print(f"Calibrated Scored: {len(detected)} / {len(detected)} (100.0%)")
    print(f"Calibrated Exact Matches: {exact} / {len(detected)} ({exact/len(detected)*100:.1f}%)")
    print(f"Calibrated Within-1 State: {within_1} / {len(detected)} ({within_1/len(detected)*100:.1f}%)")
    from collections import Counter
    majority, m_n = Counter(r["gt_condition"] for r in detected).most_common(1)[0]
    print(f"Always-'{majority}' baseline (same labels): {m_n} / {len(detected)} ({m_n/len(detected)*100:.1f}%)")
    print("NOTE: photo-level comparison. Each label is compared with the FIRST calibrated proposal in its photo,")
    print("      whatever defect that proposal is about. For defect-level (polygon-overlap) accuracy use")
    print("      tools/state_scorecard.py.")

    # Confusion matrix
    conf_matrix = {}
    for r in detected:
        gt = r["gt_condition"]
        pred = r["calib_state"]
        conf_matrix.setdefault(gt, {})
        conf_matrix[gt][pred] = conf_matrix[gt].get(pred, 0) + 1

    print("\n--- CONFUSION MATRIX (GT vs Calibrated) ---")
    states = ["I", "II", "III", "IV", "V"]
    header = "GT \\ Pred\t" + "\t".join(states)
    print(header)
    for s_gt in states:
        row_str = f"State {s_gt}\t"
        for s_pr in states:
            cnt = conf_matrix.get(s_gt, {}).get(s_pr, 0)
            row_str += f"{cnt}\t"
        print(row_str)

    # Per-anomaly match rates
    anom_stats = {}
    for r in detected:
        a = r["gt_anomaly"]
        anom_stats.setdefault(a, {"total": 0, "exact": 0, "within_1": 0})
        anom_stats[a]["total"] += 1
        if r["calib_state"] == r["gt_condition"]:
            anom_stats[a]["exact"] += 1
        if abs(roman_map.get(r["calib_state"], 0) - roman_map.get(r["gt_condition"], 0)) <= 1:
            anom_stats[a]["within_1"] += 1

    print("\n--- ANOMALY MATCHING ---")
    for a, s in sorted(anom_stats.items(), key=lambda x: -x[1]["total"]):
        print(f"{a}: Total={s['total']} | Exact={s['exact']} ({s['exact']/s['total']*100:.0f}%) | Within-1={s['within_1']} ({s['within_1']/s['total']*100:.0f}%)")

if __name__ == "__main__":
    analyze_gt_matching()

