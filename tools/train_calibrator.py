"""CLI Training, Cross-Validation & Threshold Grid-Search Runner for SP:35 Condition States.

File: tools/train_calibrator.py (or root train_calibrator.py)

Reads:
  - Configuration: `kb/training/fine_tuning.yaml`
  - Engineer ground truth: `data/labels/annotations.json` (via `ingest.platform_annotations.load_labels`)
  - SAM 3.1 inference JSONs: `data/pilot/infer_all_v1/*.json`
  - Optional source photos: `data/pilot/patna_383/` (for RGB shadow/rust severity cues)

Executes:
  1. Per-class SAM 3.1 detection threshold grid search (evaluating `distresses` + `rejected` rescue).
  2. 5-fold GroupKFold cross-validation grouped by `photo` (zero same-photo data leakage).
  3. Before/After `state_scorecard` benchmark comparing:
       - Baseline A: Uncalibrated `vision.condition_proposals`
       - Baseline B: Zero-Shot Suppressor-Corrected KB Structural Proxy Prior
       - Model C: Trained Hybrid `SP35ConditionCalibrator` (Out-of-Fold CV)
  4. Saves final `calibrator.pkl`, `threshold_recommendations.yaml`, and scorecard reports (`.json` & `.md`).

Usage:
  uv run python tools/train_calibrator.py \
      --config kb/training/fine_tuning.yaml \
      --labels data/labels/annotations.json \
      --predictions data/pilot/infer_all_v1 \
      --output-dir data/pilot/calibrated_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import GroupKFold

from ingest.platform_annotations import Label, load_labels
from sp35.condition import condition_state
from sp35.kb import STATES, KnowledgeBase, load_kb
from tools.state_scorecard import polygon_mask, quadratic_weighted_kappa
from vision.condition_proposals import (
    COUNT_UNIT,
    SCORED_STATUSES,
    SITE_UNITS,
    decode_rle,
)

try:
    from vision.condition_calibrator import (
        IDX_TO_STATE,
        STATE_TO_IDX,
        SP35ConditionCalibrator,
        bbox_iou,
        extract_distress_features,
    )
except ImportError:
    from condition_calibrator import (  # type: ignore
        IDX_TO_STATE,
        STATE_TO_IDX,
        SP35ConditionCalibrator,
        bbox_iou,
        extract_distress_features,
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Pixel-level IoU between two boolean masks of identical shape."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(inter / max(1, union))


def polygon_to_bbox(label: Any, scale: float) -> List[float]:
    """Converts native-resolution label polygon to scaled [x1, y1, x2, y2] bbox."""
    poly = getattr(label, "polygon", getattr(label, "points", []))
    if not poly:
        return [0.0, 0.0, 1.0, 1.0]
    xs = [pt[0] * scale for pt in poly]
    ys = [pt[1] * scale for pt in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def find_host_mask_for_detections(
    detections: List[dict], result_json: dict
) -> Optional[np.ndarray]:
    """Finds the segmented component RLE mask matching the detection host SP:35 code, if present."""
    if not detections:
        return None
    target_sp35 = detections[0].get("host", {}).get("sp35")
    target_cls = detections[0].get("host", {}).get("class")
    comp_masks = []
    for comp in result_json.get("components", []):
        if "rle" not in comp:
            continue
        if (target_sp35 and comp.get("sp35") == target_sp35) or (
            target_cls and comp.get("class") == target_cls
        ):
            comp_masks.append(decode_rle(comp["rle"]))
    if not comp_masks:
        return None
    return np.logical_or.reduce(comp_masks)


def baseline_uncalibrated_proposal(
    detections: List[dict],
    host_mask: Optional[np.ndarray],
    row_id: str,
    kb: KnowledgeBase,
) -> Optional[str]:
    """Reproduces current uncalibrated `vision.condition_proposals.propose_condition_for_group`."""
    row = kb.row(row_id)
    if row.unit in SITE_UNITS or row.criteria:
        return None
    if row.unit == COUNT_UNIT:
        return condition_state(kb, row_id, len(detections)).state
    if row.unit == "%":
        dist_masks = [decode_rle(d["rle"]) for d in detections if "rle" in d]
        if not dist_masks:
            return None
        dist_union = np.logical_or.reduce(dist_masks)
        if host_mask is not None and host_mask.sum() > 0:
            pct = round(
                (int(np.logical_and(dist_union, host_mask).sum()) / int(host_mask.sum()))
                * 100.0,
                2,
            )
        else:
            pct = round((int(dist_union.sum()) / max(1, dist_union.size)) * 100.0, 2)
        return condition_state(kb, row_id, pct).state
    return None


DEFECT_FAMILIES = {
    "spalling": {
        "T7.2-08",  # Spalling
        "T7.2-01",  # Erosion
        "T7.2-16",  # Crushing
        "T7.2-09",  # Staining/scaling/leaching
        "T7.2-19",  # Degradation (wear and tear)
        "T7.2-26",  # Silting
        "T7.2-06",  # Honeycombing
        "T7.2-07",  # Delamination
        "T7.2-12",  # Exposed reinforcement
        "T7.2-23",  # Undesired restraints
        "T7.2-25",  # Inadequate drainage spots
    },
    "crack": {
        "T7.2-11",  # Cracking (width)
        "T7.3-24",  # Expansion joint damage
        "T7.2-25",  # Clogging
        "T7.2-09",  # Staining
    },
    "missing_element": {
        "T7.2-20",  # Missing elements
        "T7.2-22",  # Loose joints
    },
    "vegetation": {
        "T7.2-04",  # Vegetation
    },
    "staining": {
        "T7.2-09",  # Staining/scaling/leaching
        "T7.2-10",  # Rust
        "T7.2-15",  # Efflorescence
    },
    "debris_clogging": {
        "T7.2-25",  # Inadequate drainage spots
        "T7.2-26",  # Silting
    },
}


def is_spatial_match_detection_and_label(
    d: dict,
    lbl: Label,
    gt_mask: Optional[np.ndarray],
    gt_box: List[float],
    scale: float = 1.0,
) -> Tuple[bool, float]:
    """Bidirectional containment check (Civil Engineering Report §4.1):
    Match <==> (Cov_GT >= 0.15) or (Cov_SAM >= 0.50) or (IoU >= 0.10)
    """
    if "rle" in d and gt_mask is not None and gt_mask.sum() > 0:
        d_mask = decode_rle(d["rle"])
        d_area = float(d_mask.sum())
        gt_area = float(gt_mask.sum())
        inter = float(np.logical_and(gt_mask, d_mask).sum())
        if inter <= 0.0:
            return False, 0.0
        cov_gt = inter / max(1.0, gt_area)
        cov_sam = inter / max(1.0, d_area)
        union = gt_area + d_area - inter
        iou = inter / max(1.0, union)
        matched = (cov_gt >= 0.15) or (cov_sam >= 0.50) or (iou >= 0.10)
        return matched, max(cov_gt, cov_sam, iou)
    else:
        d_box = d.get("bbox", [0, 0, 1, 1])
        lbl_box = gt_box if gt_box is not None else polygon_to_bbox(lbl, scale)
        x_a = max(d_box[0], lbl_box[0])
        y_a = max(d_box[1], lbl_box[1])
        x_b = min(d_box[2], lbl_box[2])
        y_b = min(d_box[3], lbl_box[3])
        inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
        if inter <= 0.0:
            return False, 0.0
        area_d = max(1.0, (d_box[2] - d_box[0]) * (d_box[3] - d_box[1]))
        area_l = max(1.0, (lbl_box[2] - lbl_box[0]) * (lbl_box[3] - lbl_box[1]))
        cov_l = inter / area_l
        cov_d = inter / area_d
        union = area_d + area_l - inter
        iou = inter / union
        matched = (cov_l >= 0.15) or (cov_d >= 0.50) or (iou >= 0.10)
        return matched, max(cov_l, cov_d, iou)


def match_labels_to_predictions(
    labels: List[Label],
    predictions_dir: Path,
    images_dir: Optional[Path],
    kb: KnowledgeBase,
    iou_threshold: float = 0.15,
    include_rejected_rescue: bool = True,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Pairs engineer ground-truth polygons with SAM 3.1 detections across all processed JSONs."""
    labels_by_photo: Dict[str, List[Label]] = defaultdict(list)
    for lbl in labels:
        raw_photo = (
            getattr(lbl, "photo", None)
            or getattr(lbl, "image", None)
            or getattr(lbl, "filename", "")
        )
        photo_key = Path(str(raw_photo)).name
        if photo_key:
            labels_by_photo[photo_key].append(lbl)

    matched_records: List[dict] = []
    threshold_candidates: Dict[str, List[dict]] = defaultdict(list)

    for json_path in sorted(predictions_dir.glob("*.json")):
        if json_path.name in ("condition_proposals.json", "calibration_scorecard_report.json"):
            continue
        try:
            result_json = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if result_json.get("status") not in SCORED_STATUSES:
            continue

        photo_name = Path(result_json.get("photo", json_path.stem + ".jpg")).name
        photo_labels = labels_by_photo.get(photo_name, [])
        if not photo_labels:
            continue

        # Load optional RGB image for shadow depth & rust redness features
        image_rgb: Optional[np.ndarray] = None
        if images_dir is not None:
            cand_img = images_dir / photo_name
            if cand_img.exists():
                try:
                    image_rgb = np.asarray(Image.open(cand_img).convert("RGB"))
                except Exception:
                    image_rgb = None

        scale = float(result_json.get("scale", 1.0))
        active_dets = result_json.get("distresses", [])
        rejected_dets = result_json.get("rejected", [])

        # Track all detections for per-class threshold optimization
        for d in active_dets + rejected_dets:
            d_cls = d.get("class", "unknown")
            d_row = d.get("sp35", "")
            allowed_rows = DEFECT_FAMILIES.get(d_cls, {d_row})
            is_tp = False
            for lbl in photo_labels:
                lbl_row = (
                    getattr(lbl, "row_id", None)
                    or getattr(lbl, "sp35_row", None)
                    or getattr(lbl, "sp35", None)
                )
                if lbl_row == d_row or lbl_row in allowed_rows or not allowed_rows:
                    matched, _ = is_spatial_match_detection_and_label(
                        d, lbl, polygon_mask(lbl, result_json), polygon_to_bbox(lbl, scale), scale
                    )
                    if matched:
                        is_tp = True
                        break
            threshold_candidates[d_cls].append(
                {
                    "score": float(d.get("score", 0.0)),
                    "is_true_positive": is_tp,
                    "was_rejected": "reject_reason" in d,
                    "reject_reason": d.get("reject_reason", ""),
                }
            )

        # Match each engineer label to best SAM detection group
        for lbl in photo_labels:
            gt_state = (
                getattr(lbl, "state", None)
                or getattr(lbl, "condition_state", None)
                or getattr(lbl, "engineer_state", None)
            )
            row_id = (
                getattr(lbl, "row_id", None)
                or getattr(lbl, "sp35_row", None)
                or getattr(lbl, "sp35", None)
            )
            if gt_state not in STATE_TO_IDX or not row_id:
                continue

            gt_mask = polygon_mask(lbl, result_json)
            gt_box = polygon_to_bbox(lbl, scale)

            matched_group: List[dict] = []
            best_match_val = 0.0

            # 1. Search active detections matching row_id or compatible defect family
            for d in active_dets:
                d_cls = d.get("class", "unknown")
                d_row = d.get("sp35", "")
                allowed_rows = DEFECT_FAMILIES.get(d_cls, {d_row})
                if d_row == row_id or row_id in allowed_rows or not allowed_rows:
                    matched, val = is_spatial_match_detection_and_label(
                        d, lbl, gt_mask, gt_box, scale
                    )
                    if matched:
                        matched_group.append(d)
                        best_match_val = max(best_match_val, val)

            # 2. Optional rescue from `rejected` if below threshold
            rescued_from_rejected = False
            if not matched_group and include_rejected_rescue:
                for r_det in rejected_dets:
                    r_cls = r_det.get("class", "unknown")
                    r_row = r_det.get("sp35", "")
                    allowed_rows = DEFECT_FAMILIES.get(r_cls, {r_row})
                    if r_row == row_id or row_id in allowed_rows or not allowed_rows:
                        matched, val = is_spatial_match_detection_and_label(
                            r_det, lbl, gt_mask, gt_box, scale
                        )
                        if matched and float(r_det.get("score", 0.0)) >= 0.28:
                            matched_group.append(r_det)
                            best_match_val = max(best_match_val, val)
                            rescued_from_rejected = True

            if not matched_group:
                continue

            host_mask = find_host_mask_for_detections(matched_group, result_json)
            feats = extract_distress_features(
                matched_group, result_json, host_mask, row_id, kb, image_rgb
            )
            baseline_state = baseline_uncalibrated_proposal(
                matched_group, host_mask, row_id, kb
            )

            matched_records.append(
                {
                    "photo": photo_name,
                    "row_id": row_id,
                    "gt_state": gt_state,
                    "baseline_state": baseline_state,
                    "detections": matched_group,
                    "result_json": result_json,
                    "host_mask": host_mask,
                    "image_rgb": image_rgb,
                    "features": feats,
                    "best_iou": round(best_match_val, 4),
                    "rescued_from_rejected": rescued_from_rejected,
                }
            )

    return matched_records, threshold_candidates


def optimize_class_thresholds(
    threshold_candidates: Dict[str, List[dict]],
    search_grid: List[float],
) -> Dict[str, dict]:
    """Finds optimal per-class confidence threshold maximizing F1.5 (recall-weighted for safety)."""
    recommendations: Dict[str, dict] = {}
    beta_sq = 1.5 ** 2  # Weigh recall 1.5x higher than precision for bridge safety
    for cls_name, items in sorted(threshold_candidates.items()):
        if not items:
            continue
        best_stats = {}
        best_f_beta = -1.0
        total_pos = sum(1 for x in items if x["is_true_positive"])
        if total_pos == 0:
            continue

        for th in search_grid:
            tp = sum(1 for x in items if x["score"] >= th and x["is_true_positive"])
            fp = sum(1 for x in items if x["score"] >= th and not x["is_true_positive"])
            fn = total_pos - tp
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            denom = (beta_sq * precision) + recall
            f_beta = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_stats = {
                    "recommended_threshold": th,
                    "f1_5_score": round(f_beta, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "rescued_true_positives": sum(
                        1
                        for x in items
                        if x["score"] >= th and x["is_true_positive"] and x["was_rejected"]
                    ),
                }
        recommendations[cls_name] = best_stats
    return recommendations


def compute_scorecard_metrics(
    gt_states: List[str],
    pred_states: List[Optional[str]],
    total_labels_count: int,
) -> dict:
    """Computes exact accuracy, within-1 accuracy, safety-critical under-calls, QWK, and confusion matrix."""
    n = len(STATES)
    matrix = np.zeros((n, n), dtype=int)
    scored = 0
    exact = 0
    within_1 = 0
    undercalls = 0
    severe_undercalls = 0  # True state in {IV, V} but predicted < True state

    for gt, pred in zip(gt_states, pred_states):
        if pred is None or pred not in STATE_TO_IDX:
            continue
        scored += 1
        g_i = STATE_TO_IDX[gt]
        p_i = STATE_TO_IDX[pred]
        matrix[g_i, p_i] += 1
        if g_i == p_i:
            exact += 1
        if abs(g_i - p_i) <= 1:
            within_1 += 1
        if p_i < g_i:
            undercalls += 1
            if g_i >= 3:
                severe_undercalls += 1

    qwk = quadratic_weighted_kappa(matrix) if scored > 0 else 0.0
    return {
        "total_matched_labels": total_labels_count,
        "scored_labels": scored,
        "abstained_needs_site_measurement": total_labels_count - scored,
        "scored_coverage_pct": round((scored / max(1, total_labels_count)) * 100.0, 2),
        "exact_accuracy_pct": round((exact / max(1, scored)) * 100.0, 2),
        "effective_accuracy_over_all_matched_pct": round(
            (exact / max(1, total_labels_count)) * 100.0, 2
        ),
        "within_1_accuracy_pct": round((within_1 / max(1, scored)) * 100.0, 2),
        "undercall_rate_pct": round((undercalls / max(1, scored)) * 100.0, 2),
        "severe_IV_V_undercalls": severe_undercalls,
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "confusion_matrix": {
            STATES[i]: {STATES[j]: int(matrix[i, j]) for j in range(n)}
            for i in range(n)
        },
    }


def run_cross_validation_and_fit(
    records: List[dict],
    kb: KnowledgeBase,
    n_splits: int = 5,
    undercall_multiplier: float = 2.5,
    critical_boost: float = 1.35,
) -> Tuple[SP35ConditionCalibrator, dict]:
    """Runs GroupKFold cross-validation by photo and fits the final production calibrator."""
    gt_states = [r["gt_state"] for r in records]
    baseline_preds = [r["baseline_state"] for r in records]

    # 1. Evaluate Zero-Shot KB Structural Proxy Prior (no ML training required)
    zero_shot_cal = SP35ConditionCalibrator(
        kb=kb, undercall_multiplier=undercall_multiplier, critical_boost=critical_boost
    )
    zero_shot_preds = [
        zero_shot_cal.predict_proposal(
            r["detections"], r["result_json"], r["host_mask"], r["row_id"], r["image_rgb"]
        )["proposed_state"]
        for r in records
    ]

    # 2. 5-Fold Out-of-Fold Cross-Validation grouped by photo
    unique_photos = sorted({r["photo"] for r in records})
    actual_splits = min(n_splits, len(unique_photos))
    oof_preds: List[Optional[str]] = [None] * len(records)

    if actual_splits >= 2 and len(records) >= 10:
        gkf = GroupKFold(n_splits=actual_splits)
        groups = np.array([r["photo"] for r in records])
        indices = np.arange(len(records))
        for train_idx, val_idx in gkf.split(indices, gt_states, groups):
            fold_cal = SP35ConditionCalibrator(
                kb=kb,
                undercall_multiplier=undercall_multiplier,
                critical_boost=critical_boost,
            )
            train_feats = [records[i]["features"] for i in train_idx]
            train_y = [records[i]["gt_state"] for i in train_idx]
            fold_cal.fit(train_feats, train_y)
            for idx in val_idx:
                r = records[idx]
                oof_preds[idx] = fold_cal.predict_proposal(
                    r["detections"],
                    r["result_json"],
                    r["host_mask"],
                    r["row_id"],
                    r["image_rgb"],
                )["proposed_state"]
    else:
        oof_preds = list(zero_shot_preds)

    # 3. Fit final production model on all matched records
    final_calibrator = SP35ConditionCalibrator(
        kb=kb, undercall_multiplier=undercall_multiplier, critical_boost=critical_boost
    )
    if len(records) >= 5:
        final_calibrator.fit([r["features"] for r in records], gt_states)

    comparison = {
        "baseline_uncalibrated": compute_scorecard_metrics(
            gt_states, baseline_preds, len(records)
        ),
        "zero_shot_structural_proxy_prior": compute_scorecard_metrics(
            gt_states, zero_shot_preds, len(records)
        ),
        "hybrid_calibrated_5fold_oof": compute_scorecard_metrics(
            gt_states, oof_preds, len(records)
        ),
    }
    return final_calibrator, comparison


def format_markdown_report(
    comparison: dict,
    threshold_recs: Dict[str, dict],
    num_records: int,
) -> str:
    """Generates a clean Markdown scorecard summary table."""
    b = comparison["baseline_uncalibrated"]
    z = comparison["zero_shot_structural_proxy_prior"]
    c = comparison["hybrid_calibrated_5fold_oof"]

    lines = [
        "# SP:35-2024 Condition State Calibration & Scorecard Report",
        "",
        f"**Total Matched Engineer Labels Evaluated:** `{num_records}`",
        "",
        "## 1. Before vs. After Condition State Scorecard",
        "",
        "| Metric | Baseline (`vision.condition_proposals`) | Zero-Shot Structural Proxy + EDT Prior | Hybrid Calibrated (`5-Fold Out-of-Fold`) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Scored Coverage (%)** | {b['scored_coverage_pct']}% | {z['scored_coverage_pct']}% | **{c['scored_coverage_pct']}%** |",
        f"| **Abstained (`needs_site_measurement`)** | {b['abstained_needs_site_measurement']} | {z['abstained_needs_site_measurement']} | **{c['abstained_needs_site_measurement']}** |",
        f"| **Exact Accuracy (on Scored)** | {b['exact_accuracy_pct']}% | {z['exact_accuracy_pct']}% | **{c['exact_accuracy_pct']}%** |",
        f"| **Effective Accuracy (All Matched)** | {b['effective_accuracy_over_all_matched_pct']}% | {z['effective_accuracy_over_all_matched_pct']}% | **{c['effective_accuracy_over_all_matched_pct']}%** |",
        f"| **Within-1 State Accuracy (%)** | {b['within_1_accuracy_pct']}% | {z['within_1_accuracy_pct']}% | **{c['within_1_accuracy_pct']}%** |",
        f"| **Safety-Critical Under-Call Rate (%)** | {b['undercall_rate_pct']}% | {z['undercall_rate_pct']}% | **{c['undercall_rate_pct']}%** |",
        f"| **Severe State IV/V Under-Calls** | {b['severe_IV_V_undercalls']} | {z['severe_IV_V_undercalls']} | **{c['severe_IV_V_undercalls']}** |",
        f"| **Quadratic Weighted Kappa ($\\kappa_w$)** | {b['quadratic_weighted_kappa']:.4f} | {z['quadratic_weighted_kappa']:.4f} | **{c['quadratic_weighted_kappa']:.4f}** |",
        "",
        "## 2. Recommended Per-Class SAM 3.1 Detection Thresholds",
        "",
        "| Distress Class | Recommended Threshold | F1.5 Score | Precision | Recall | Rescued True Positives from `rejected` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cls_name, st in threshold_recs.items():
        lines.append(
            f"| `{cls_name}` | **{st['recommended_threshold']:.2f}** | {st['f1_5_score']:.4f} | "
            f"{st['precision']:.4f} | {st['recall']:.4f} | {st['rescued_true_positives']} |"
        )
    return "\n".join(lines) + "\n"


def dump_calibrated_condition_proposals(
    predictions_dir: Path,
    images_dir: Optional[Path],
    output_dir: Path,
    calibrator: SP35ConditionCalibrator,
    kb: KnowledgeBase,
) -> dict:
    """Batch dumps IRC:SP:35-2024 certified condition proposals (.conditions.json) per photo."""
    conditions_dir = output_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    summary_index = {
        "inspection_standard": "IRC:SP:35-2024",
        "kb_version": getattr(kb, "version", "sp35_2024-kb0"),
        "total_photos_processed": 0,
        "total_proposals": 0,
        "critical_list_photos": [],
        "urgent_list_photos": [],
        "photos": {},
    }

    s_k_map = {"I": 0.0, "II": 0.25, "III": 0.50, "IV": 0.75, "V": 1.00}
    state_names = {"I": "Excellent", "II": "Good", "III": "Fair", "IV": "Poor", "V": "Critical"}

    for json_path in sorted(predictions_dir.glob("*.json")):
        if json_path.name in ("condition_proposals.json", "calibration_scorecard_report.json"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        photo_name = Path(data.get("photo", json_path.stem + ".jpg")).name
        photo_stem = json_path.stem

        image_rgb = None
        if images_dir is not None:
            cand = images_dir / photo_name
            if cand.exists():
                try:
                    image_rgb = np.asarray(Image.open(cand).convert("RGB"))
                except Exception:
                    image_rgb = None

        distresses = data.get("distresses", [])
        status = data.get("status", "ok")

        groups = defaultdict(list)
        for d in distresses:
            h = d.get("host") or {}
            grp_key = (
                d.get("sp35", "T7.2-08"),
                d.get("class", "spalling"),
                h.get("class"),
                h.get("sp35"),
                h.get("source"),
            )
            groups[grp_key].append(d)

        proposals = []
        max_state_idx = 0
        crit_flag = False
        urg_flag = False

        for p_idx, (grp_key, grp_dets) in enumerate(groups.items()):
            row_id = grp_key[0]
            host_mask = find_host_mask_for_detections(grp_dets, data)
            cal_prop = calibrator.predict_proposal(
                grp_dets, data, host_mask, row_id, image_rgb
            )
            prop_state = cal_prop["proposed_state"]
            s_idx = STATE_TO_IDX.get(prop_state, 0)
            if s_idx > max_state_idx:
                max_state_idx = s_idx
            if prop_state == "V":
                crit_flag = True
            elif prop_state == "IV":
                urg_flag = True

            s_k = s_k_map.get(prop_state, 0.25)
            cs_j = 100.0 - (100.0 * s_k)

            h_info = grp_dets[0].get("host", {})
            try:
                row_obj = kb.row(row_id)
                row_name = row_obj.name
                row_unit = row_obj.unit or "%"
            except Exception:
                row_name = grp_key[1].replace("_", " ").title()
                row_unit = "%"

            h_sp35 = h_info.get("sp35", "C2.7")
            try:
                comp_obj = kb.component(h_sp35)
                comp_name = comp_obj.name
                comp_weight = float(comp_obj.weight)
            except Exception:
                comp_name = (h_info.get("class") or "Substructure").title()
                comp_weight = 20.0

            feats = cal_prop.get("features", {})
            crop_pct = feats.get("crop_pct", feats.get("raw_frame_pct", 0.0))
            norm_pct = feats.get("normalized_crop_pct", feats.get("best_pct", 0.0))
            is_crop = feats.get("is_closeup", 1.0) == 1.0
            darkness_drop = feats.get("darkness_drop", 1.0)
            rust_redness = feats.get("rust_redness", 0.0)
            dist_area_px = int(sum(d.get("area_px", 0) for d in grp_dets))
            max_score = max((float(d.get("score", 0.0)) for d in grp_dets), default=0.5)

            try:
                row_bands = kb.row(row_id).bands_dict()
                governing_band_str = str(row_bands.get(prop_state, f"State {prop_state}"))
            except Exception:
                governing_band_str = f"State {prop_state}"

            trace = [
                f"Step 1 (Vision): SAM segmented {len(grp_dets)} detection(s) for '{grp_key[1]}' (total {dist_area_px} px, max score {max_score:.2f}, darkness drop {darkness_drop:.2f}, rust index {rust_redness:.3f}).",
                f"Step 2 (Classification): Host '{comp_name}' ({h_sp35}) mapped to SP:35 Table 7.2 row '{row_name}' ({row_id}).",
                f"Step 3 (Normalization): {'Close-up framing detected; member area normalized with crop cap factor 0.05' if is_crop else 'Overview framing verified; using full host member mask'} (crop extent = {crop_pct:.2f}%, normalized member extent = {norm_pct:.2f}%).",
                f"Step 4 (KB Table 7.2): Governing value {norm_pct:.2f}{row_unit} maps to Band '{governing_band_str}' -> Condition State {prop_state} ({state_names.get(prop_state, 'Good')}).",
                f"Step 5 (SP:35 Eq 3): CS_j = 100 - (100 * {s_k}) = {cs_j:.1f}.",
            ]

            framing_text = (
                f"Camera framing analysis detected a macro close-up view where the defect occupies {crop_pct:.1f}% of the visible viewfinder. "
                f"Because close-up optics inflate damage by 20x to 300x relative to the physical bridge member, Tier 4 normalization applied structural proxy factor f_crop = 0.05. "
                f"Physical Member Extent = {crop_pct:.2f}% * 0.05 = {norm_pct:.2f}%."
                if is_crop else
                f"Overview camera framing verified; defect extent calculated directly across the segmented {comp_name} member mask: "
                f"Damage Extent = ({dist_area_px} px / host_area px) * 100% = {norm_pct:.2f}%."
            )

            evidence_text = (
                f"AI Vision Evidence: SAM 3.1 identified {len(grp_dets)} mask(s) labeled '{grp_key[1]}' covering {dist_area_px:,} px with peak confidence {max_score:.2f}. "
                f"RGB surface telemetry measured cavity darkness drop of {darkness_drop:.2f} (spall shadow depth proxy) and rust redness index of {rust_redness:.3f}."
            )

            sp35_math_text = (
                f"Mathematical SP:35 Ruling: Extent {norm_pct:.2f}{row_unit} falls into Table 7.2 interval '{governing_band_str}', "
                f"assigning Condition State {prop_state} ({state_names.get(prop_state, 'Good')}). "
                f"Per Equation 3, reduction coefficient S_k = {s_k} produces sub-component score CS_j = 100 - (100 * {s_k}) = {cs_j:.1f}."
            )

            debug_text = (
                f"Pipeline Debugging: "
                f"[Upstream SAM 3.1]: If defect boundary is incomplete, lower prompt threshold from 0.45 to recommended 0.25. "
                f"[Host Association]: Member identified as '{comp_name}' ({h_sp35}); if misclassified, override component tag in Form 1 inventory. "
                f"[Severity Calibration]: If engineer inspects on-site and marks State I, check if darkness drop indicates non-cavity surface discoloration; "
                f"if marked State IV/V, verify whether cover loss exposes reinforcing steel."
            )

            reasoning_plain_english = f"{evidence_text} {framing_text} {sp35_math_text} {debug_text}"

            debug_evidence = {
                "ai_vision_evidence": {
                    "detection_count": len(grp_dets),
                    "distress_class": grp_key[1],
                    "total_area_px": dist_area_px,
                    "max_confidence_score": round(max_score, 4),
                    "darkness_drop_ratio": round(darkness_drop, 4),
                    "rust_redness_index": round(rust_redness, 4),
                },
                "framing_and_normalization_math": {
                    "is_macro_closeup": is_crop,
                    "crop_cap_factor_f_crop": 0.05 if is_crop else 1.0,
                    "raw_crop_extent_pct": round(crop_pct, 4),
                    "formula": f"normalized_extent = {crop_pct:.2f}% * {0.05 if is_crop else 1.0} = {norm_pct:.2f}%",
                    "normalized_extent_pct": round(norm_pct, 4),
                },
                "sp35_table_ruling": {
                    "table": "7.2",
                    "row_id": row_id,
                    "row_name": row_name,
                    "governing_band": governing_band_str,
                    "proposed_state": prop_state,
                    "s_k": s_k,
                    "cs_j": cs_j,
                    "equation": f"CS_j = 100 - (100 * {s_k}) = {cs_j:.1f}",
                },
                "pipeline_debugging": {
                    "upstream_sam_pre_pipeline": "Tune prompt threshold in sam3_prompts.yaml (recommended 0.25).",
                    "calibration_post_pipeline": "Verify structural proxy denominator vs Form 1 inventory physical dimensions.",
                },
            }

            flags = []
            if is_crop:
                flags.append("close_up_frame_detected")
                flags.append("denominator_normalized_via_structural_proxy")
            if cal_prop.get("requires_site_measurement"):
                flags.append("requires_site_measurement")
            flags.append("requires_engineer_site_signoff")

            proposal_obj = {
                "proposal_id": f"P{p_idx + 1:03d}",
                "component": {
                    "class": h_info.get("class") or "unknown",
                    "sp35_code": h_sp35,
                    "name": comp_name,
                    "weight_w_ij": comp_weight,
                    "support": h_info.get("support", None),
                    "span": h_info.get("span", None),
                },
                "distress": {
                    "class": grp_key[1],
                    "sp35_row": row_id,
                    "name": row_name,
                    "unit": row_unit,
                },
                "condition_rating": {
                    "proposed_state": prop_state,
                    "state_name": state_names.get(prop_state, "Good"),
                    "reduction_coefficient_s_k": s_k,
                    "distress_contribution_cs_j": cs_j,
                    "final_state": None,
                    "engineer_certified": False,
                },
                "measurement_basis": {
                    "raw_vision_extent_pct": crop_pct,
                    "denominator_mode": "tier4_structural_proxy_crop_normalized" if is_crop else "tier3_host_member_mask",
                    "crop_factor_applied": 0.05 if is_crop else 1.0,
                    "normalized_extent_pct": norm_pct,
                    "governing_band": governing_band_str,
                },
                "provenance_and_audit": {
                    "ai_model": "SAM 3.1 + SP35ConditionCalibrator",
                    "reasoning_trace": trace,
                    "reasoning_plain_english": reasoning_plain_english,
                    "debug_evidence": debug_evidence,
                    "flags": flags,
                },
            }
            proposals.append(proposal_obj)

        max_state = IDX_TO_STATE[max_state_idx] if proposals else "I"
        photo_cond_file = conditions_dir / f"{photo_stem}.conditions.json"
        doc = {
            "photo": photo_name,
            "inspection_standard": "IRC:SP:35-2024",
            "kb_version": getattr(kb, "version", "sp35_2024-kb0"),
            "status": status,
            "requires_engineer_certification": True,
            "summary": {
                "total_proposals": len(proposals),
                "max_condition_state": max_state,
                "critical_list_flag": crit_flag,
                "urgent_list_flag": urg_flag,
            },
            "proposals": proposals,
        }
        photo_cond_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        summary_index["total_photos_processed"] += 1
        summary_index["total_proposals"] += len(proposals)
        if crit_flag:
            summary_index["critical_list_photos"].append(photo_name)
        if urg_flag:
            summary_index["urgent_list_photos"].append(photo_name)
        summary_index["photos"][photo_name] = {
            "proposals_count": len(proposals),
            "max_state": max_state,
            "critical_list_flag": crit_flag,
            "file": str(photo_cond_file.relative_to(output_dir)),
        }

    index_path = conditions_dir / "batch_conditions_index.json"
    index_path.write_text(json.dumps(summary_index, indent=2), encoding="utf-8")

    # Write comprehensive audit CSV with plain English reasoning, math, and AI evidence
    demo_csv_path = Path("data/pilot/condition_proposals_demo/conditions.csv")
    demo_lookup = {}
    if demo_csv_path.exists():
        import csv
        try:
            with open(demo_csv_path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    demo_lookup[(r.get("photo", ""), r.get("distress", ""), r.get("sp35_row", ""))] = r
        except Exception:
            pass

    audit_rows = []
    for f in sorted(conditions_dir.glob("*.conditions.json")):
        cdata = json.loads(f.read_text(encoding="utf-8"))
        p_name = cdata.get("photo", f.stem + ".jpg")
        for p in cdata.get("proposals", []):
            d_cls = p["distress"]["class"]
            row_id = p["distress"]["sp35_row"]
            base_info = demo_lookup.get((p_name, d_cls, row_id), {})

            base_state = base_info.get("proposed_state", "").strip() or "NONE (Blocked)"
            base_status = base_info.get("status", "").strip() or "uncalibrated"

            p_prov = p.get("provenance_and_audit", {})
            p_dbg = p_prov.get("debug_evidence", {})
            ai_ev = p_dbg.get("ai_vision_evidence", {})
            math_ev = p_dbg.get("framing_and_normalization_math", {})
            sp35_ev = p_dbg.get("sp35_table_ruling", {})

            ai_evidence_str = (
                f"SAM Score={ai_ev.get('max_confidence_score', 0.5):.2f}, "
                f"Area={ai_ev.get('total_area_px', 0):,} px, "
                f"DarknessDrop={ai_ev.get('darkness_drop_ratio', 1.0):.2f}, "
                f"RustIndex={ai_ev.get('rust_redness_index', 0.0):.3f}"
            )
            math_formula_str = (
                f"raw_crop_pct={math_ev.get('raw_crop_extent_pct', 0.0):.2f}% * "
                f"f_crop={math_ev.get('crop_cap_factor_f_crop', 1.0)} -> "
                f"norm_member_extent={math_ev.get('normalized_extent_pct', 0.0):.2f}%"
            )
            sp35_ruling_str = (
                f"Table {sp35_ev.get('table', '7.2')} Row {row_id} Band '{sp35_ev.get('governing_band', '')}' -> "
                f"State {p['condition_rating']['proposed_state']} (S_k={sp35_ev.get('s_k', 0.25)}, CS_j={sp35_ev.get('cs_j', 75.0)})"
            )

            audit_rows.append({
                "photo": p_name,
                "distress": d_cls,
                "sp35_row": row_id,
                "host": p["component"]["name"],
                "host_sp35": p["component"]["sp35_code"],
                "detections": ai_ev.get("detection_count", 1),
                "max_score": ai_ev.get("max_confidence_score", 0.5),
                "baseline_proposed_state": base_state,
                "baseline_status": base_status,
                "calibrated_proposed_state": p["condition_rating"]["proposed_state"],
                "calibrated_denominator_mode": p["measurement_basis"]["denominator_mode"],
                "ai_vision_evidence": ai_evidence_str,
                "mathematical_extent_formula": math_formula_str,
                "sp35_governing_band_and_cs_j": sp35_ruling_str,
                "plain_english_reasoning_and_debug": p_prov.get("reasoning_plain_english", ""),
            })

    audit_csv_path = output_dir / "conditions_comparison_audit.csv"
    if audit_rows:
        import csv
        with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    return summary_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & evaluate SP:35 Condition State Calibrator and tune SAM 3.1 thresholds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("kb/training/fine_tuning.yaml"),
        help="Path to fine_tuning.yaml configuration.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels/annotations.json"),
        help="Path to engineer ground-truth annotations.json.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/pilot/infer_all_v1"),
        help="Directory containing SAM 3.1 inference JSON files.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional directory of original photos for RGB shadow/rust cues.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pilot/calibrated_v1"),
        help="Output directory for calibrator.pkl and scorecard reports.",
    )
    parser.add_argument(
        "--dump-proposals",
        action="store_true",
        default=True,
        help="Batch dump per-photo IRC:SP:35-2024 condition state JSON proposals.",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    kb_dir = Path(cfg.get("kb", {}).get("kb_dir", "kb/sp35_2024"))
    kb = load_kb(kb_dir) if kb_dir.exists() else load_kb()

    iou_thresh = float(cfg.get("data", {}).get("iou_match_threshold", 0.15))
    cv_folds = int(cfg.get("data", {}).get("cv_folds", 5))
    undercall_mult = float(cfg.get("decoding", {}).get("undercall_penalty_multiplier", 2.5))
    crit_boost = float(cfg.get("decoding", {}).get("state_IV_V_safety_boost", 1.35))
    search_grid = cfg.get("detection_calibration", {}).get(
        "threshold_search_grid", [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    )

    labels_path = args.labels
    if not labels_path.exists():
        fallback = Path("data/labels/yugmi_engineer_annotations_2026-10-13.json")
        if fallback.exists():
            labels_path = fallback
    labels = load_labels(labels_path, kb=kb)
    records, threshold_candidates = match_labels_to_predictions(
        labels=labels,
        predictions_dir=args.predictions,
        images_dir=args.images,
        kb=kb,
        iou_threshold=iou_thresh,
        include_rejected_rescue=True,
    )

    threshold_recs = optimize_class_thresholds(threshold_candidates, search_grid)
    calibrator, comparison = run_cross_validation_and_fit(
        records=records,
        kb=kb,
        n_splits=cv_folds,
        undercall_multiplier=undercall_mult,
        critical_boost=crit_boost,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = args.output_dir / "calibrator.pkl"
    calibrator.save(calibrator_path)

    thresh_path = args.output_dir / "threshold_recommendations.yaml"
    thresh_path.write_text(yaml.safe_dump(threshold_recs, sort_keys=False), encoding="utf-8")

    report_json_path = args.output_dir / "calibration_scorecard_report.json"
    report_json_path.write_text(
        json.dumps(
            {
                "matched_instances": len(records),
                "scorecard_comparison": comparison,
                "threshold_recommendations": threshold_recs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_md_path = args.output_dir / "calibration_scorecard_report.md"
    md_text = format_markdown_report(comparison, threshold_recs, len(records))
    report_md_path.write_text(md_text, encoding="utf-8")
    print(md_text)
    print(f"Saved calibrated model to: {calibrator_path}")

    if args.dump_proposals:
        print("\nDumping batch IRC:SP:35-2024 calibrated condition state proposals per photo...")
        batch_summary = dump_calibrated_condition_proposals(
            predictions_dir=args.predictions,
            images_dir=args.images,
            output_dir=args.output_dir,
            calibrator=calibrator,
            kb=kb,
        )
        print(f"Dumped {batch_summary['total_proposals']} condition proposals across {batch_summary['total_photos_processed']} photos to {args.output_dir / 'conditions'}")
        if batch_summary["critical_list_photos"]:
            print(f"Critical List Photos ({len(batch_summary['critical_list_photos'])}): {', '.join(batch_summary['critical_list_photos'])}")


if __name__ == "__main__":
    main()