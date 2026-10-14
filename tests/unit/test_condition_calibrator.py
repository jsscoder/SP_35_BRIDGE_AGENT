"""Unit tests for SP:35 Condition State Calibrator & Integrations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sp35.kb import STATES, load_kb
from vision.condition_calibrator import (
    IDX_TO_STATE,
    STATE_TO_IDX,
    SP35ConditionCalibrator,
    bbox_iou,
    build_asymmetric_cost_matrix,
    compute_corrected_denominators,
    decode_optimal_state,
    extract_distress_features,
    extract_medial_crack_geometry,
    extract_rgb_severity_features,
    valid_state_mask_for_row,
)
from vision.condition_proposals import propose_photo


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_bbox_iou():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    # Intersection: [5, 0, 10, 10] -> 5 * 10 = 50
    # Union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    assert abs(bbox_iou(box1, box2) - 0.3333) < 0.001
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_valid_state_mask_respects_kb_null_bands(kb):
    # Standard % row: Spalling (T7.2-08) has all states I..V
    mask_spall = valid_state_mask_for_row(kb, "T7.2-08")
    assert mask_spall.all()

    # T7.2-16 (Crushing) has State II as null in SP:35 Table 7.2
    mask_crush = valid_state_mask_for_row(kb, "T7.2-16")
    assert mask_crush[STATE_TO_IDX["II"]] is False or mask_crush[STATE_TO_IDX["II"]] == False
    assert mask_crush[STATE_TO_IDX["I"]] == True
    assert mask_crush[STATE_TO_IDX["III"]] == True


def test_compute_corrected_denominators():
    result = {
        "processed_size": [1000, 1000],
        "suppressors": [{"area_px": 200_000}],
        "components": [{"area_px": 300_000}],
    }
    host_mask = np.zeros((1000, 1000), dtype=bool)
    host_mask[:200, :200] = True  # 40,000 px

    denoms = compute_corrected_denominators(result, host_mask)
    assert denoms["frame_area"] == 1_000_000.0
    assert denoms["effective_frame_area"] == 800_000.0
    assert denoms["component_area"] == 300_000.0
    assert denoms["explicit_host_area"] == 40_000.0


def test_extract_medial_crack_geometry():
    # Empty mask
    empty = np.zeros((50, 50), dtype=bool)
    w_max, w_mean, length = extract_medial_crack_geometry(empty)
    assert (w_max, w_mean, length) == (0.0, 0.0, 0.0)

    # 10-pixel wide vertical crack of height 40
    crack = np.zeros((100, 100), dtype=bool)
    crack[20:60, 45:55] = True  # area = 40 * 10 = 400
    w_max, w_mean, length = extract_medial_crack_geometry(crack)
    assert w_max >= 8.0
    assert w_mean >= 6.0
    assert length > 0.0


def test_extract_rgb_severity_features():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True

    # Darker inside (shadow/depth proxy)
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    img[10:30, 10:30] = 50

    feats = extract_rgb_severity_features(img, mask)
    assert feats["darkness_drop"] < 1.0
    assert "rust_redness" in feats
    assert "intensity_std" in feats


def test_asymmetric_cost_matrix_penalizes_undercalls():
    cost = build_asymmetric_cost_matrix(5, undercall_multiplier=2.5, critical_boost=1.35)
    # Correct prediction: 0 cost
    for i in range(5):
        assert cost[i, i] == 0.0

    # Under-call (true State IV, pred State II) vs Over-call (true State II, pred State IV)
    # True=3 (IV), Pred=1 (II) is an undercall
    under_cost = cost[3, 1]
    # True=1 (II), Pred=3 (IV) is an overcall
    over_cost = cost[1, 3]
    assert under_cost > over_cost * 2.0


def test_decode_optimal_state_respects_null_mask():
    cost = build_asymmetric_cost_matrix(5)
    # High probability on State II, but State II is masked out
    proba = np.array([0.05, 0.85, 0.05, 0.03, 0.02])
    valid_mask = np.array([True, False, True, True, True])  # State II masked

    best_state, conf, prob_dict = decode_optimal_state(proba, valid_mask, cost)
    assert best_state != "II"
    assert prob_dict["II"] == 0.0


def test_calibrator_fit_save_load_roundtrip(kb, tmp_path):
    calibrator = SP35ConditionCalibrator(kb=kb)

    # Create dummy training data
    features_list = []
    y_states = []
    for i in range(25):
        st = STATES[i % 5]
        feats = {
            "kb_prior_idx": float(i % 5),
            "raw_frame_pct": float(i * 2),
            "effective_frame_pct": float(i * 2.2),
            "structural_proxy_pct": float(i * 2.5),
            "best_pct": float(i * 2),
            "det_count": float(1 + (i % 3)),
            "max_score": 0.8,
            "mean_score": 0.75,
            "host_missing": 0.0,
            "is_site_row": 0.0,
            "is_count_row": 0.0,
            "max_width_px": 5.0,
            "mean_width_px": 3.0,
            "rel_length": 0.1,
            "compound_rejected_count": 0.0,
            "max_compound_rejected_score": 0.0,
            "darkness_drop": 0.8,
            "rust_redness": 0.1,
            "intensity_std": 0.05,
            "row_hash_mod": 12.0,
        }
        features_list.append(feats)
        y_states.append(st)

    fit_stats = calibrator.fit(features_list, y_states)
    assert fit_stats["samples"] == 25
    assert calibrator.is_fitted is True

    # Save and reload
    save_file = tmp_path / "test_calibrator.pkl"
    calibrator.save(save_file)
    assert save_file.exists()

    loaded = SP35ConditionCalibrator.load(save_file, kb=kb)
    assert loaded.is_fitted is True
    assert loaded.feature_names == calibrator.feature_names


def test_calibrator_propose_photo_integration(kb):
    calibrator = SP35ConditionCalibrator(kb=kb)

    # Synthetic inference result for cracking (site measurement row T7.2-11)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 50:52] = True
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts, prev, run = [], 0, 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            run, prev = 1, v
    counts.append(run)
    rle_dict = {"size": [100, 100], "counts": counts}

    result = {
        "photo": "test_crack.jpg",
        "status": "ok",
        "processed_size": [100, 100],
        "distresses": [
            {
                "_index": 0,
                "class": "cracking",
                "sp35": "T7.2-11",
                "score": 0.88,
                "bbox": [50, 20, 52, 80],
                "rle": rle_dict,
                "host": {"class": "pier", "sp35": "C2.7", "source": "iou_overlap"},
            }
        ],
        "components": [
            {
                "class": "pier",
                "sp35": "C2.7",
                "rle": {"size": [100, 100], "counts": [0, 10000]},
            }
        ],
    }

    # Propose with calibrator
    proposals_with_cal = propose_photo(result, kb, calibrator=calibrator)
    props = proposals_with_cal["proposals"]
    assert len(props) == 1
    # Site row without scale now has calibrated visual proposal with safety certification requirement
    assert props[0]["status"] == "calibrated_visual_proposal"
    assert props[0]["proposed_state"] in STATES
    assert "ai_visual_proposal_requires_site_certification" in props[0]["flags"]

