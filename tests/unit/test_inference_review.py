import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from tools.inference_review import (BOX_STYLE, build_defect_coverage, decode_rle, mask_iou, match_masks,
                                    prediction_record, render_defect_coverage_html, write_report)


def uncompressed_rle(mask):
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts, previous, run = [], 0, 0
    for value in flat:
        if int(value) == previous:
            run += 1
        else:
            counts.append(run)
            run, previous = 1, int(value)
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_decode_uncompressed_coco_rle_round_trip():
    expected = np.array([[True, False, True], [True, False, False]], dtype=bool)
    assert np.array_equal(decode_rle(uncompressed_rle(expected)), expected)


def test_decode_rle_rejects_invalid_counts():
    with pytest.raises(ValueError, match="do not cover"):
        decode_rle({"size": [2, 2], "counts": [3]})


def test_defect_coverage_compares_yaml_catalog_to_distress_proposals(tmp_path):
    knowledge = tmp_path / "kb"
    knowledge.mkdir()
    (knowledge / "distress_bridge.yaml").write_text(
        "table: '7.2'\ndistresses:\n"
        "  - {id: T7.2-01, name: Erosion}\n"
        "  - {id: T7.2-02, name: Rust}\n",
        encoding="utf-8",
    )
    (knowledge / "distress_approach.yaml").write_text(
        "table: '7.3'\ndistresses:\n  - {id: T7.3-01, name: Potholes}\n",
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        "distresses:\n"
        "  - {id: erosion, sp35_row: T7.2-01}\n"
        "  - {id: rust_stain, sp35_row: T7.2-02}\n"
        "  - {id: pothole, sp35_row: T7.3-01}\n",
        encoding="utf-8",
    )

    coverage = build_defect_coverage(
        Counter({"erosion": 4, "pothole": 2, "unmapped_label": 5}),
        knowledge,
        prompts,
        source_photo_count=12,
        inference_result_photo_count=11,
        successful_photo_count=10,
    )

    assert coverage["dataset"] == {
        "source_photos": 12,
        "inference_result_photos": 11,
        "successful_inference_photos": 10,
    }
    assert coverage["catalog_row_count"] == 3
    assert coverage["prompted_catalog_rows"] == 3
    assert coverage["observed_prompted_rows"] == 2
    assert coverage["proposal_count"] == 11
    assert coverage["unmapped_proposal_counts"] == {"unmapped_label": 5}
    assert coverage["catalog_rows_by_table"]["7.2"] == {
        "catalog_rows": 2, "prompted_rows": 2, "observed_prompted_rows": 1, "proposal_count": 4,
    }
    html = render_defect_coverage_html(coverage)
    assert "not the number of true defects present, unique physical defects, recall, or accuracy" in html
    assert "unmapped_label: 5" in html
    assert "T7.2-02" in html and "Not prompted" not in html


def test_defect_coverage_rejects_prompt_rows_missing_from_catalog(tmp_path):
    knowledge = tmp_path / "kb"
    knowledge.mkdir()
    (knowledge / "distress_bridge.yaml").write_text("distresses: []\n", encoding="utf-8")
    (knowledge / "distress_approach.yaml").write_text("distresses: []\n", encoding="utf-8")
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text("distresses:\n  - {id: unknown, sp35_row: T7.2-99}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown SP:35 distress rows"):
        build_defect_coverage(Counter(), knowledge, prompts, 0, 0, 0)


def test_mask_iou_and_one_to_one_matching():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [0, 0]], dtype=bool)
    result = match_masks([{"score": 0.9, "mask": a}, {"score": 0.8, "mask": b}],
                         [{"mask": a}], 0.5)
    assert mask_iou(a, b) == 0.5
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["matched_mask_iou"] == 1


def test_write_visual_only_report_does_not_claim_metrics(tmp_path):
    predictions, photos, output = tmp_path / "predictions", tmp_path / "photos", tmp_path / "review"
    predictions.mkdir()
    photos.mkdir()
    Image.new("RGB", (4, 3), "white").save(photos / "bridge.jpg")
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1] = True
    (predictions / "bridge.json").write_text(json.dumps({
        "photo": "bridge.jpg", "status": "ok", "native_size": [4, 3],
        "components": [{"class": "pier", "kind": "component", "score": 0.9,
                        "rle": uncompressed_rle(np.array([[False, False], [False, True]]))}],
        "distresses": [],
    }), encoding="utf-8")

    summary = write_report(predictions, photos, output, None, 0.5)

    assert summary["photos_with_successful_inference"] == 1
    assert summary["classes"] == []
    assert summary["detection_counts_by_class"] == {"pier": 1}
    assert "no accuracy metrics" in summary["note"]
    assert (output / "bridge_review.png").exists()
    assert Image.open(output / "bridge_review.png").size == (8, 31)
    assert (output / "detection_counts.png").exists()
    assert (output / "index.html").exists()
    assert not (output / "metrics.csv").exists()


def test_write_report_compares_coco_ground_truth(tmp_path):
    predictions, photos, output = tmp_path / "predictions", tmp_path / "photos", tmp_path / "review"
    predictions.mkdir()
    photos.mkdir()
    Image.new("RGB", (4, 3), "white").save(photos / "bridge.jpg")
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1] = True
    rle = uncompressed_rle(mask)
    (predictions / "bridge.json").write_text(json.dumps({
        "photo": "bridge.jpg", "status": "ok",
        "components": [{"class": "pier", "kind": "component", "score": 0.9, "rle": rle}],
        "distresses": [],
    }), encoding="utf-8")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "bridge.jpg", "height": 3, "width": 4}],
        "categories": [{"id": 1, "name": "pier"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": rle}],
    }), encoding="utf-8")

    summary = write_report(predictions, photos, output, gt, 0.5)

    assert summary["photos_with_ground_truth_match"] == 1
    assert summary["classes"][0]["tp"] == 1
    assert summary["classes"][0]["fp"] == summary["classes"][0]["fn"] == 0
    assert summary["classes"][0]["matched_mask_iou"] == 1
    assert (output / "metrics.csv").exists()


def test_ground_truth_matching_does_not_cross_photo_boundaries(tmp_path):
    predictions, photos, output = tmp_path / "predictions", tmp_path / "photos", tmp_path / "review"
    predictions.mkdir()
    photos.mkdir()
    for name in ("first.jpg", "second.jpg"):
        Image.new("RGB", (4, 3), "white").save(photos / name)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1] = True
    (predictions / "first.json").write_text(json.dumps({
        "photo": "first.jpg", "status": "ok",
        "components": [{"class": "pier", "score": 0.9, "rle": uncompressed_rle(mask)}],
        "distresses": [],
    }), encoding="utf-8")
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "second.jpg", "height": 3, "width": 4}],
        "categories": [{"id": 1, "name": "pier"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": uncompressed_rle(mask)}],
    }), encoding="utf-8")

    summary = write_report(predictions, photos, output, gt, 0.5)

    assert summary["classes"] == []
    assert summary["ground_truth_images_without_inference"] == 1


def _entity(cls, kind, box, shape, with_bbox=True):
    """An inference-JSON entity whose mask fills `box` (x0, y0, x1, y1) in a frame of `shape` (h, w)."""
    mask = np.zeros(shape, dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    item = {"class": cls, "kind": kind, "score": 0.9, "rle": uncompressed_rle(mask)}
    if with_bbox:
        item["bbox"] = list(box)
    return item


def test_review_boxes_components_and_distresses_but_not_suppressors_or_rejected(tmp_path):
    predictions, photos, output = tmp_path / "predictions", tmp_path / "photos", tmp_path / "review"
    predictions.mkdir()
    photos.mkdir()
    Image.new("RGB", (200, 160), "white").save(photos / "bridge.png")
    processed, native = (80, 100), (160, 200)            # (h, w): inference ran at half size
    raw = json.dumps({
        "photo": "bridge.png", "status": "ok", "native_size": [200, 160], "processed_size": [100, 80],
        "components": [_entity("pier", "component", (10, 20, 60, 70), processed)],
        "distresses": [_entity("spalling", "distress", (30, 40, 50, 60), processed)],
        "suppressors": [_entity("cable", "suppression", (0, 70, 100, 80), processed)],
        "rejected": [_entity("crack", "distress", (65, 50, 90, 75), processed)],
        "tiled": {"frame": "native", "size": [200, 160],
                  "distresses": [_entity("exposed_rebar", "distress", (150, 20, 190, 60), native)],
                  "rejected": [_entity("crack", "distress", (100, 145, 140, 158), native)]},
    })
    (predictions / "bridge.json").write_text(raw, encoding="utf-8")

    write_report(predictions, photos, output, None, 0.5)

    assert (predictions / "bridge.json").read_text(encoding="utf-8") == raw      # inference JSON untouched
    px = Image.open(output / "bridge_review.png").convert("RGB").crop((200, 28, 400, 188)).load()
    blue, red, white = BOX_STYLE["component"][0], BOX_STYLE["distress"][0], (255, 255, 255)
    assert px[20, 90] == px[119, 90] == px[70, 139] == blue     # pier: processed x2 -> (20, 40, 120, 140)
    assert px[60, 100] == px[99, 100] == red                     # spalling -> (60, 80, 100, 120)
    assert px[150, 40] == px[189, 40] == red                     # tiled rebar, already native
    assert px[130, 125] == px[179, 125] == white                 # rejected crack -> (130, 100, 180, 150)
    assert px[199, 150] == white                                 # suppressor cable -> (0, 140, 200, 160)
    assert px[100, 150] == px[139, 150] == white                 # tiled rejected crack


def test_prediction_record_falls_back_to_mask_extent_without_bbox():
    rec = prediction_record(_entity("pier", "component", (2, 3, 6, 8), (10, 10), with_bbox=False),
                            "component", (20, 20))
    assert rec["kind"] == "component" and rec["bbox"] == [4, 6, 12, 16]
