import csv
import json

import numpy as np
from PIL import Image

from tools.bifurcate_distress import apply_decisions, prepare_queue


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


def make_fixture(tmp_path):
    predictions = tmp_path / "predictions"
    photos = tmp_path / "photos"
    output = tmp_path / "bifurcated"
    predictions.mkdir()
    photos.mkdir()
    Image.new("RGB", (8, 6), "white").save(photos / "bridge.jpg")
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1:3] = True
    (predictions / "bridge.json").write_text(json.dumps({
        "photo": "bridge.jpg",
        "status": "ok",
        "components": [{"class": "pier", "score": 0.99}],
        "distresses": [{
            "class": "spalling",
            "score": 0.8,
            "prompts": ["spalled concrete"],
            "rle": uncompressed_rle(mask),
        }],
    }), encoding="utf-8")
    return predictions, photos, output


def test_prepare_queue_includes_distresses_only_and_apply_sorts_verdicts(tmp_path):
    predictions, photos, output = make_fixture(tmp_path)

    summary = prepare_queue(predictions, photos, output)

    assert summary == {"photos_reviewed": 1, "distress_proposals": 1}
    with (output / "decisions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["distress_type"] == "spalling"
    assert "component" not in rows[0]
    assert (output / "review_queue" / "D000001.png").exists()

    rows[0].update(verdict="apt", confidence="high", rationale="Mask covers visibly broken concrete.")
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    applied = apply_decisions(output)

    assert applied["counts"] == {"apt": 1, "misclassified": 0, "uncertain": 0}
    assert (output / "apt" / "D000001.png").exists()
    assert (output / "apt" / "manifest.csv").exists()
    assert not (output / "misclassified" / "D000001.png").exists()


def test_apply_requires_a_verdict_for_every_candidate(tmp_path):
    predictions, photos, output = make_fixture(tmp_path)
    prepare_queue(predictions, photos, output)

    try:
        apply_decisions(output)
    except ValueError as exc:
        assert "missing verdict" in str(exc)
    else:
        raise AssertionError("Expected incomplete decisions to be rejected")


def test_apply_rejects_unknown_verdict(tmp_path):
    predictions, photos, output = make_fixture(tmp_path)
    prepare_queue(predictions, photos, output)
    with (output / "decisions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(verdict="true_positive", confidence="high", rationale="visual evidence")
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    try:
        apply_decisions(output)
    except ValueError as exc:
        assert "invalid verdict" in str(exc)
    else:
        raise AssertionError("Expected unknown verdict to be rejected")
