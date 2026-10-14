"""Engineer-label import and the condition-state scorecard (CPU, synthetic data)."""
import json

import numpy as np
import pytest

from ingest.platform_annotations import load_labels, map_anomaly, map_asset
from sp35.kb import load_kb
from tools.state_scorecard import metrics, qwk, score

KB = load_kb()
AMAP = {"table_7_3_members": ["C1", "C6.1"],
        "anomalies": {"Vegetation": ["T7.2-04", "T7.3-32"], "Not Visible": ["engineer_judgement"]}}


@pytest.mark.parametrize("name,expected", [
    ("Pier protection works — P5", ("C2.9", "P5", None)),
    ("Pier — P1", ("C2.7", "P1", None)),
    ("Pier cap — P2", ("C2.8", "P2", None)),
    ("Girders/Main beams — Span 2 (P1 to P2)", ("C5.1", None, 2)),
    ("Expansion joints — Span 4 (P3 to P4)", ("C6.3", None, 4)),
    ("A1 Bearing 1", ("C7.1", "A1", None)),
    ("Bearings — P2", ("C7.1", "P2", None)),
    ("Wearing surface", ("C6.1", None, None)),
    ("Appurtenances — Span 3 (P2 to P3)", (None, None, 3)),
])
def test_assets_map_to_sp35_sub_components(name, expected):
    assert map_asset(name, KB) == expected


def test_anomalies_map_through_aliases_and_the_member_table():
    assert map_anomaly("Clogging", "C6.3", KB, AMAP)[0] == "T7.2-25"          # KB Table 7.6 alias
    assert map_anomaly("Vegetation", "C2.9", KB, AMAP)[0] == "T7.2-04"
    assert map_anomaly("Vegetation", "C6.1", KB, AMAP)[0] == "T7.3-32"        # wearing surface -> Table 7.3
    assert map_anomaly("Not Visible", "C2.4", KB, AMAP)[0] is None
    assert map_anomaly("Spalling", "C2.7", KB, AMAP)[0] == "T7.2-08"          # KB row name


def record(name="IMG_1.jpg", anomaly="Spalling", asset="Pier — P1", state="II", poly=((10, 10), (60, 10), (60, 60), (10, 60))):
    return {"annotationId": name + anomaly, "anomalyName": anomaly, "assetName": asset, "condition": state,
            "imagePayload": json.dumps({"name": name, "width": 100, "height": 100}),
            "imageAnnotationCoordinates": json.dumps({"geometry": {"type": "Polygon", "coordinates": [list(map(list, poly))]}})}


def test_unusable_records_say_why(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"data": [record(), record(state=None), record(asset="Appurtenances — Span 3"),
                                      {"anomalyName": "Joint damage"}]}), encoding="utf-8")
    labels = load_labels(f, KB)
    assert [l.usable for l in labels] == [True, False, False, False]
    assert "no condition state" in labels[1].problems[0]
    assert "asset not mapped" in labels[2].problems[0]


def test_metrics_count_abstentions_and_dangerous_under_calls():
    m = metrics([("II", "II"), ("III", "II"), ("I", "III"), ("II", None)])
    assert (m["labels"], m["answered"], m["abstained"]) == (4, 3, 1)
    assert m["exact_all"] == 0.25 and m["exact_answered"] == 0.333
    assert (m["under_calls"], m["over_calls"]) == (1, 1)
    assert m["confusion"]["II"]["none"] == 1
    assert qwk([("I", "I"), ("II", "II"), ("III", "III")]) == 1.0


def rle(mask):
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts, prev, run = [], 0, 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            run, prev = 1, v
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_scorecard_end_to_end(tmp_path):
    """One label SAM covered and rated, one SAM missed; the oracle rates both from the polygon."""
    pred = tmp_path / "pred"
    (pred / "conditions").mkdir(parents=True)
    pier = np.zeros((100, 100), bool)
    pier[:, :] = True
    spall = np.zeros((100, 100), bool)
    spall[10:60, 10:60] = True                                    # 25 % of the frame
    (pred / "IMG_1.json").write_text(json.dumps({
        "photo": "IMG_1.jpg", "status": "ok", "processed_size": [100, 100], "scale": 1.0,
        "components": [{"class": "pier", "sp35": "C2.7", "rle": rle(pier)}],
        "distresses": [{"class": "spalling", "sp35": "T7.2-08", "rle": rle(spall)}]}), encoding="utf-8")
    (pred / "conditions/IMG_1.conditions.json").write_text(json.dumps({"proposals": [
        {"proposed_state": "V", "status": "proposed_visual_estimate", "detections": [{"index": 0, "frame": "photo"}]}]}),
        encoding="utf-8")
    (pred / "IMG_2.json").write_text(json.dumps({
        "photo": "IMG_2.jpg", "status": "ok", "processed_size": [100, 100], "scale": 1.0,
        "components": [], "distresses": []}), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"data": [record(), record(name="IMG_2.jpg", anomaly="Missing", state="III"),
                                           record(name="IMG_9.jpg")]}), encoding="utf-8")
    rep = score(labels, pred)
    assert rep["scored"] == 2 and rep["skipped"] == {"photo not in the inference run": 1}
    assert rep["pipeline"]["answered"] == 1 and rep["pipeline"]["over_calls"] == 1         # V vs engineer II
    rows = {r["photo"]: r for r in rep["rows"]}
    assert rows["IMG_1.jpg"]["oracle"]["state"] == "V"            # 25 % of the pier -> KB spalling band V
    assert rows["IMG_2.jpg"]["pipeline"]["why"] == "SAM found no distress on the polygon"
    assert rows["IMG_2.jpg"]["oracle"]["state"] == "II"            # 1 missing element -> KB band "<2"
    assert rep["baseline_majority"]["state"] in ("II", "III")
