"""Proposed condition states from SAM 3.1 inference JSON (CPU, synthetic records)."""
import json

import numpy as np
import pytest

from sp35.condition import condition_state
from sp35.kb import load_kb
from vision.condition_proposals import decode_rle, main, propose_photo, run

KB = load_kb()


def dist(cls, row, host=("pier", "C2.7", "detected"), coverage=0.05, score=0.8, **extra):
    h = {"class": host[0], "sp35": host[1], "source": host[2], "inside_fraction": 1.0}
    if coverage is not None:
        h["host_coverage"] = coverage
    return dict({"class": cls, "kind": "distress", "sp35": row, "score": score, "bbox": [0, 0, 10, 10],
                 "area_px": 100, "frame": "photo", "host": h}, **extra)


def photo(*distresses, status="ok"):
    return {"photo": "IMG_1.jpg", "status": status, "components": [], "distresses": list(distresses)}


def only(result):
    (p,) = result["proposals"]
    return p


def test_spalling_extent_of_the_host_gives_a_proposed_state():
    p = only(propose_photo(photo(dist("spalling", "T7.2-08", coverage=0.12)), KB))
    assert p["value"] == 12.0 and p["proposed_state"] == "IV"           # Table 7.2 row 8: 11-20 % -> IV
    assert p["status"] == "proposed_visual_estimate" and "visual estimate" in p["method"]
    assert p["final_state"] is None and p["requires_engineer"] is True
    assert p["state_name"] == "Poor" and p["severity"] and "T7.2-08" in p["explanation"]


def rle(mask):
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts, prev, run_ = [], 0, 0
    for v in flat:
        if v == prev:
            run_ += 1
        else:
            counts.append(run_)
            run_, prev = 1, v
    counts.append(run_)
    return {"size": list(mask.shape), "counts": counts}


def box_mask(x0, y0, x1, y1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_compressed_coco_rle_decodes_like_pycocotools():
    expected = np.zeros((6, 7), bool)
    expected[1:4, 2:6] = True
    expected[5, 0] = True
    assert np.array_equal(decode_rle({"size": [6, 7], "counts": "5172L000005"}), expected)  # from pycocotools


def test_overlapping_masks_on_a_host_are_counted_once():
    pier = {"class": "pier", "sp35": "C2.7", "rle": rle(box_mask(0, 0, 100, 100))}
    a = dist("spalling", "T7.2-08", coverage=0.06, rle=rle(box_mask(0, 0, 6, 100)))
    b = dist("spalling", "T7.2-08", coverage=0.06, rle=rle(box_mask(2, 0, 8, 100)))    # 4 % over `a`
    res = dict(photo(a, b), components=[pier])
    p = only(propose_photo(res, KB))
    assert p["value"] == 8.0 and p["proposed_state"] == "III"          # summing would say 12 % -> IV
    assert "union of the masks" in p["method"]


def test_damage_over_most_of_the_member_needs_a_mask_check():
    pier = {"class": "pier", "sp35": "C2.7", "rle": rle(box_mask(0, 0, 100, 100))}
    pieces = [dist("spalling", "T7.2-08", coverage=0.3, rle=rle(box_mask(x, 0, x + 30, 100))) for x in (0, 30, 60)]
    p = only(propose_photo(dict(photo(*pieces), components=[pier]), KB))
    assert p["proposed_state"] is None and p["status"] == "needs_mask_check"


def test_detections_of_one_row_on_one_host_are_summed_when_masks_are_missing():
    p = only(propose_photo(photo(dist("spalling", "T7.2-08", coverage=0.03),
                                 dist("spalling", "T7.2-08", coverage=0.04)), KB))
    assert p["value"] == 7.0 and p["proposed_state"] == "III" and len(p["detections"]) == 2
    assert "overlaps may be counted twice" in p["method"]


def test_states_come_from_the_kb_engine():
    for cls, row, cov in [("honeycombing", "T7.2-06", 0.02), ("rust_stain", "T7.2-10", 0.3),
                          ("exposed_reinforcement", "T7.2-12", 0.08)]:
        p = only(propose_photo(photo(dist(cls, row, coverage=cov)), KB))
        assert p["proposed_state"] == condition_state(KB, row, round(cov * 100, 2)).state


def test_crack_width_is_never_guessed_from_a_photo():
    p = only(propose_photo(photo(dist("crack", "T7.2-11")), KB))
    assert p["proposed_state"] is None and p["status"] == "needs_site_measurement"
    assert "measure at site" in p["reason"]


def test_count_rows_use_the_count_in_the_photo():
    dets = [dist("debris_clogging", "T7.2-25", coverage=0.01) for _ in range(3)]
    p = only(propose_photo(photo(*dets), KB))
    assert p["value"] == 3 and p["proposed_state"] == condition_state(KB, "T7.2-25", 3).state
    assert "lower bound" in p["method"]


def test_orphans_and_tag_hosts_and_wearing_course_get_no_state():
    orphan = dist("spalling", "T7.2-08", host=("unassigned_structural", None, "orphan_rescue"), coverage=None)
    assert only(propose_photo(photo(orphan), KB))["status"] == "needs_host"
    tagged = dist("spalling", "T7.2-08", host=("pier", "C2.7", "photo tag"), coverage=0.15)
    assert only(propose_photo(photo(tagged), KB))["status"] == "needs_member_extent"
    road = dist("crack", "T7.2-11", host=("wearing_surface", "C6.1", "detected"))
    road["host"]["note"] = "host is wearing course / approach: rate with Table 7.3"
    assert only(propose_photo(photo(road), KB))["status"] == "needs_table_7_3_row"


def test_frame_as_member_is_opt_in_and_labelled():
    tagged = dist("spalling", "T7.2-08", host=("pier", "C2.7", "photo tag"), coverage=0.15)
    p = only(propose_photo(photo(tagged), KB, frame_as_member=True))
    assert p["proposed_state"] == "IV" and "frame taken as the tagged member" in p["method"]


def test_a_mask_covering_the_whole_member_is_not_rated():
    """IMG_4195: one 'spalling' mask covered 99.8 % of the pier mask; that is not state V."""
    p = only(propose_photo(photo(dist("spalling", "T7.2-08", coverage=0.998)), KB))
    assert p["proposed_state"] is None and p["status"] == "needs_mask_check"


def test_review_notes_and_tiles_are_carried():
    d = dist("spalling", "T7.2-08", coverage=0.02, review_required=True, review_note="shares its region with crack")
    res = photo(d)
    res["tiled"] = {"distresses": [dist("exposed_reinforcement", "T7.2-12", coverage=0.01, frame="tile")]}
    out = propose_photo(res, KB)
    assert [p["distress_class"] for p in out["proposals"]] == ["spalling", "exposed_reinforcement"]
    assert out["proposals"][0]["review_notes"] == ["shares its region with crack"]


def test_failed_inference_gives_no_proposals():
    assert propose_photo(photo(dist("spalling", "T7.2-08"), status="error"), KB)["proposals"] == []


def test_batch_and_single_inputs_write_reports(tmp_path):
    src = tmp_path / "infer"
    src.mkdir()
    (src / "IMG_1.json").write_text(json.dumps(photo(dist("spalling", "T7.2-08", coverage=0.12))), encoding="utf-8")
    second = dict(photo(dist("crack", "T7.2-11")), photo="IMG_2.jpg")
    (src / "IMG_2.json").write_text(json.dumps(second), encoding="utf-8")
    (src / "run_summary.json").write_text("{}", encoding="utf-8")
    summary = run(src, tmp_path / "out")
    assert summary["photos"] == 2 and summary["proposed_states"]["IV"] == 1
    assert summary["by_status"] == {"proposed_visual_estimate": 1, "needs_site_measurement": 1}
    assert summary["kb_signed"] is False and "measured at site" in summary["note"]
    assert (tmp_path / "out/IMG_1.conditions.json").exists() and (tmp_path / "out/conditions.csv").exists()
    assert main(["--input", str(src / "IMG_1.json"), "--outdir", str(tmp_path / "single")]) == 0
    one = json.loads((tmp_path / "single/IMG_1.conditions.json").read_text(encoding="utf-8"))
    assert one["proposals"][0]["proposed_state"] == "IV" and one["proposals"][0]["final_state"] is None


def test_empty_input_stops(tmp_path):
    with pytest.raises(SystemExit, match="no inference JSON"):
        run(tmp_path, tmp_path / "out")


# ------------------------------------------------------------ reasoning trace and review loop
def test_reasoning_trace_explains_every_stage():
    pier = {"class": "pier", "sp35": "C2.7", "score": 0.81, "rle": rle(box_mask(0, 0, 100, 100))}
    s = dist("spalling", "T7.2-08", coverage=0.15, rle=rle(box_mask(0, 0, 15, 100)),
             prompts=["spalled concrete"], area_px=1500)
    p = only(propose_photo(dict(photo(s), components=[pier]), KB))
    steps = {x["stage"]: x for x in p["reasoning"]}
    assert [x["stage"] for x in p["reasoning"]] == ["detection", "member", "measurement", "kb_band", "result"]
    assert "'spalled concrete'" in steps["detection"]["finding"] and "0.80" in steps["detection"]["finding"]
    assert "pier (C2.7)" in steps["member"]["finding"] and "0.81" in steps["member"]["finding"]
    assert "1,500 / 10,000 px = 15.00 %" in steps["measurement"]["finding"]
    assert "15 % in (10,20] -> IV" in steps["kb_band"]["finding"] and "'11-20'" in steps["kb_band"]["finding"]
    assert steps["kb_band"]["data"]["bands"]["%"]["IV"] == "(10,20]"
    assert all("if_wrong" in steps[k] for k in ("detection", "member", "measurement", "kb_band"))
    assert p["reasoning_text"].startswith("1 SAM:") and "| 4 KB:" in p["reasoning_text"]
    assert set(p["review"]) == {"review_verdict", "error_stage", "correct_distress", "correct_member",
                                "correct_value", "correct_state", "review_notes"}
    assert not any(p["review"].values())


def test_unrated_proposals_still_explain_why():
    p = only(propose_photo(photo(dist("crack", "T7.2-11")), KB))
    steps = {x["stage"]: x for x in p["reasoning"]}
    assert "measure at site" in steps["measurement"]["finding"]
    assert steps["kb_band"]["finding"].startswith("not reached")
    orphan = dist("spalling", "T7.2-08", host=("unassigned_structural", None, "orphan_rescue"), coverage=None)
    p = only(propose_photo(photo(orphan), KB))
    assert "orphan rescue" in p["reasoning"][1]["finding"]


def test_kb_band_in_the_trace_always_contains_the_value():
    """The check a reviewer does by eye ('15 % should be IV'): the band shown must hold the value."""
    for cov in (0.0001, 0.04, 0.05, 0.10, 0.105, 0.11, 0.2, 0.21, 0.5):
        p = only(propose_photo(photo(dist("spalling", "T7.2-08", coverage=cov)), KB))
        if p["status"] != "proposed_visual_estimate":
            continue
        band = p["reasoning"][3]["data"]["bands"]["%"][p["proposed_state"]]
        from sp35.intervals import Interval
        assert p["value"] in Interval.parse(band)


def test_csv_carries_reasoning_and_empty_review_columns(tmp_path):
    src = tmp_path / "infer"
    src.mkdir()
    (src / "IMG_1.json").write_text(json.dumps(photo(dist("spalling", "T7.2-08", coverage=0.12))), encoding="utf-8")
    summary = run(src, tmp_path / "out")
    import csv
    rows = list(csv.DictReader(open(tmp_path / "out/conditions.csv", encoding="utf-8-sig")))
    assert rows[0]["reasoning"].startswith("1 SAM:") and "-> IV" in rows[0]["reasoning"]
    assert rows[0]["error_stage"] == "" and rows[0]["correct_state"] == ""
    assert "wrong_distress_type" in summary["error_stages"]


def test_tally_sends_each_error_to_its_fix(tmp_path):
    import csv
    path = tmp_path / "reviewed.csv"
    cols = ["photo", "distress", "host", "error_stage", "correct_distress", "correct_member"]
    data = [["IMG_1.jpg", "spalling", "pier", "wrong_distress_type", "debris_clogging", ""],
            ["IMG_2.jpg", "spalling", "pier", "wrong_distress_type", "debris_clogging", ""],
            ["IMG_3.jpg", "spalling", "crash_barrier", "wrong_member", "", "pier_cap"],
            ["IMG_4.jpg", "spalling", "pier", "wrong_state_for_value", "", ""],
            ["IMG_5.jpg", "crack", "girder", "ok", "", ""],
            ["IMG_6.jpg", "crack", "girder", "", "", ""],
            ["IMG_7.jpg", "crack", "girder", "typo_stage", "", ""]]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(data)
    t = main(["--tally", str(path)]) == 0 and json.loads((tmp_path / "review_tally.json").read_text(encoding="utf-8"))
    assert t["reviewed"] == 6 and t["unknown_error_stages"] == ["typo_stage"]
    assert t["by_error_stage"]["wrong_distress_type"]["count"] == 2
    assert "SAM 3.1" in t["by_error_stage"]["wrong_distress_type"]["fix_in"]
    assert "KB band" in t["by_error_stage"]["wrong_state_for_value"]["fix_in"]
    assert t["distress_relabels"] == [{"sam_said": "spalling", "engineer_said": "debris_clogging", "count": 2}]
    assert t["member_relabels"] == [{"sam_said": "crash_barrier", "engineer_said": "pier_cap", "count": 1}]


def test_extent_uses_only_the_member_the_damage_is_on():
    """IMG_4190: three piers in the photo, a spall on one; rate it against that pier."""
    piers = [{"class": "pier", "sp35": "C2.7", "rle": rle(box_mask(x, 0, x + 20, 100))} for x in (0, 40, 80)]
    spall = dist("spalling", "T7.2-08", coverage=0.15, rle=rle(box_mask(0, 0, 3, 100)))     # 300 of pier 1's 2,000 px
    p = only(propose_photo(dict(photo(spall), components=piers), KB))
    assert p["value"] == 15.0 and p["proposed_state"] == "IV"           # not 5 % of all three piers
    assert "1 member mask(s) touched, of 3 in the photo" in p["reasoning"][2]["finding"]
