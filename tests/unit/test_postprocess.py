"""Vision post-processing rules on synthetic masks (no GPU)."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision.postprocess import (Detection, _color, load_prompt_library, mask_iou, postprocess, record,
                                render_overlay)

LIB = load_prompt_library(Path(__file__).resolve().parents[2] / "kb/prompts/sam3_prompts.yaml")
S = 4                      # boxes are written on a 100-unit grid and drawn at 4 px per unit,
H, W = 100 * S, 100 * S    # so masks clear the library's real 400 px minimum area


def box(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0 * S:y1 * S, x0 * S:x1 * S] = True
    return m


def det(cid, mask, score=0.9, prompt=None):
    c = LIB.by_id(cid)
    return Detection(cls=c, prompt=prompt or c.prompts[0], score=score, mask=mask)


def test_library_maps_every_class_to_sp35():
    """Audit A4: distresses carry their Table 7.2/7.3 row, components their C-code."""
    assert all(c.sp35_ref and c.sp35_ref.startswith("T7.") for c in LIB.of_kind("distress"))
    assert all(c.sp35_ref and c.sp35_ref.startswith("C") for c in LIB.of_kind("component"))
    assert len(LIB.prompt_list()) == 87          # 41 component + 32 distress + 14 suppression prompts
    assert LIB.by_id("spalling").sp35_ref == "T7.2-08"


def test_record_keeps_the_mapping():
    r = record(det("spalling", box(10, 10, 20, 20)))
    assert r["sp35"] == "T7.2-08" and r["bbox"] == [40, 40, 80, 80] and r["area_px"] == 1600


def test_synonyms_are_merged():
    pier = det("pier", box(0, 0, 60, 100))
    a = det("spalling", box(10, 10, 30, 30), 0.8, "spalled concrete")
    b = det("spalling", box(11, 10, 30, 30), 0.7, "concrete spalling")
    res = postprocess([pier, a, b], LIB)
    assert len(res.distresses) == 1
    assert res.distresses[0].prompts == ["spalled concrete", "concrete spalling"]
    assert any("merged" in r.reject_reason for r in res.rejected)


def test_whole_component_redetection_is_rejected():
    """Audit A5: 'spalled concrete' returning the entire girder must not count as spalling."""
    girder = det("girder", box(0, 0, 80, 40))
    spall = det("spalling", box(0, 0, 80, 38))
    res = postprocess([girder, spall], LIB)
    assert res.distresses == []
    assert "re-detects the whole girder" in res.rejected[0].reject_reason


def test_distress_outside_components_is_rejected():
    res = postprocess([det("pier", box(0, 0, 30, 100)), det("crack", box(60, 10, 90, 20))], LIB)
    assert res.distresses == [] and "inside any component" in res.rejected[0].reject_reason


def test_no_component_means_no_distress():
    res = postprocess([det("crack", box(60, 10, 90, 20))], LIB)
    assert res.distresses == [] and "no component" in res.rejected[0].reject_reason


def test_cable_suppression():
    pier = det("pier_cap", box(0, 0, 100, 60))
    cable = det("cable", box(0, 20, 100, 24))
    crack_on_cable = det("crack", box(5, 20, 95, 23))           # fully under the cable
    real_crack = det("crack", box(40, 30, 44, 55), prompt="hairline crack")
    res = postprocess([pier, cable, crack_on_cable, real_crack], LIB)
    assert [d.prompt for d in res.distresses] == ["hairline crack"]
    assert any("covered by cable" in r.reject_reason for r in res.rejected)


def test_partial_suppression_trims_the_mask():
    pier = det("pier", box(0, 0, 100, 100))
    rope = det("rope", box(0, 0, 100, 10))
    spall = det("spalling", box(20, 5, 40, 40))                 # 5 of 35 rows under the rope
    res = postprocess([pier, rope, spall], LIB)
    assert len(res.distresses) == 1
    assert res.distresses[0].area == (20 * S) * (30 * S)
    assert res.distresses[0].host["class"] == "pier" and res.distresses[0].host["sp35"] == "C2.7"


def test_cross_class_duplicate_keeps_higher_score():
    cap = det("pier_cap", box(0, 0, 100, 100))
    honey = det("honeycombing", box(10, 10, 40, 40), 0.78)
    debris = det("debris_clogging", box(10, 10, 40, 41), 0.94)
    res = postprocess([cap, honey, debris], LIB)
    assert [d.cls.id for d in res.distresses] == ["debris_clogging"]


def test_photo_tag_hosts_when_no_component_is_found():
    spall = det("spalling", box(10, 10, 40, 40))
    res = postprocess([spall], LIB, photo_host=("bearing_seat", "C7.2"))
    assert len(res.distresses) == 1
    assert res.distresses[0].host["sp35"] == "C7.2" and res.distresses[0].host["source"] == "photo tag"


def test_photo_tag_stands_in_for_an_undetected_component():
    barrier = det("crash_barrier", box(0, 80, 100, 100))
    spall = det("spalling", box(10, 10, 40, 40))                  # outside the barrier
    res = postprocess([barrier, spall], LIB, photo_host=("deck_soffit", "C5.4"))
    assert res.distresses[0].host["source"] == "photo tag (component not detected)"
    # but not when the tagged component *was* detected and the distress lies outside it
    soffit = det("deck_soffit", box(0, 60, 100, 80))
    res = postprocess([barrier, soffit, det("spalling", box(10, 10, 40, 40))], LIB, photo_host=("deck_soffit", "C5.4"))
    assert res.distresses == []


def test_photo_tag_does_not_accept_frame_filling_masks():
    res = postprocess([det("spalling", box(0, 0, 100, 95))], LIB, photo_host=("pier", "C2.7"))
    assert res.distresses == [] and "of the photo" in res.rejected[0].reject_reason


def test_suppressors_have_no_size_floor():
    thin = det("cable", box(0, 50, 100, 51))                      # 400 x 4 px
    res = postprocess([det("pier", box(0, 0, 100, 100)), thin], LIB)
    assert len(res.suppressors) == 1


def test_wearing_course_host_points_to_table_7_3():
    road = det("wearing_surface", box(0, 0, 100, 100))
    crack = det("crack", box(10, 40, 90, 44))
    res = postprocess([road, crack], LIB)
    assert "Table 7.3" in res.distresses[0].host["note"]


def test_class_threshold_applies():
    res = postprocess([det("pier", box(0, 0, 50, 50), score=0.3)], LIB)
    assert res.components == [] and "below class threshold" in res.rejected[0].reject_reason


GRAY = (128, 128, 128)


def _scene():
    """One accepted component, one accepted distress, one suppressor, one rejected distress."""
    res = postprocess([det("pier", box(0, 0, 60, 100)),
                       det("spalling", box(10, 10, 30, 30), 0.8, "spalled concrete"),
                       det("cable", box(0, 50, 100, 52)),
                       det("crack", box(70, 10, 90, 20))], LIB)
    assert res.counts() == {"components": 1, "distresses": 1, "suppressors": 1, "rejected": 1}
    return res


def test_overlay_boxes_components_and_distresses_distinctly():
    res = _scene()
    px = render_overlay(Image.new("RGB", (W, H), GRAY), res).load()
    blue = tuple(int(v) for v in _color("pier", False))
    red = tuple(int(v) for v in _color("spalling", True))
    assert blue != red
    # pier bbox [0, 0, 240, 400]: 1 px cool outline on every side, x1/y1 exclusive
    assert px[0, 200] == px[239, 200] == px[120, 399] == blue
    assert px[1, 200] != blue
    # spalling bbox [40, 40, 120, 120]: 2 px warm outline
    assert px[40, 80] == px[41, 80] == px[119, 80] == red
    assert px[42, 80] != red


def test_overlay_does_not_draw_suppressors_or_rejected():
    px = render_overlay(Image.new("RGB", (W, H), GRAY), _scene()).load()
    assert px[399, 204] == px[300, 200] == GRAY      # cable [0, 200, 400, 208], right of the pier
    assert px[280, 60] == px[300, 40] == px[359, 60] == GRAY   # rejected crack [280, 40, 360, 80]


def test_rendering_leaves_inference_json_unchanged():
    res = _scene()
    every = res.components + res.distresses + res.suppressors + res.rejected
    before = json.dumps([record(d) for d in every])
    render_overlay(Image.new("RGB", (W, H), GRAY), res)
    assert json.dumps([record(d) for d in every]) == before
    # serialization pinned from the renderer before the box fix
    assert json.dumps(record(res.components[0])) == (
        '{"class": "pier", "kind": "component", "sp35": "C2.7", "prompts": ["bridge pier"], "score": 0.9, '
        '"bbox": [0, 0, 240, 400], "area_px": 96000, "frame": "photo"}')
    assert json.dumps(record(res.distresses[0])) == (
        '{"class": "spalling", "kind": "distress", "sp35": "T7.2-08", "prompts": ["spalled concrete"], '
        '"score": 0.8, "bbox": [40, 40, 120, 120], "area_px": 6400, "frame": "photo", '
        '"host": {"class": "pier", "sp35": "C2.7", "source": "detected", "inside_fraction": 1.0, '
        '"host_coverage": 0.067}}')


@pytest.mark.parametrize("a,b,expected", [((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
                                          ((0, 0, 10, 10), (10, 10, 20, 20), 0.0),
                                          ((0, 0, 10, 10), (0, 0, 10, 5), 0.5)])
def test_mask_iou(a, b, expected):
    assert mask_iou(box(*a), box(*b)) == pytest.approx(expected)
