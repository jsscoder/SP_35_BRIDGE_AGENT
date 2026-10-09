import math
from pathlib import Path

import pytest
import yaml

from sp35 import KBError, Span, Support, assess, assess_bridge, condition_state, load_kb, priority_values
from sp35.intervals import Interval, check_partition


# ------------------------------------------------------------------ KB
def test_kb_shape(kb):
    assert len(kb.components.components) == 7
    assert len(kb.weights_sub) == 50
    assert len(kb.bridge.distresses) == 28 and len(kb.approach.distresses) == 34
    assert len(kb.mpi.factors) == 8
    assert kb.s_k == {"I": 0.0, "II": 0.25, "III": 0.5, "IV": 0.75, "V": 1.0}
    assert len(kb.version_hash) == 12


def test_kb_hash_ignores_line_endings(tmp_path):
    src = Path(__file__).resolve().parents[2] / "kb/sp35_2024"
    for f in src.glob("*.yaml"):
        (tmp_path / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert load_kb(tmp_path).version_hash == load_kb(src).version_hash


def test_kb_rejects_a_gap(tmp_path):
    src = Path(__file__).resolve().parents[2] / "kb/sp35_2024"
    for f in src.glob("*.yaml"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    text = (tmp_path / "distress_bridge.yaml").read_text(encoding="utf-8")
    broken = text.replace('III: "(0.3,1.0]"', 'III: "(0.4,1.0]"', 1)
    assert broken != text
    (tmp_path / "distress_bridge.yaml").write_text(broken, encoding="utf-8")
    with pytest.raises(KBError, match="T7.2-11"):
        load_kb(tmp_path)


def test_kb_rejects_bad_weight_total(tmp_path):
    src = Path(__file__).resolve().parents[2] / "kb/sp35_2024"
    for f in src.glob("*.yaml"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    data = yaml.safe_load((tmp_path / "components.yaml").read_text(encoding="utf-8"))
    data["components"][0]["weight"] = 13.6
    (tmp_path / "components.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(KBError, match="total"):
        load_kb(tmp_path)


def test_interval_parsing():
    assert 0.0 in Interval.parse("=0") and 0.1 not in Interval.parse("=0")
    iv = Interval.parse("(0.3,1.0]")
    assert 0.3 not in iv and 0.31 in iv and 1.0 in iv
    assert math.inf not in Interval.parse("(2,inf)")
    for bad in ("(1,0]", "[2,inf]", "1-2", "(3,3)"):
        with pytest.raises(ValueError):
            Interval.parse(bad)
    with pytest.raises(ValueError, match="overlap"):
        check_partition([Interval.parse("[0,1]"), Interval.parse("[1,2]"), Interval.parse("(2,inf)")])


# ------------------------------------------------------------ condition
@pytest.mark.parametrize("row,value,state", [
    ("T7.2-11", 0.0, "I"), ("T7.2-11", 0.3, "II"), ("T7.2-11", 0.6, "III"),
    ("T7.2-11", 1.05, "IV"), ("T7.2-11", 2.0, "IV"), ("T7.2-11", 2.01, "V"),
    ("T7.2-02", 9.9, "I"), ("T7.2-02", 10, "II"),           # abrasion: I is "<10", not Nil
    ("T7.2-16", 0.1, "III"),                                  # crushing has no state II
    ("T7.2-22", 2, "III"),                                    # loose joints: printed gap at 2
    ("T7.4-TILT", 1 / 80, "III"), ("T7.4-TILT", 1 / 50, "V"),
    ("T7.3-16", 3, "I"), ("T7.3-30", 3, "III"),              # rain cuts; shattered slab point band
])
def test_condition_states(kb, row, value, state):
    assert condition_state(kb, row, value).state == state


def test_dual_criteria_take_the_worse_state(kb):
    r = condition_state(kb, "T7.2-04", {"No.": 3, "%": 10})
    assert r.state == "III" and "(2,5]" in r.explanation
    assert condition_state(kb, "T7.2-04", {"%": 80}).state == "V"
    with pytest.raises(ValueError):
        condition_state(kb, "T7.2-04", 3)


def test_aliases_and_errors(kb):
    assert condition_state(kb, "Clogging", 3).row_id == "T7.2-25"
    assert condition_state(kb, "rusting", 6).row_id == "T7.2-10"
    with pytest.raises(KeyError, match="engineer judgement"):
        condition_state(kb, "Not visible", 1)
    with pytest.raises(ValueError):
        condition_state(kb, "T7.2-11", -0.1)
    with pytest.raises(KeyError):
        condition_state(kb, "T9.9-99", 1)


def test_explanation_quotes_the_book(kb):
    e = condition_state(kb, "T7.2-11", 0.6).explanation
    assert e.startswith("Table 7.2 T7.2-11 (Cracking (width))") and "printed: 0.3-1.0" in e


# ---------------------------------------------------------------- scores
def test_assess_validates_inputs(kb):
    with pytest.raises(ValueError, match="unknown sub-component"):
        assess(kb, {"C9.9": ["II"]})
    with pytest.raises(ValueError, match="condition state"):
        assess(kb, {"C2.8": ["VI"]})
    with pytest.raises(ValueError, match="outside scope"):
        assess(kb, {"C2.8": ["II"]}, scope=["C2.7"])


def test_nothing_noticed_is_five_star(kb):
    a = assess(kb, {})
    assert a.bhi == 100.0 and a.stars == 5 and a.critical == [] and a.condition == "Excellent"


# ------------------------------------------------------- span model (E14)
def _abut(label, **states):
    return Support(label, "abutment", states)


def _pier(label, **states):
    return Support(label, "pier", states)


def test_single_span_excludes_pier_items(kb):
    b = assess_bridge(kb, [Span("S1", _abut("A1"), _abut("A2"))])
    s = b.spans["S1"]
    assert {"C2.7", "C2.8", "C2.9"} <= set(s.excluded)
    assert "C2.1" in s.cs and "C1.4" in s.cs


def test_intermediate_span_excludes_abutment_and_approach(kb):
    b = assess_bridge(kb, [Span("S1", _abut("A1"), _pier("P1")),
                           Span("S2", _pier("P1"), _pier("P2")),
                           Span("S3", _pier("P2"), _abut("A2"))])
    mid = b.spans["S2"]
    assert not any(code.startswith(("C1.", "C2.1", "C2.2", "C2.3", "C2.4", "C2.5", "C2.6")) for code in mid.cs)
    assert "C2.8" in mid.cs and "C3.1" in mid.cs


def test_span_takes_the_worse_support(kb):
    p1 = Support("P1", "pier", {"C2.8": ["III"]})
    p2 = Support("P2", "pier", {"C2.8": ["II"]})
    b = assess_bridge(kb, [Span("S1", _abut("A1"), p1), Span("S2", p1, p2), Span("S3", p2, _abut("A2"))])
    assert b.spans["S2"].cs["C2.8"] == 50.0           # worse of 50 (III) and 75 (II)
    assert b.spans["S3"].cs["C2.8"] == 75.0


def test_bridge_lists_critical_items_once(kb):
    p1 = Support("P1", "pier", {"C7.1": ["V"]})
    b = assess_bridge(kb, [Span("S1", _abut("A1"), p1), Span("S2", p1, _abut("A2"))])
    assert b.critical == ["P1 C7.1"]
    assert b.bhi == min(s.bhi for s in b.spans.values())


def test_bridge_rejects_misplaced_observations(kb):
    with pytest.raises(ValueError, match="does not belong"):
        assess_bridge(kb, [Span("S1", Support("P1", "pier", {"C2.1": ["II"]}), _abut("A1"))])
    with pytest.raises(ValueError, match="does not belong"):
        assess_bridge(kb, [Span("S1", Support("P1", "pier", {"C1.4": ["V"]}), _abut("A1"))])
    with pytest.raises(ValueError, match="support or the whole bridge"):
        assess_bridge(kb, [Span("S1", _abut("A1"), _abut("A2"), {"C2.1": ["II"]})])
    with pytest.raises(ValueError, match="bridge-level"):
        assess_bridge(kb, [Span("S1", _abut("A1"), _abut("A2"))], waterway={"C5.1": ["II"]})


# ------------------------------------------------------------------ MPI
def test_priority_values(kb):
    pr = priority_values(kb, {"F1.1": 25.0, "F6.4": 2000, "F6.3": "national highways",
                              "F4.4": "Zone IV", "F4.3": 10, "F1.4": "Adequate"})
    assert pr == {"F1.1": 50, "F6.4": 75, "F6.3": 75, "F4.4": 75, "F4.3": 25, "F1.4": 0}
    with pytest.raises(ValueError):
        priority_values(kb, {"F4.4": "Zone IX"})
    with pytest.raises(ValueError):
        priority_values(kb, {"F1.1": "long"})
