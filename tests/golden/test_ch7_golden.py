"""Chapter 7 worked examples (kb/sp35_2024/golden_examples.yaml).

Never edit an expected value to make these pass (.agents/rules/testing-and-gates.md).
"""
import pytest

from sp35 import assess, f7_from_bhi, mpi, subcomponent_score


def test_cs_examples(kb, golden):
    for case in golden["cs_examples"]:
        got = subcomponent_score(case["distress_states"], kb.s_k)
        assert got == pytest.approx(case["expected"], abs=case["tol"] or 1e-9), case["id"]


def test_p130_book_value_is_not_reproduced(kb, golden):
    """E1: the book prints 50.0; Eq. 3 gives 75.0. Guard against 'fixing' the engine to the misprint."""
    case = next(c for c in golden["cs_examples"] if c["id"] == "p130_deck_slab")
    assert subcomponent_score(case["distress_states"], kb.s_k) != pytest.approx(case["book"])


@pytest.fixture(scope="module")
def yamuna(golden):
    return golden["bhi_examples"][0]


def test_yamuna_with_printed_cs(kb, yamuna):
    exp = yamuna["expected"]
    a = assess(kb, yamuna["states"], cs_overrides=yamuna["printed_cs_overrides"])
    assert a.bhi == pytest.approx(exp["bhi_with_printed_cs"]["value"], abs=0.005)
    assert abs(a.bhi - yamuna["book"]["bhi"]) <= exp["bhi_with_printed_cs"]["tol"]
    for comp, value in exp["ci_with_printed_cs"].items():
        assert a.ci[comp] == pytest.approx(value, abs=0.005), comp


def test_yamuna_strict_states(kb, yamuna):
    exp = yamuna["expected"]
    a = assess(kb, yamuna["states"])
    assert a.bhi == pytest.approx(exp["bhi_strict_states"]["value"], abs=0.005)
    assert abs(a.bhi - yamuna["book"]["bhi"]) <= exp["bhi_strict_states"]["tol"]
    assert a.stars == exp["stars"] == yamuna["book"]["stars"]
    assert a.critical == exp["critical_list"]  # E4: railing in state V is listed although BHI is 5-star


@pytest.mark.parametrize("idx", [0, 1])
def test_mpi_examples(kb, golden, idx):
    case = golden["mpi_examples"][idx]
    got = mpi(kb, case["pr"]).mpi
    assert got == pytest.approx(case["expected"]["value"], abs=0.005)
    assert abs(got - case["book"]) <= case["expected"]["tol"]


@pytest.mark.parametrize("idx", [0, 1])
def test_mpi_with_f7_from_star_bands(kb, golden, idx):
    """E9 default: bridge condition PR derived from BHI via the star bands."""
    case = golden["mpi_examples"][idx]
    pr = dict(case["pr"], F7=f7_from_bhi(kb, case["bhi"]))
    assert mpi(kb, pr).mpi == pytest.approx(case["reference_with_f7_from_star_bands"], abs=0.005)
