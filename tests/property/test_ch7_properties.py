"""Properties the scoring must keep for any input (hypothesis)."""
from hypothesis import given, settings
from hypothesis import strategies as st

from sp35 import STATES, Span, Support, assess, assess_bridge, condition_state, load_kb

KB = load_kb()
SUBS = list(KB.weights_sub)
state_lists = st.lists(st.sampled_from(STATES), max_size=4)
observations = st.dictionaries(st.sampled_from(SUBS), state_lists, max_size=12)


@settings(max_examples=200, deadline=None)
@given(observations)
def test_bhi_within_0_100(obs):
    a = assess(KB, obs)
    assert 0.0 <= a.bhi <= 100.0
    assert all(0.0 <= v <= 100.0 for v in a.cs.values())


@settings(max_examples=200, deadline=None)
@given(observations, st.sampled_from(SUBS), st.sampled_from(STATES))
def test_worse_state_never_raises_bhi(obs, sub, state):
    before = assess(KB, obs)
    worse_obs = dict(obs)
    current = list(worse_obs.get(sub, []))
    if current:
        i = 0
        new = STATES[max(STATES.index(current[i]), STATES.index(state))]
        current[i] = new
    else:
        current = [state]  # noticing a distress (state >= I) never improves on "none noticed"
    worse_obs[sub] = current
    after = assess(KB, worse_obs)
    assert after.bhi <= before.bhi + 1e-9
    assert after.cs[sub] <= before.cs[sub] + 1e-9


@settings(max_examples=150, deadline=None)
@given(observations, st.sampled_from(SUBS))
def test_na_element_never_lowers_bhi(obs, sub):
    """Bringing an absent (NA, CS=100) sub-component into scope cannot lower BHI."""
    scope = [s for s in SUBS if s != sub]
    obs = {k: v for k, v in obs.items() if k != sub}
    without = assess(KB, obs, scope=scope)
    with_na = assess(KB, obs)
    assert with_na.bhi >= without.bhi - 1e-9


@settings(max_examples=100, deadline=None)
@given(st.lists(state_lists, min_size=2, max_size=4))
def test_bridge_bhi_is_lowest_span(span_states):
    supports = [Support(f"A1", "abutment")] + [Support(f"P{i}", "pier") for i in range(1, len(span_states))] \
        + [Support("A2", "abutment")]
    spans = [Span(f"S{i + 1}", supports[i], supports[i + 1], {"C5.1": states})
             for i, states in enumerate(span_states)]
    b = assess_bridge(KB, spans)
    assert b.bhi == min(s.bhi for s in b.spans.values())
    assert b.spans[b.governing_span].bhi == b.bhi


def _sample_values(row):
    for m in row.measures():
        finite = [iv.hi for iv in m.bands.values() if iv is not None and iv.hi != float("inf")]
        top = max(finite) * 2 + 1
        yield m.unit, sorted({0.0, top, *finite, *[f * 0.999 for f in finite], *[f * 1.001 for f in finite]})


def test_every_row_is_monotonic():
    """A larger measurement never gives a better state, for all 62 rows."""
    for row in KB.rows:
        for unit, xs in _sample_values(row):
            last = 0
            for x in xs:
                value = x if row.criteria is None else {unit: x}
                idx = STATES.index(condition_state(KB, row.id, value).state)
                assert idx >= last, (row.id, unit, x)
                last = idx


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=0, max_value=100, allow_nan=False), st.floats(min_value=0, max_value=100, allow_nan=False))
def test_star_rating_monotonic(a, b):
    lo, hi = sorted((a, b))
    assert KB.star(lo).stars <= KB.star(hi).stars
