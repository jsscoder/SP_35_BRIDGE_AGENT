"""Bridge Health Index — SP:35 §7.5, Eq. 1-3, span rule (§7.5.1, decision E14), star rating.

Inputs are condition states per sub-component, one entry per distress type noticed:
``{"C2.8": ["II", "II", "II", "III"]}``. An empty list (or a missing code)
means no distress noticed or element absent (NA); both score 100.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from .kb import STATES, KnowledgeBase

States = Sequence[str]


def subcomponent_score(states: States, s_k: Mapping[str, float]) -> float:
    """Eq. 3: CS_j = sum(100 - 100*S_k) / d over the d distress types noticed."""
    states = list(states)
    if not states:
        return 100.0
    return sum(100.0 - 100.0 * s_k[s] for s in states) / len(states)


def weighted_mean(pairs: Iterable[Tuple[float, float]]) -> float:
    """Eq. 1 and Eq. 2 share this form: sum(W*x) / sum(W) over the items present."""
    pairs = list(pairs)
    total = sum(w for w, _ in pairs)
    if total <= 0:
        raise ValueError("nothing to average")
    return sum(w * x for w, x in pairs) / total


def _check_states(where: str, states_by_sub: Mapping[str, States], kb: KnowledgeBase) -> None:
    for code, states in states_by_sub.items():
        if code not in kb.weights_sub:
            raise ValueError(f"{where}: unknown sub-component {code!r}")
        bad = [s for s in states if s not in STATES]
        if bad:
            raise ValueError(f"{where} {code}: unknown condition state(s) {bad}")


@dataclass
class Assessment:
    cs: Dict[str, float]
    ci: Dict[str, float]
    bhi: float
    stars: int
    condition: str
    critical: List[str] = field(default_factory=list)
    urgent: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)


def _roll_up(kb: KnowledgeBase, cs: Mapping[str, float]) -> Tuple[Dict[str, float], float]:
    ci: Dict[str, float] = {}
    for comp in kb.components.components:
        present = [(s.weight, cs[s.code]) for s in comp.subcomponents if s.code in cs]
        if present:
            ci[comp.code] = weighted_mean(present)          # Eq. 2
    bhi = weighted_mean((kb.weights_main[c], v) for c, v in ci.items())  # Eq. 1
    return ci, bhi


def assess(kb: KnowledgeBase, states_by_sub: Mapping[str, States], *,
           scope: Optional[Iterable[str]] = None,
           cs_overrides: Optional[Mapping[str, float]] = None) -> Assessment:
    """One assessment over the sub-components in `scope` (default: all of Table 7.1).

    Used directly for a bridge assessed as a whole (the book's Table 7.6 example)
    and internally for each span by `assess_bridge`.
    """
    _check_states("observations", states_by_sub, kb)
    in_scope = list(kb.weights_sub) if scope is None else list(scope)
    outside = set(states_by_sub) - set(in_scope)
    if outside:
        raise ValueError(f"observations for sub-components outside scope: {sorted(outside)}")
    cs = {code: subcomponent_score(states_by_sub.get(code, ()), kb.s_k) for code in in_scope}
    for code, value in (cs_overrides or {}).items():
        if code not in cs:
            raise ValueError(f"override for {code} outside scope")
        cs[code] = float(value)
    ci, bhi = _roll_up(kb, cs)
    band = kb.star(bhi)
    return Assessment(
        cs=cs, ci=ci, bhi=bhi, stars=band.stars, condition=band.condition,
        critical=sorted(c for c in in_scope if "V" in states_by_sub.get(c, ())),
        urgent=sorted(c for c in in_scope if "IV" in states_by_sub.get(c, ())),
        excluded=sorted(set(kb.weights_sub) - set(in_scope)),
    )


# ------------------------------------------------------------- span model
@dataclass
class Support:
    """An abutment or pier. Holds its own observations: substructure (C2) for its
    kind, foundations (C4), bearings (C7) and, for an abutment, the approach (C1)
    at that end."""
    label: str
    kind: Literal["abutment", "pier"]
    states: Mapping[str, States] = field(default_factory=dict)


@dataclass
class Span:
    """A span between two supports, with its own superstructure (C5) and
    appurtenance (C6) observations."""
    label: str
    left: Support
    right: Support
    states: Mapping[str, States] = field(default_factory=dict)


@dataclass
class BridgeAssessment:
    spans: Dict[str, Assessment]
    bhi: float
    governing_span: str
    stars: int
    condition: str
    critical: List[str]
    urgent: List[str]
    kb_version: str


def _relevant(kb: KnowledgeBase, sub: str, support: Support) -> bool:
    model = kb.components.span_model
    comp = kb.component_of[sub]
    scope = model.scope[comp]
    if scope == "end_spans_only":
        return support.kind == "abutment"
    if scope != "per_support":
        return False
    typed = {code for subs in model.support_types.values() for code in subs}
    if sub in typed:
        return sub in model.support_types.get(support.kind, [])
    return True  # e.g. foundations and bearings apply to every support


def assess_bridge(kb: KnowledgeBase, spans: Sequence[Span],
                  waterway: Optional[Mapping[str, States]] = None) -> BridgeAssessment:
    """BHI per span and for the bridge (= lowest span), per SP:35 §7.5.1 and decision E14."""
    if not spans:
        raise ValueError("a bridge needs at least one span")
    if len({s.label for s in spans}) != len(spans):
        raise ValueError("span labels must be unique")
    model = kb.components.span_model
    waterway = dict(waterway or {})
    _check_states("waterway", waterway, kb)
    for code in waterway:
        if model.scope[kb.component_of[code]] != "bridge_level":
            raise ValueError(f"waterway observations may only use bridge-level codes, got {code}")

    results: Dict[str, Assessment] = {}
    critical, urgent = set(), set()
    for span in spans:
        _check_states(span.label, span.states, kb)
        for code in span.states:
            if model.scope[kb.component_of[code]] != "per_span":
                raise ValueError(f"{span.label}: {code} belongs to a support or the whole bridge")
        for sup in (span.left, span.right):
            _check_states(sup.label, sup.states, kb)
            for code in sup.states:
                if not _relevant(kb, code, sup):
                    raise ValueError(f"{sup.label} ({sup.kind}): {code} does not belong to this support")

        cs: Dict[str, float] = {}
        for code in kb.weights_sub:
            scope = model.scope[kb.component_of[code]]
            if scope == "per_span":
                cs[code] = subcomponent_score(span.states.get(code, ()), kb.s_k)
            elif scope == "bridge_level":
                cs[code] = subcomponent_score(waterway.get(code, ()), kb.s_k)
            else:  # per_support or end_spans_only: worse of the supports it applies to
                vals = [subcomponent_score(sup.states.get(code, ()), kb.s_k)
                        for sup in (span.left, span.right) if _relevant(kb, code, sup)]
                if vals:
                    cs[code] = min(vals)
                # else: not part of this span -> excluded (model.not_in_span)
        ci, bhi = _roll_up(kb, cs)
        band = kb.star(bhi)
        span_crit, span_urg = [], []
        for code, states in span.states.items():
            if "V" in states:
                span_crit.append(f"{span.label} {code}")
            elif "IV" in states:
                span_urg.append(f"{span.label} {code}")
        for sup in (span.left, span.right):
            for code, states in sup.states.items():
                if "V" in states:
                    span_crit.append(f"{sup.label} {code}")
                elif "IV" in states:
                    span_urg.append(f"{sup.label} {code}")
        for code, states in waterway.items():
            if "V" in states:
                span_crit.append(f"bridge {code}")
            elif "IV" in states:
                span_urg.append(f"bridge {code}")
        results[span.label] = Assessment(
            cs=cs, ci=ci, bhi=bhi, stars=band.stars, condition=band.condition,
            critical=sorted(set(span_crit)), urgent=sorted(set(span_urg)),
            excluded=sorted(set(kb.weights_sub) - set(cs)))
        critical.update(span_crit)
        urgent.update(span_urg)

    governing = min(results, key=lambda k: results[k].bhi)
    bhi = results[governing].bhi
    band = kb.star(bhi)
    return BridgeAssessment(spans=results, bhi=bhi, governing_span=governing, stars=band.stars,
                            condition=band.condition, critical=sorted(critical),
                            urgent=sorted(urgent - critical), kb_version=kb.version_hash)
