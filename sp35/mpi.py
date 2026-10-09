"""Maintenance Priority Index — SP:35 §7.7-7.10, Eq. 4-5, Tables 7.7-7.9."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .kb import KnowledgeBase, SubFactor


@dataclass(frozen=True)
class MpiResult:
    factor_ratings: Dict[str, float]
    mpi: float


def _subfactors(kb: KnowledgeBase) -> Dict[str, SubFactor]:
    return {s.code: s for f in kb.mpi.factors for s in f.subfactors}


def priority_values(kb: KnowledgeBase, inputs: Mapping[str, object]) -> Dict[str, int]:
    """Turn raw inputs (numbers or class labels, by sub-factor code) into PR values (Table 7.8)."""
    subs = _subfactors(kb)
    unknown = set(inputs) - set(subs)
    if unknown:
        raise ValueError(f"unknown sub-factor(s) {sorted(unknown)}")
    return {code: subs[code].pr(value) for code, value in inputs.items()}


def f7_from_bhi(kb: KnowledgeBase, bhi: float) -> int:
    """Bridge-condition PR from BHI via the star bands (decision E9)."""
    f7 = next(f for f in kb.mpi.factors if f.code == "F7")
    return f7.subfactors[0].classes[kb.star(bhi).condition]


def mpi(kb: KnowledgeBase, pr: Mapping[str, int]) -> MpiResult:
    """Eq. 5 per main factor, then Eq. 4. Every sub-factor needs a PR value."""
    subs = _subfactors(kb)
    missing = set(subs) - set(pr)
    unknown = set(pr) - set(subs)
    if missing or unknown:
        raise ValueError(f"PR values: missing {sorted(missing)}, unknown {sorted(unknown)}")
    bad = {c: v for c, v in pr.items() if v not in (0, 25, 50, 75, 100)}
    if bad:
        raise ValueError(f"PR values must be 0/25/50/75/100: {bad}")
    ratings: Dict[str, float] = {}
    for f in kb.mpi.factors:
        ratings[f.code] = (sum(s.weight * pr[s.code] for s in f.subfactors)
                           / sum(s.weight for s in f.subfactors))                    # Eq. 5
    total = sum(f.weight for f in kb.mpi.factors)
    value = sum(f.weight * ratings[f.code] for f in kb.mpi.factors) / total          # Eq. 4
    return MpiResult(factor_ratings=ratings, mpi=value)
