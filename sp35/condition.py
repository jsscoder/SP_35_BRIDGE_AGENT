"""Measured distress value -> condition state (Tables 7.2 / 7.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

from .intervals import find
from .kb import STATES, DistressRow, KnowledgeBase

Number = Union[int, float]


@dataclass(frozen=True)
class StateResult:
    state: str
    row_id: str
    explanation: str


def worse(a: str, b: str) -> str:
    return a if STATES.index(a) >= STATES.index(b) else b


def condition_state(kb: KnowledgeBase, distress: str,
                    value: Union[Number, Mapping[str, Number], Sequence[Optional[Number]]]) -> StateResult:
    """State for one measured distress.

    `distress` is a row id (``T7.2-11``) or a Table 7.6 alias (``Clogging``).
    Single-measure rows take a number. Two-measure rows (vegetation No./%,
    settlement No./mm) take a mapping by unit or a sequence in KB order; the
    worse state wins and a missing measure is skipped (at least one needed).
    """
    row: DistressRow = kb.row(distress)
    measures = row.measures()
    if row.criteria is None:
        values = [value]
    elif isinstance(value, Mapping):
        unknown = set(value) - {m.unit for m in measures}
        if unknown:
            raise ValueError(f"{row.id}: unknown unit(s) {sorted(unknown)}")
        values = [value.get(m.unit) for m in measures]
    elif isinstance(value, (list, tuple)):
        if len(value) != len(measures):
            raise ValueError(f"{row.id}: expected {len(measures)} values")
        values = list(value)
    else:
        raise ValueError(f"{row.id} has two measures {[m.unit for m in measures]}; pass a mapping")
    parts, state = [], None
    for m, v in zip(measures, values):
        if v is None:
            continue
        if v < 0:
            raise ValueError(f"{row.id}: negative value {v}")
        s = find(m.bands, float(v))
        state = s if state is None else worse(state, s)
        parts.append(f"{v:g} {m.unit} in {m.bands[s].text} -> {s}")
    if state is None:
        raise ValueError(f"{row.id}: no value given")
    table = kb.table_of(row)
    explanation = (f"Table {table} {row.id} ({row.name}): " + "; ".join(parts)
                   + f" => {state} (printed: {row.printed[state]})")
    return StateResult(state=state, row_id=row.id, explanation=explanation)
