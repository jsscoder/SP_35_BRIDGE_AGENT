"""Interval notation used by the knowledge base.

`"=0"` is the single value zero (printed "Nil"); otherwise `(a,b]`, `[a,b)`,
`[a,b]`, `(a,b)`, with `inf` as an open upper bound.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

_RX = re.compile(r"^([\[(])\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?|inf)\s*([\])])$")


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    lo_closed: bool
    hi_closed: bool
    text: str

    @classmethod
    def parse(cls, text: object) -> "Interval":
        t = str(text).strip()
        if t == "=0":
            return cls(0.0, 0.0, True, True, t)
        m = _RX.match(t)
        if not m:
            raise ValueError(f"bad interval {t!r}")
        lo = float(m.group(2))
        hi = math.inf if m.group(3) == "inf" else float(m.group(3))
        if hi == math.inf and m.group(4) == "]":
            raise ValueError(f"inf cannot be a closed bound: {t!r}")
        if hi < lo or (hi == lo and not (m.group(1) == "[" and m.group(4) == "]")):
            raise ValueError(f"empty interval {t!r}")
        return cls(lo, hi, m.group(1) == "[", m.group(4) == "]", t)

    def __contains__(self, x: float) -> bool:
        above = x > self.lo or (self.lo_closed and x == self.lo)
        below = x < self.hi or (self.hi_closed and x == self.hi)
        return above and below


def check_partition(intervals: Iterable[Optional[Interval]], start: float = 0.0,
                    end: float = math.inf, closed_start: bool = True) -> None:
    """Raise ValueError unless the intervals, in the given order, cover
    [start, end) (or [start, end] when end is finite) exactly once.

    Order matters: the intervals must ascend, which is what makes state
    assignment monotonic (a larger measurement never gives a better state).
    None entries (a state that cannot occur) are skipped. With
    closed_start=False the first interval may exclude `start` (e.g. a bridge
    length class "(0,10)").
    """
    ivs = [iv for iv in intervals if iv is not None]
    if not ivs:
        raise ValueError("no intervals")
    first = ivs[0]
    if first.lo != start or (closed_start and not first.lo_closed):
        raise ValueError(f"first interval {first.text} does not start at {'a closed ' if closed_start else ''}{start}")
    for a, b in zip(ivs, ivs[1:]):
        if a.hi != b.lo:
            raise ValueError(f"gap or overlap between {a.text} and {b.text}")
        if a.hi_closed == b.lo_closed:
            kind = "overlap" if a.hi_closed else "gap"
            raise ValueError(f"{kind} at {a.hi} between {a.text} and {b.text}")
    last = ivs[-1]
    if end == math.inf:
        if last.hi != math.inf:
            raise ValueError(f"last interval {last.text} does not reach inf")
    elif not (last.hi == end and last.hi_closed):
        raise ValueError(f"last interval {last.text} does not end at a closed {end}")


def find(intervals: Mapping[str, Optional[Interval]], x: float) -> str:
    """Return the single key whose interval contains x."""
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x):
        raise ValueError(f"not a number: {x!r}")
    hits = [k for k, iv in intervals.items() if iv is not None and x in iv]
    if len(hits) != 1:
        raise ValueError(f"value {x} matches {len(hits)} bands {list(intervals)}")
    return hits[0]
