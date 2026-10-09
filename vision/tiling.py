"""Overlapping tiles for the native-resolution pass (hairline cracks, rebar)."""
from __future__ import annotations

from typing import List, Tuple


def tile_boxes(width: int, height: int, size: int = 1008, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) boxes covering the image; edge tiles are shifted inward, not shrunk."""
    if width <= 0 or height <= 0:
        raise ValueError("empty image")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(size * (1 - overlap)))

    def starts(n: int) -> List[int]:
        if n <= size:
            return [0]
        s = list(range(0, n - size + 1, stride))
        if s[-1] + size < n:
            s.append(n - size)
        return s

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]
