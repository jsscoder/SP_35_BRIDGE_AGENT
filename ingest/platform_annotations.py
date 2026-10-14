"""Engineer annotations exported from the inspection platform -> SP:35 labels.

Each platform record has a polygon on a photo, an anomaly name, an asset name
("Pier protection works — P2", "Girders/Main beams — Span 2 (P1 to P2)") and the
engineer's condition state I-V. This module maps them onto SP:35 codes:
- asset -> sub-component code (Table 7.1 names in kb/sp35_2024/components.yaml)
  plus support (A1, P3) or span number;
- anomaly -> distress row, via the KB's Table 7.6 aliases, the KB row names, then
  kb/platform/anomaly_map.yaml; the table (7.2 / 7.3) follows the member.
Nothing is guessed: a record that does not map keeps a `problem` and is left out
of scoring. Photos are never fetched from the platform's URLs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from sp35.kb import STATES, KnowledgeBase

ANOMALY_MAP = Path(__file__).resolve().parents[1] / "kb/platform/anomaly_map.yaml"
_SUPPORT = re.compile(r"\b([AP]\d+)\b")
_SPAN = re.compile(r"\bSpan\s+(\d+)\b", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*[—–-]\s*", " - ", text or "")).strip().casefold()


@dataclass
class Label:
    annotation_id: str
    photo: Optional[str]
    image_size: Optional[Tuple[int, int]]
    polygon: List[Tuple[float, float]]
    anomaly: str
    asset: Optional[str]
    state: Optional[str]                       # engineer's condition state I-V
    sub_code: Optional[str] = None             # SP:35 sub-component, e.g. C2.9
    support: Optional[str] = None              # A1, P3 ...
    span: Optional[int] = None
    row_id: Optional[str] = None               # SP:35 distress row, e.g. T7.2-26
    problems: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems


def map_asset(name: Optional[str], kb: KnowledgeBase) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """'Pier protection works — P5' -> ('C2.9', 'P5', None). Longest sub-component name wins."""
    if not name:
        return None, None, None
    text = _norm(name)
    span = int(_SPAN.search(name).group(1)) if _SPAN.search(name) else None
    # "Span 2 (P1 to P2)" is a span member: the piers in brackets only describe it
    support = None if span is not None else (_SUPPORT.search(name) or [None, None])[1]
    if "bearing" in text and "seat" not in text and "cap" not in text:
        return "C7.1", support, span                  # "A1 Bearing 1", "Bearings — P2"
    best = None
    for code, sub in kb.sub_names.items():
        key = _norm(sub)
        if text == key or text.startswith(key + " ") or text.startswith(key + " -"):
            if best is None or len(key) > len(_norm(kb.sub_names[best])):
                best = code
    return best, support, span


def _table_7_3(sub_code: Optional[str], prefixes) -> bool:
    return bool(sub_code) and any(sub_code == p or sub_code.split(".")[0] == p for p in prefixes)


def map_anomaly(name: str, sub_code: Optional[str], kb: KnowledgeBase, amap: dict) -> Tuple[Optional[str], str]:
    """(row id or None, how it was mapped)."""
    rows = amap.get("anomalies", {})
    candidates = next((v for k, v in rows.items() if _norm(k) == _norm(name)), None)
    if candidates is None:
        alias = kb.aliases.get((name or "").casefold())
        if alias:
            candidates = [alias]
        else:
            candidates = [r.id for r in kb.rows if _norm(r.name) == _norm(name)]
    if not candidates:
        return None, "no SP:35 row for this anomaly name"
    if candidates == ["engineer_judgement"]:
        return None, "engineer judgement (not visible)"
    want = "7.3" if _table_7_3(sub_code, amap.get("table_7_3_members", [])) else "7.2"
    pick = next((c for c in candidates if c.startswith(f"T{want}-")), candidates[0])
    return pick, f"{'alias/name' if len(candidates) == 1 else 'by member table ' + want}"


def load_labels(path: Path, kb: KnowledgeBase, anomaly_map: Path = ANOMALY_MAP) -> List[Label]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw.get("data", raw) if isinstance(raw, dict) else raw
    amap = yaml.safe_load(Path(anomaly_map).read_text(encoding="utf-8"))
    labels = []
    for r in records:
        payload = json.loads(r["imagePayload"]) if r.get("imagePayload") else {}
        geo = json.loads(r["imageAnnotationCoordinates"]) if r.get("imageAnnotationCoordinates") else {}
        coords = (geo.get("geometry") or {}).get("coordinates") or [[]]
        lab = Label(annotation_id=r.get("annotationId"), photo=payload.get("name"),
                    image_size=(payload["width"], payload["height"]) if payload.get("width") else None,
                    polygon=[(float(x), float(y)) for x, y in coords[0]], anomaly=r.get("anomalyName") or "",
                    asset=r.get("assetName"), state=r.get("condition"))
        if lab.photo is None:
            lab.problems.append("no photo")
        if len(lab.polygon) < 3:
            lab.problems.append("no polygon")
        if lab.state not in STATES:
            lab.problems.append(f"no condition state ({lab.state!r})")
        lab.sub_code, lab.support, lab.span = map_asset(lab.asset, kb)
        if lab.sub_code is None:
            lab.problems.append(f"asset not mapped to a sub-component ({lab.asset!r})")
        lab.row_id, how = map_anomaly(lab.anomaly, lab.sub_code, kb, amap)
        if lab.row_id is None:
            lab.problems.append(f"anomaly not mapped ({lab.anomaly!r}: {how})")
        labels.append(lab)
    return labels
