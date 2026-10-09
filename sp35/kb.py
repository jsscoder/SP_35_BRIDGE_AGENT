"""SP:35 knowledge base: load and validate kb/sp35_2024/*.yaml.

Every SP:35 number the engine uses comes from these files (AGENTS.md rule 1).
Validation runs at load time, so a bad edit fails before any score is computed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .intervals import Interval, check_partition, find

STATES: tuple[str, ...] = ("I", "II", "III", "IV", "V")
DEFAULT_KB_DIR = Path(__file__).resolve().parents[1] / "kb" / "sp35_2024"
KB_FILES = ("components.yaml", "distress_bridge.yaml", "distress_approach.yaml",
            "condition_states.yaml", "mpi.yaml")


class KBError(ValueError):
    """The knowledge base is malformed or inconsistent."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)


# ---------------------------------------------------------------- Table 7.1
class SubComponent(_Model):
    code: str
    name: str
    weight: float


class Component(_Model):
    code: str
    name: str
    weight: float
    subcomponents: List[SubComponent]
    visibility: Optional[str] = None


class SpanModel(_Model):
    decision: str
    support_types: Dict[str, List[str]]
    scope: Dict[str, str]
    not_in_span: str
    absent_element: str
    bridge_bhi: str


class ComponentsKB(_Model):
    kb_version: str
    status: str
    components: List[Component]
    span_model: SpanModel

    @model_validator(mode="after")
    def _check(self) -> "ComponentsKB":
        codes = [s.code for c in self.components for s in c.subcomponents]
        if len(codes) != len(set(codes)):
            raise KBError("duplicate sub-component codes")
        _check_total("Table 7.1 main components", (c.weight for c in self.components))
        for c in self.components:
            if c.weight <= 0 or any(s.weight <= 0 for s in c.subcomponents):
                raise KBError(f"non-positive weight in {c.code}")
            if any(not s.code.startswith(c.code + ".") for s in c.subcomponents):
                raise KBError(f"sub-component code outside {c.code}")
            _check_total(f"Table 7.1 {c.code} sub-components", (s.weight for s in c.subcomponents))
        known = set(codes)
        for kind, subs in self.span_model.support_types.items():
            if set(subs) - known:
                raise KBError(f"span_model.support_types.{kind} names unknown codes")
        if set(self.span_model.scope) != {c.code for c in self.components}:
            raise KBError("span_model.scope must list every component")
        return self


# ---------------------------------------------------------- Tables 7.2 / 7.3
def _bands(raw: Mapping[str, object]) -> Dict[str, Optional[Interval]]:
    if set(raw) != set(STATES):
        raise KBError(f"bands must have keys {STATES}, got {sorted(raw)}")
    return {s: (None if raw[s] is None else Interval.parse(raw[s])) for s in STATES}


class Criterion(_Model):
    unit: str
    bands: Dict[str, Optional[Interval]]

    @field_validator("bands", mode="before")
    @classmethod
    def _parse(cls, v):
        return _bands(v)


class DistressRow(_Model):
    id: str
    name: str
    printed: Dict[str, str]
    unit: Optional[str] = None
    bands: Optional[Dict[str, Optional[Interval]]] = None
    criteria: Optional[List[Criterion]] = None
    note: Optional[str] = None
    source: Optional[str] = None

    @field_validator("printed", mode="before")
    @classmethod
    def _printed(cls, v):
        return {str(k): str(x) for k, x in v.items()}

    @field_validator("bands", mode="before")
    @classmethod
    def _parse(cls, v):
        return None if v is None else _bands(v)

    @model_validator(mode="after")
    def _check(self) -> "DistressRow":
        if set(self.printed) != set(STATES):
            raise KBError(f"{self.id}: printed must have keys {STATES}")
        if (self.bands is None) == (self.criteria is None):
            raise KBError(f"{self.id}: exactly one of bands / criteria is required")
        if self.bands is not None and not self.unit:
            raise KBError(f"{self.id}: unit is required with bands")
        for crit in self.measures():
            try:
                check_partition(crit.bands[s] for s in STATES)
            except ValueError as e:
                raise KBError(f"{self.id} ({crit.unit}): {e}") from None
        return self

    def measures(self) -> List[Criterion]:
        if self.criteria is not None:
            return list(self.criteria)
        return [Criterion.model_construct(unit=self.unit, bands=self.bands)]


class DistressTable(_Model):
    kb_version: str
    table: str
    status: str
    s_k: Dict[str, float]
    distresses: List[DistressRow]
    aliases: Dict[str, str] = {}


# ------------------------------------------------- Table 5.1 + star rating
class ConditionState(_Model):
    id: Literal["I", "II", "III", "IV", "V"]
    name: str
    s_k: float
    summary: str


class StarBand(_Model):
    stars: int
    condition: str
    interval: Interval

    @field_validator("interval", mode="before")
    @classmethod
    def _parse(cls, v):
        return Interval.parse(v)


class StarRating(_Model):
    decision: str
    bands: List[StarBand]


class ConditionStatesKB(_Model):
    kb_version: str
    status: str
    states: List[ConditionState]
    rules: Dict[str, object]
    star_rating: StarRating

    @model_validator(mode="after")
    def _check(self) -> "ConditionStatesKB":
        if [s.id for s in self.states] != list(STATES):
            raise KBError("condition states must be listed I..V in order")
        sk = [s.s_k for s in self.states]
        if sk != sorted(sk) or sk[0] != 0.0 or sk[-1] != 1.0:
            raise KBError("s_k must rise from 0 (I) to 1 (V)")
        ordered = sorted(self.star_rating.bands, key=lambda b: b.interval.lo)
        try:
            check_partition((b.interval for b in ordered), start=0.0, end=100.0)
        except ValueError as e:
            raise KBError(f"star bands: {e}") from None
        return self


# ------------------------------------------------------- Tables 7.7 - 7.9
class SubFactor(_Model):
    code: str
    name: str
    weight: float
    classes: Dict[str, int]
    unit: Optional[str] = None

    @field_validator("classes", mode="before")
    @classmethod
    def _classes(cls, v):
        return {str(k): int(x) for k, x in v.items()}

    @model_validator(mode="after")
    def _check(self) -> "SubFactor":
        if any(pr not in (0, 25, 50, 75, 100) for pr in self.classes.values()):
            raise KBError(f"{self.code}: priority ranking values must be 0/25/50/75/100")
        numeric = [k for k in self.classes if k[:1] in "[("]
        if numeric:
            if len(numeric) != len(self.classes):
                raise KBError(f"{self.code}: mixes numeric and label classes")
            ivs = sorted((Interval.parse(k) for k in numeric), key=lambda iv: iv.lo)
            try:
                check_partition(ivs, start=0.0, closed_start=False)
            except ValueError as e:
                raise KBError(f"{self.code}: {e}") from None
        return self

    @property
    def numeric(self) -> bool:
        return next(iter(self.classes))[:1] in "[("

    def pr(self, value: object) -> int:
        """Priority ranking value for a measured number or a class label."""
        if self.numeric:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{self.code} needs a number ({self.unit}), got {value!r}")
            key = find({k: Interval.parse(k) for k in self.classes}, float(value))
            return self.classes[key]
        wanted = str(value).strip().casefold()
        for label, pr in self.classes.items():
            if label.casefold() == wanted:
                return pr
        raise ValueError(f"{self.code}: unknown class {value!r}; expected one of {list(self.classes)}")


class Factor(_Model):
    code: str
    name: str
    weight: float
    subfactors: List[SubFactor]
    from_bhi: Optional[str] = None


class MpiKB(_Model):
    kb_version: str
    status: str
    factors: List[Factor]

    @model_validator(mode="after")
    def _check(self) -> "MpiKB":
        _check_total("MPI main factors", (f.weight for f in self.factors))
        for f in self.factors:
            _check_total(f"MPI {f.code} sub-factors", (s.weight for s in f.subfactors))
        return self


def _check_total(what: str, weights) -> None:
    """SP:35 requires weights to total 100; the printed tables total 99.9-100.3."""
    total = sum(weights)
    if abs(total - 100.0) > 0.5:
        raise KBError(f"{what} total {total:.2f}, expected 100 +/- 0.5")


# -------------------------------------------------------------- aggregate
class KnowledgeBase:
    """All SP:35 tables, validated and cross-checked."""

    def __init__(self, components: ComponentsKB, bridge: DistressTable, approach: DistressTable,
                 states: ConditionStatesKB, mpi: MpiKB, version_hash: str, path: Path):
        self.components = components
        self.bridge = bridge
        self.approach = approach
        self.states = states
        self.mpi = mpi
        self.version_hash = version_hash
        self.path = path
        self.s_k: Dict[str, float] = {s.id: s.s_k for s in states.states}
        self._rows: Dict[str, DistressRow] = {}
        for table in (bridge, approach):
            if table.s_k != self.s_k:
                raise KBError(f"Table {table.table} s_k differs from condition_states.yaml")
            for row in table.distresses:
                if row.id in self._rows:
                    raise KBError(f"duplicate distress id {row.id}")
                self._rows[row.id] = row
        self.aliases: Dict[str, str] = {}
        for table in (bridge, approach):
            for alias, target in table.aliases.items():
                if target != "engineer_judgement" and target not in self._rows:
                    raise KBError(f"alias {alias!r} points to unknown row {target!r}")
                self.aliases[alias.casefold()] = target
        self.weights_main = {c.code: c.weight for c in components.components}
        self.weights_sub = {s.code: s.weight for c in components.components for s in c.subcomponents}
        self.component_of = {s.code: c.code for c in components.components for s in c.subcomponents}
        self.sub_names = {s.code: s.name for c in components.components for s in c.subcomponents}

    # -- lookups
    def row(self, key: str) -> DistressRow:
        """Distress row by id (T7.2-11) or by a Table 7.6 alias (Clogging)."""
        if key in self._rows:
            return self._rows[key]
        target = self.aliases.get(key.casefold())
        if target is None:
            raise KeyError(f"unknown distress {key!r}")
        if target == "engineer_judgement":
            raise KeyError(f"{key!r} needs an engineer judgement, not a measurement")
        return self._rows[target]

    @property
    def rows(self) -> Sequence[DistressRow]:
        return tuple(self._rows.values())

    def table_of(self, row: DistressRow) -> str:
        return self.bridge.table if row in self.bridge.distresses else self.approach.table

    def star(self, bhi: float) -> StarBand:
        if not 0.0 <= bhi <= 100.0:
            raise ValueError(f"BHI {bhi} outside 0..100")
        for band in self.states.star_rating.bands:
            if bhi in band.interval:
                return band
        raise KBError(f"no star band for {bhi}")  # unreachable after validation

    @property
    def status(self) -> Dict[str, str]:
        return {"components": self.components.status, "distress_bridge": self.bridge.status,
                "distress_approach": self.approach.status, "condition_states": self.states.status,
                "mpi": self.mpi.status}

    @property
    def is_signed(self) -> bool:
        return all(s == "signed" for s in self.status.values())


def _version_hash(kb_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(p.name for p in kb_dir.glob("*.yaml")):
        # normalise line endings so Windows and CI checkouts hash the same
        h.update(name.encode())
        h.update((kb_dir / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def load_kb(kb_dir: Optional[Path] = None) -> KnowledgeBase:
    kb_dir = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
    raw = {}
    for name in KB_FILES:
        path = kb_dir / name
        if not path.exists():
            raise KBError(f"missing {path}")
        raw[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return KnowledgeBase(
            components=ComponentsKB.model_validate(raw["components.yaml"]),
            bridge=DistressTable.model_validate(raw["distress_bridge.yaml"]),
            approach=DistressTable.model_validate(raw["distress_approach.yaml"]),
            states=ConditionStatesKB.model_validate(raw["condition_states.yaml"]),
            mpi=MpiKB.model_validate(raw["mpi.yaml"]),
            version_hash=_version_hash(kb_dir),
            path=kb_dir,
        )
    except KBError:
        raise
    except Exception as e:  # pydantic ValidationError wraps our KBError messages
        raise KBError(str(e)) from e
