"""Turn raw SAM 3.1 masks into reviewable detections. Pure numpy (no torch),
so every rule here is unit-tested on CPU.

Rules (vision-gpu.md, audit A4/A5):
- components keep their SP:35 sub-component code (`sp35`), distresses their
  Table 7.2/7.3 row (`sp35_row`);
- synonyms of one class that find the same region are merged;
- suppression masks (cables, ropes, posters, people, vehicles + per-class extras)
  are subtracted from distresses; a distress mostly covered by them is rejected;
- two component classes on the same region keep one: a clear score gap
  decides, else the photo's tag, else `component_group_rank`;
- a distress must lie inside a component, and one that re-detects a whole
  component (IoU above `redetect_iou`) is rejected; a confident distress of a
  class with `orphan_rescue_min` whose host SAM missed is kept as
  `unassigned_structural` with `review_required`;
- two different distress classes on the same region keep the higher score,
  except `co_occur` pairs, which are both kept (the lower one for review).
Rejected detections are kept with a reason, for audit.
"""
from __future__ import annotations

import colorsys
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

GLOBAL_SUPPRESSORS = ("cable", "rope", "poster", "person", "vehicle")
# Hosts rated with Table 7.3 (approaches / wearing course) rather than Table 7.2.
WEARING_COURSE_HOSTS = ("C6.1", "C1.5")
# Host of a rescued distress whose component SAM did not find: no SP:35 code until the engineer assigns one.
ORPHAN_HOST = "unassigned_structural"
_COMPONENT_REF = re.compile(r"^C\d\.\d+$")
_DISTRESS_REF = re.compile(r"^T7\.[234]-[A-Z0-9]+$")
_GROUP_REF = re.compile(r"^C\d$")


@dataclass(frozen=True)
class PromptClass:
    id: str
    kind: str                      # component | distress | suppression
    prompts: Tuple[str, ...]
    sp35_ref: Optional[str]
    threshold: float
    min_area_px: int
    tile: bool = False
    suppress_with: Tuple[str, ...] = ()
    orphan_rescue_min: Optional[float] = None   # distresses only


@dataclass(frozen=True)
class Rules:
    merge_iou: float = 0.6
    host_inside_min: float = 0.5
    redetect_iou: float = 0.8
    suppressed_max: float = 0.5
    cross_class_iou: float = 0.9
    component_conflict_iou: float = 0.5
    component_conflict_margin: float = 0.15


@dataclass(frozen=True)
class PromptLibrary:
    version: str
    classes: Tuple[PromptClass, ...]
    rules: Rules
    sha12: str
    group_rank: Mapping[str, int] = field(default_factory=dict, compare=False)
    co_occur: FrozenSet[FrozenSet[str]] = frozenset()

    def of_kind(self, kind: str) -> List[PromptClass]:
        return [c for c in self.classes if c.kind == kind]

    def by_id(self, cid: str) -> PromptClass:
        return next(c for c in self.classes if c.id == cid)

    @property
    def min_threshold(self) -> float:
        return min(c.threshold for c in self.classes)

    def prompt_list(self, kinds: Sequence[str] = ("component", "distress", "suppression"),
                    tile_only: bool = False) -> List[Tuple[PromptClass, str]]:
        return [(c, p) for c in self.classes if c.kind in kinds and (c.tile or not tile_only)
                for p in c.prompts]


def load_prompt_library(path: Path) -> PromptLibrary:
    raw_bytes = Path(path).read_bytes()
    y = yaml.safe_load(raw_bytes.decode("utf-8"))
    d = y.get("defaults", {})
    thr, min_area = float(d.get("score_threshold", 0.5)), int(d.get("min_area_px", 0))
    rules = Rules(**{k: float(d[k]) for k in Rules.__dataclass_fields__ if k in d})

    def make(c: dict, kind: str, ref: Optional[str]) -> PromptClass:
        rescue = c.get("orphan_rescue_min")
        if rescue is not None and kind != "distress":
            raise ValueError(f"{c['id']}: orphan_rescue_min applies to distresses only")
        return PromptClass(id=c["id"], kind=kind, prompts=tuple(c["prompts"]), sp35_ref=ref,
                           threshold=float(c.get("score_threshold", thr)),
                           min_area_px=int(c.get("min_area_px", min_area)),
                           tile=bool(c.get("tile", False)),
                           suppress_with=tuple(c.get("suppress_with", ())),
                           orphan_rescue_min=None if rescue is None else float(rescue))

    classes: List[PromptClass] = []
    for c in y.get("components", []):
        classes.append(make(c, "component", c.get("sp35")))
    for c in y.get("distresses", []):
        classes.append(make(c, "distress", c.get("sp35_row")))       # audit A4: distresses use sp35_row
    for name, prompts in (y.get("suppression") or {}).items():
        # thin cables and ropes are small by nature; a size floor would discard exactly what we need
        classes.append(make({"id": name, "prompts": prompts, "min_area_px": 0}, "suppression", None))

    ids = [c.id for c in classes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate class ids in prompt library")
    names = {c.id for c in classes if c.kind == "suppression"}
    for c in classes:
        if c.kind == "component" and not (c.sp35_ref and _COMPONENT_REF.match(c.sp35_ref)):
            raise ValueError(f"component {c.id}: sp35 code missing or malformed ({c.sp35_ref!r})")
        if c.kind == "distress" and not (c.sp35_ref and _DISTRESS_REF.match(c.sp35_ref)):
            raise ValueError(f"distress {c.id}: sp35_row missing or malformed ({c.sp35_ref!r})")
        if set(c.suppress_with) - names:
            raise ValueError(f"{c.id}: unknown suppressors {sorted(set(c.suppress_with) - names)}")
        if not c.prompts:
            raise ValueError(f"{c.id}: no prompts")
    missing = set(GLOBAL_SUPPRESSORS) - names
    if missing:
        raise ValueError(f"suppression list lacks {sorted(missing)}")
    group_rank = {str(k): int(v) for k, v in (y.get("component_group_rank") or {}).items()}
    bad = [k for k in group_rank if not _GROUP_REF.match(k)]
    if bad:
        raise ValueError(f"component_group_rank keys must be SP:35 groups like C2: {bad}")
    distress_ids = {c.id for c in classes if c.kind == "distress"}
    co_occur = set()
    for pair in y.get("co_occur") or []:
        if len(pair) != 2 or len(set(pair)) != 2 or set(pair) - distress_ids:
            raise ValueError(f"co_occur pair must name two different distress classes: {pair}")
        co_occur.add(frozenset(pair))
    sha = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()[:12]
    return PromptLibrary(version=str(y.get("version", "?")), classes=tuple(classes), rules=rules, sha12=sha,
                         group_rank=group_rank, co_occur=frozenset(co_occur))


# ------------------------------------------------------------------ detections
@dataclass
class Detection:
    cls: PromptClass
    prompt: str
    score: float
    mask: np.ndarray                      # bool, H x W (processed-image coordinates)
    prompts: List[str] = field(default_factory=list)
    host: Optional[dict] = None
    reject_reason: Optional[str] = None
    frame: str = "photo"                  # "photo" or "tile"
    review_required: bool = False         # kept, but the engineer must look (orphan, shared region)
    review_note: Optional[str] = None

    def flag(self, note: str) -> None:
        self.review_required = True
        self.review_note = note if self.review_note is None else f"{self.review_note}; {note}"

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        self._bbox: Optional[List[int]] = None
        if not self.prompts:
            self.prompts = [self.prompt]

    def set_mask(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self._bbox = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bbox(self) -> List[int]:
        if self._bbox is None:
            rows, cols = np.any(self.mask, axis=1), np.any(self.mask, axis=0)
            if not rows.any():
                self._bbox = [0, 0, 0, 0]
            else:
                y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
                x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
                self._bbox = [int(x0), int(y0), int(x1), int(y1)]
        return self._bbox


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def det_iou(a: "Detection", b: "Detection") -> float:
    """IoU computed only inside the union of the two boxes (0 when boxes are disjoint)."""
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return 0.0
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    return mask_iou(a.mask[y0:y1, x0:x1], b.mask[y0:y1, x0:x1])


def _frac_inside(inner: np.ndarray, outer: np.ndarray) -> float:
    area = inner.sum()
    return float(np.logical_and(inner, outer).sum() / area) if area else 0.0


def merge_synonyms(dets: Sequence[Detection], iou: float) -> Tuple[List[Detection], List[Detection]]:
    """Within each class keep the highest-scoring mask of a region; fold the rest into it."""
    kept: List[Detection] = []
    merged: List[Detection] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        twin = next((k for k in kept if k.cls.id == det.cls.id and det_iou(k, det) >= iou), None)
        if twin is None:
            kept.append(det)
        else:
            if det.prompt not in twin.prompts:
                twin.prompts.append(det.prompt)
            det.reject_reason = f"merged into same-class detection ({twin.prompt!r})"
            merged.append(det)
    return kept, merged


def _group(code: Optional[str]) -> str:
    return (code or "").split(".")[0]


def _conflict_winner(a: Detection, b: Detection, lib: PromptLibrary,
                     photo_host: Optional[Tuple[str, str]]) -> Tuple[Detection, str]:
    """`a` scores at least as high as `b`; both are components on the same region.
    The photo's tag decides first: in the pilot run "concrete crash barrier" outscored
    the real pier and bearing seat on pier-tagged IMG_4195 by more than the margin."""
    if photo_host is not None:
        tag = photo_host[1]
        for matches, why in ((lambda c: c.cls.sp35_ref == tag, "photo tag"),
                             (lambda c: _group(c.cls.sp35_ref) == _group(tag), "photo tag group")):
            if matches(a) != matches(b):
                return (a if matches(a) else b), why
    if a.score - b.score > lib.rules.component_conflict_margin:
        return a, "higher score"
    last = max(lib.group_rank.values(), default=0) + 1
    ra, rb = (lib.group_rank.get(_group(c.cls.sp35_ref), last) for c in (a, b))
    if ra != rb:
        return (a if ra < rb else b), "structural priority"
    return a, "higher score"


def resolve_components(components: Sequence[Detection], lib: PromptLibrary,
                       photo_host: Optional[Tuple[str, str]] = None) -> Tuple[List[Detection], List[Detection]]:
    """Two different component classes on one region (mask IoU >= component_conflict_iou)
    are one member: keep one, so a phantom crash barrier cannot claim a pier cap."""
    kept = sorted(components, key=lambda d: d.score, reverse=True)
    lost: List[Detection] = []
    while True:
        conflict = None
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                if a.cls.id != b.cls.id:
                    iou = det_iou(a, b)
                    if iou >= lib.rules.component_conflict_iou:
                        conflict = (a, b, iou)
                        break
            if conflict:
                break
        if conflict is None:
            return kept, lost
        a, b, iou = conflict
        winner, why = _conflict_winner(a, b, lib, photo_host)
        loser = b if winner is a else a
        loser.reject_reason = f"same region as {winner.cls.id} (IoU {iou:.2f}; component conflict, {why})"
        kept = [k for k in kept if k is not loser]     # by identity: == would compare the numpy masks
        lost.append(loser)


def _rescue_orphan(d: Detection, rules: Rules, why: str, inside: float) -> bool:
    """Keep a confident distress whose host SAM missed, for the engineer to place,
    instead of dropping it. A mask filling the frame is the member, not a defect."""
    floor = d.cls.orphan_rescue_min
    if floor is None or d.score < floor or d.area / d.mask.size > rules.redetect_iou:
        return False
    d.host = {"class": ORPHAN_HOST, "sp35": None, "source": "orphan_rescue",
              "inside_fraction": round(float(inside), 3)}
    d.flag(f"{why}: engineer assigns the component")
    return True


@dataclass
class PostResult:
    components: List[Detection]
    distresses: List[Detection]
    suppressors: List[Detection]
    rejected: List[Detection]

    def counts(self) -> Dict[str, int]:
        return {"components": len(self.components), "distresses": len(self.distresses),
                "suppressors": len(self.suppressors), "rejected": len(self.rejected)}


def postprocess(raw: Sequence[Detection], lib: PromptLibrary,
                photo_host: Optional[Tuple[str, str]] = None) -> PostResult:
    """`photo_host` = (label, SP:35 code) from the photo's tag (tags.csv). When SAM
    finds no component in the frame (typical for close-ups), the whole photo is
    the host, so distresses are still proposed against the tagged component."""
    rules = lib.rules
    rejected: List[Detection] = []
    passing: List[Detection] = []
    for d in raw:
        if d.score < d.cls.threshold:
            d.reject_reason = f"score {d.score:.2f} below class threshold {d.cls.threshold:.2f}"
            rejected.append(d)
        elif d.area < d.cls.min_area_px:
            d.reject_reason = f"area {d.area} px below minimum {d.cls.min_area_px}"
            rejected.append(d)
        else:
            passing.append(d)
    kept, merged = merge_synonyms(passing, rules.merge_iou)
    rejected += merged

    components, lost = resolve_components([d for d in kept if d.cls.kind == "component"], lib, photo_host)
    rejected += lost
    suppressors = [d for d in kept if d.cls.kind == "suppression"]
    distresses = [d for d in kept if d.cls.kind == "distress"]

    sup_masks: Dict[str, np.ndarray] = {}
    for s in suppressors:
        sup_masks[s.cls.id] = s.mask if s.cls.id not in sup_masks else (sup_masks[s.cls.id] | s.mask)

    surviving: List[Detection] = []
    comp_union = None
    if components:
        comp_union = np.logical_or.reduce([c.mask for c in components])
    for d in distresses:
        names = [n for n in dict.fromkeys(GLOBAL_SUPPRESSORS + d.cls.suppress_with) if n in sup_masks]
        if names:
            sup = np.logical_or.reduce([sup_masks[n] for n in names])
            covered = _frac_inside(d.mask, sup)
            if covered > rules.suppressed_max:
                d.reject_reason = f"{covered:.0%} covered by {', '.join(names)}"
                rejected.append(d)
                continue
            d.set_mask(d.mask & ~sup)
            if d.area < d.cls.min_area_px:
                d.reject_reason = f"area {d.area} px below minimum after removing {', '.join(names)}"
                rejected.append(d)
                continue
        if comp_union is None:
            if photo_host is None:
                if _rescue_orphan(d, rules, "no component found and the photo has no component tag", 0.0):
                    surviving.append(d)
                else:
                    d.reject_reason = "no component found to host it (and the photo has no component tag)"
                    rejected.append(d)
            elif _host_by_tag(d, photo_host, rules, "photo tag"):
                surviving.append(d)
            else:
                rejected.append(d)
            continue
        inside = _frac_inside(d.mask, comp_union)
        if inside < rules.host_inside_min:
            tagged_missing = photo_host is not None and not any(c.cls.sp35_ref == photo_host[1] for c in components)
            if not tagged_missing:
                if _rescue_orphan(d, rules, f"only {inside:.0%} inside any detected component", inside):
                    surviving.append(d)
                else:
                    d.reject_reason = f"only {inside:.0%} inside any component"
                    rejected.append(d)
                continue
            # the tagged component was not detected: the photo frame stands in for it
            if _host_by_tag(d, photo_host, rules, "photo tag (component not detected)"):
                surviving.append(d)
            else:
                rejected.append(d)
            continue
        ious = [(det_iou(d, c), c) for c in components]
        best_iou, best = max(ious, key=lambda t: t[0])
        # (a distress re-detecting a whole component is rejected even if the photo is tagged)
        if best_iou > rules.redetect_iou:
            d.reject_reason = f"re-detects the whole {best.cls.id} (IoU {best_iou:.2f})"
            rejected.append(d)
            continue
        host = max(components, key=lambda c: np.logical_and(d.mask, c.mask).sum())
        inter = np.logical_and(d.mask, host.mask).sum()
        d.host = {"class": host.cls.id, "sp35": host.cls.sp35_ref, "source": "detected",
                  "inside_fraction": round(float(inter / max(d.area, 1)), 3),
                  "host_coverage": round(float(inter / max(host.area, 1)), 3)}
        _table_note(d)
        surviving.append(d)

    final: List[Detection] = []
    for d in sorted(surviving, key=lambda x: x.score, reverse=True):
        same = [k for k in final if k.cls.id != d.cls.id and det_iou(k, d) >= rules.cross_class_iou]
        blocking = [k for k in same if frozenset((k.cls.id, d.cls.id)) not in lib.co_occur]
        if blocking:
            d.reject_reason = f"same region as {blocking[0].cls.id} (higher score)"
            rejected.append(d)
            continue
        if same:
            d.flag(f"shares its region with {', '.join(k.cls.id for k in same)} (co-occurring distresses)")
        final.append(d)
    return PostResult(components=components, distresses=final, suppressors=suppressors, rejected=rejected)


def _host_by_tag(d: Detection, photo_host: Tuple[str, str], rules: Rules, source: str) -> bool:
    """Use the photo's tagged component as host. A 'distress' covering most of the
    frame is the tagged component itself, not a defect on it."""
    share = d.area / d.mask.size
    if share > rules.redetect_iou:
        d.reject_reason = f"covers {share:.0%} of the photo (the tagged {photo_host[0]} itself)"
        return False
    d.host = {"class": photo_host[0], "sp35": photo_host[1], "source": source,
              "inside_fraction": 1.0, "host_coverage": round(share, 3)}
    _table_note(d)
    return True


def _table_note(d: Detection) -> None:
    """SP:35 rates wearing course and approaches with Table 7.3, not Table 7.2."""
    if d.host and d.host.get("sp35") in WEARING_COURSE_HOSTS and (d.cls.sp35_ref or "").startswith("T7.2-"):
        d.host["note"] = ("host is wearing course / approach: rate with Table 7.3 "
                          "(e.g. T7.3-02 longitudinal or T7.3-03 transverse cracking) - reviewer picks the row")


# --------------------------------------------------------------------- output
def record(d: Detection, rle: Optional[dict] = None) -> dict:
    out = {"class": d.cls.id, "kind": d.cls.kind, "sp35": d.cls.sp35_ref, "prompts": list(d.prompts),
           "score": round(float(d.score), 3), "bbox": d.bbox(), "area_px": d.area, "frame": d.frame}
    if d.host is not None:
        out["host"] = d.host
    if d.reject_reason is not None:
        out["reject_reason"] = d.reject_reason
    if d.review_required:
        out["review_required"] = True
        out["review_note"] = d.review_note
    if rle is not None:
        out["rle"] = rle
    return out


def _color(key: str, warm: bool) -> np.ndarray:
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    hue = (0.95 + 0.17 * h) % 1.0 if warm else 0.45 + 0.25 * h   # reds/oranges vs teals/blues
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return np.array([r, g, b]) * 255


def render_overlay(image, result: PostResult):
    """Components as a light wash with a thin cool box (label: black on the box colour),
    distresses as strong fills with a thicker warm box (label: white on black).
    Suppressors and rejected detections are not drawn."""
    from PIL import Image, ImageDraw

    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    for det, alpha, warm in [(c, 0.18, False) for c in result.components] + \
                            [(d, 0.55, True) for d in result.distresses]:
        col = _color(det.cls.id, warm)
        arr[det.mask] = arr[det.mask] * (1 - alpha) + col * alpha
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    boxes = [(c, False, 1, f"{c.cls.id} {c.cls.sp35_ref}") for c in result.components] + \
            [(d, True, 2, f"{d.cls.id} {d.cls.sp35_ref} {d.score:.2f}" + (" REVIEW" if d.review_required else ""))
             for d in result.distresses]
    for det, warm, width, label in boxes:          # distresses last, so they sit on top
        x0, y0, x1, y1 = det.bbox()               # x1, y1 exclusive
        if x1 <= x0 or y1 <= y0:
            continue
        col = tuple(int(v) for v in _color(det.cls.id, warm))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=width)
        ty = y0 - 12 if y0 >= 12 else y0          # inside the box when it touches the top edge
        draw.rectangle([x0, ty, x0 + 7 * len(label), ty + 12], fill=(0, 0, 0) if warm else col)
        draw.text((x0 + 2, ty), label, fill=(255, 255, 255) if warm else (0, 0, 0))
    return out
