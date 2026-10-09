"""Build the Phase 0 pilot set from the demo photos (plan task 0.6, audit A9).

Every input file gets a category and, if excluded, a reason:
  HEIC-conversion twin -> exact duplicate -> thermal frame (DJI XMP) ->
  screenshot / Street View capture -> near-duplicate (16x16 dHash, Hamming <= 20).
Writes data/pilot/manifest.csv, tags.csv (span/component left for the engineer
in WP1.3; probe photos carry a visual hint), streetview.csv, and fills
data/pilot/probe10 with a representative probe set. An existing probe folder
is moved to data/pilot/_superseded/, never deleted.

    uv run --group ingest python -m ingest.build_pilot --src "../yugmi_images"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import ExifTags, Image, ImageOps

# Manual decisions that cannot be read from the files themselves (reviewed by eye, 7-8 Oct 2026).
MANUAL = {
    "Screenshot_2026-02-27_201207.png": ("unrelated", "Windows 'Reset this PC' dialog, not a bridge"),
    "14930adc-d817-4c24-8136-045d9616902f.jpg": ("streetview", "Google Street View capture of the abutment area"),
}
# Representative probe set: both resolutions and the main component types.
# (SP:35 sub-component hint, what the photo shows) — visual hints, not engineer-confirmed.
PROBE_SET = {
    "IMG_4231.jpg": ("C2.8", "pier cap top: projecting rebar, open gap or crack above (12 MP)"),
    "IMG_4227.jpg": ("C7.2", "bearing pedestal: broken edges, rope across frame (12 MP)"),
    "IMG_4297.jpg": ("C6.1", "deck: expansion joint, sealed cracks, crash barrier (12 MP)"),
    "IMG_4241.jpg": ("C5.1", "PSC girder soffit with diaphragms (12 MP)"),
    "IMG_4290.jpg": ("C2.7", "circular pier with posters (12 MP)"),
    "IMG_4160.jpg": ("C7.2", "girder seat: displaced plate, rubble (24 MP)"),
    "IMG_4238.jpg": ("C2.8", "pier cap top with rebar and crack line (24 MP)"),
    "IMG_4168.jpg": ("C1.3", "approach retaining wall with posters (24 MP)"),
    "IMG_4195.jpg": ("C2.7", "pier base with debris heap (24 MP)"),
    "IMG_4185.jpg": ("C5.4", "deck soffit between girders (24 MP)"),
}
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
XMP_SOURCE = re.compile(rb'drone-dji:ImageSource="([^"]+)"')


@dataclass
class Item:
    path: Path
    sha256: str
    width: int
    height: int
    make: str
    model: str
    taken: str
    gps: bool
    xmp_source: str
    dhash: Optional[int]
    category: str = ""
    keep: bool = True
    reason: str = ""
    duplicate_of: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


def dhash(img: Image.Image, size: int = 16) -> int:
    """Difference hash on a (size+1) x size grayscale thumbnail (size*size bits)."""
    g = ImageOps.exif_transpose(img).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (row[c] > row[c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_item(path: Path) -> Item:
    data = path.read_bytes()
    make = model = taken = ""
    gps, w, h, dh = False, 0, 0, None
    try:
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
            exif = im.getexif()
            make = str(exif.get(TAGS["Make"], "") or "").strip()
            model = str(exif.get(TAGS["Model"], "") or "").strip()
            sub = exif.get_ifd(0x8769)
            taken = str(sub.get(36867) or exif.get(TAGS["DateTime"], "") or "")
            gps = bool(exif.get_ifd(0x8825))
            dh = dhash(im)
    except Exception:  # unreadable image: keep the row, mark it below
        pass
    m = XMP_SOURCE.search(data[:512 * 1024])
    return Item(path=path, sha256=hashlib.sha256(data).hexdigest(), width=w, height=h, make=make,
                model=model, taken=taken, gps=gps, xmp_source=m.group(1).decode() if m else "", dhash=dh)


def classify(items: List[Item], near_dup_bits: int = 20) -> None:
    names = {it.path.name for it in items}
    by_sha: Dict[str, Item] = {}
    for it in sorted(items, key=lambda i: i.path.name):
        n = it.path.name
        if ".HEIC." in n.upper():
            original = re.split(r"\.HEIC\.", n, flags=re.I)[0].rstrip("_") + Path(n).suffix
            if original in names:
                it.category, it.keep, it.reason, it.duplicate_of = "iphone", False, "HEIC-conversion twin", original
                continue
        if it.sha256 in by_sha:
            first = by_sha[it.sha256]
            it.category, it.keep, it.reason, it.duplicate_of = first.category or "iphone", False, "exact duplicate", first.path.name
            continue
        by_sha[it.sha256] = it
        if n in MANUAL:
            it.category, why = MANUAL[n]
            it.keep, it.reason = False, why
        elif it.xmp_source == "InfraredCamera":
            it.category, it.keep, it.reason = "thermal", False, "DJI thermal frame of a non-bridge site"
        elif it.path.suffix.lower() == ".png" and n.startswith("Screenshot_"):
            it.category, it.keep, it.reason = "streetview", False, "Street View capture of the approach walls (separate set)"
        elif it.dhash is None:
            it.category, it.keep, it.reason = "unreadable", False, "could not decode image"
        elif it.make == "Apple":
            it.category = "iphone"
        else:
            it.category = "other"
    # iPhone file numbers run in capture order and resaved copies sort after their
    # original ("IMG_4206.jpg" < "IMG_4206_1.jpg"), so the first by name is the original kept.
    candidates = sorted((i for i in items if i.keep and i.category == "iphone"), key=lambda i: i.path.name)
    kept: List[Item] = []
    for it in candidates:
        twin = next((k for k in kept if hamming(k.dhash, it.dhash) <= near_dup_bits), None)
        if twin is not None:
            it.keep, it.reason, it.duplicate_of = False, f"near-duplicate (dHash distance {hamming(twin.dhash, it.dhash)})", twin.path.name
        else:
            kept.append(it)


def write_outputs(items: List[Item], out: Path, probe_dir: Path, src: Path) -> Dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "category", "keep", "reason", "duplicate_of", "width", "height", "megapixels",
                    "camera", "taken", "gps", "sha256_12", "dhash16"])
        for it in sorted(items, key=lambda i: i.path.name):
            w.writerow([it.path.name, it.category, "yes" if it.keep else "no", it.reason, it.duplicate_of,
                        it.width, it.height, round(it.width * it.height / 1e6, 1),
                        f"{it.make} {it.model}".strip(), it.taken, "yes" if it.gps else "no",
                        it.sha256[:12], f"{it.dhash:064x}" if it.dhash is not None else ""])
    pilot = [it for it in items if it.keep]
    with open(out / "tags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "span", "support", "sp35_hint", "component_hint", "note"])
        for it in sorted(pilot, key=lambda i: i.path.name):
            code, what = PROBE_SET.get(it.path.name, ("", ""))
            note = "visual hint, engineer to confirm; " if code else ""
            w.writerow([it.path.name, "", "", code, what, note + "span/support tagging with the bridge engineer in WP1.3"])
    with open(out / "streetview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "note"])
        for it in sorted((i for i in items if i.category == "streetview" and not i.duplicate_of), key=lambda i: i.path.name):
            w.writerow([it.path.name, it.reason])

    missing = [n for n in PROBE_SET if n not in {i.path.name for i in pilot}]
    if missing:
        raise SystemExit(f"probe photos not in the pilot set: {missing}")
    if probe_dir.exists() and any(probe_dir.iterdir()):
        current = sorted(p.name for p in probe_dir.iterdir())
        if current != sorted(PROBE_SET):
            dest = out / "_superseded" / f"{probe_dir.name}_{dt.date.today().isoformat()}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(probe_dir), str(dest))
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name in PROBE_SET:
        if not (probe_dir / name).exists():
            shutil.copy2(src / name, probe_dir / name)
    counts: Dict[str, int] = {"files": len(items), "pilot": len(pilot)}
    for it in items:
        key = f"excluded: {it.reason}" if not it.keep else f"kept: {it.category}"
        key = re.sub(r" \(dHash distance \d+\)", "", key)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the pilot photo set")
    ap.add_argument("--src", default="../yugmi_images")
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--probe-dir", default="data/pilot/probe10")
    args = ap.parse_args(argv)
    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no files in {src}")
    items = [read_item(p) for p in files]
    classify(items)
    counts = write_outputs(items, Path(args.out), Path(args.probe_dir), src)
    for k, v in counts.items():
        print(f"{v:5d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
