"""SAM 3.1 inferencer: one photo or a whole folder, model loaded once.

For each photo it writes `<stem>.json` (components, distresses, rejected
detections with reasons, all with SP:35 codes and COCO-RLE masks) and
`<stem>_overlay.png`, and appends a line to `manifest.jsonl`. A failure on one
photo is recorded and the run continues. Distress output is a proposal only:
it is not scored until the Phase 2 evaluation baseline exists (audit A5).

    .venv-win/Scripts/python.exe -m vision.infer --folder data/pilot/probe10 --resume
    .venv-win/Scripts/python.exe -m vision.infer --manifest data/pilot/manifest.csv --src ../yugmi_images --outdir data/pilot/infer_all --resume
    .venv-win/Scripts/python.exe -m vision.infer --photo data/pilot/probe10/IMG_4231.jpg --tile
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PHOTO_EXT = (".jpg", ".jpeg", ".png")
GIB = 2 ** 30


def _rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as cocomask

    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _resize_mask(mask: np.ndarray, size) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def _crop_scaled(mask_small: np.ndarray, scale: float, box) -> np.ndarray:
    """The part of a photo-level mask under a native-resolution tile, at tile size."""
    x0, y0, x1, y1 = box
    sx0, sy0 = int(x0 * scale), int(y0 * scale)
    sx1 = max(sx0 + 1, math.ceil(x1 * scale))
    sy1 = max(sy0 + 1, math.ceil(y1 * scale))
    return _resize_mask(mask_small[sy0:sy1, sx0:sx1], (x1 - x0, y1 - y0))


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def photos_from_manifest(manifest: Path, src: Path) -> List[Path]:
    """Photos marked keep=yes in a build_pilot manifest, resolved under `src`."""
    with open(manifest, newline="", encoding="utf-8") as f:
        photos = [Path(src) / r["file"] for r in csv.DictReader(f) if r.get("keep") == "yes"]
    missing = [p for p in photos if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} photo(s) listed in {manifest} are missing under {src}, e.g. {missing[0]}")
    return photos


def load_tags(path: Optional[Path], names: dict) -> dict:
    """file -> (label, SP:35 code) from tags.csv `sp35_hint`; empty if the file is absent."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("sp35_hint") or "").strip()
            if code:
                out[row["file"]] = (names.get(code, code), code)
    return out


def tiled_pass(proc, full, res, scale: float, lib, tile_size: int, photo_host=None) -> dict:
    """Native-resolution pass for `tile: true` classes. Each tile is checked against the
    photo-level components and suppressors cropped to it, so the same rules apply;
    a defect seen in two overlapping tiles is kept once."""
    from .model import detect, inference_context
    from .postprocess import Detection, postprocess, record
    from .tiling import tile_boxes

    W, H = full.size
    hosts = res.components + res.suppressors
    prompts = lib.prompt_list(("distress",), tile_only=True)
    kept: List[Tuple[Detection, int, int]] = []
    rejected = []
    boxes = tile_boxes(W, H, size=tile_size)
    with inference_context("cuda"):
        for box in boxes:
            x0, y0 = box[0], box[1]
            raw = detect(proc, full.crop(box), prompts, frame="tile")
            if not raw:
                continue
            tile_hosts = [Detection(cls=h.cls, prompt=h.prompt, score=h.score,
                                    mask=_crop_scaled(h.mask, scale, box)) for h in hosts]
            tres = postprocess(tile_hosts + raw, lib, photo_host=photo_host)
            kept += [(d, x0, y0) for d in tres.distresses]
            for d in tres.rejected:
                if d.cls.kind == "distress":
                    r = record(d)
                    r["bbox"] = [r["bbox"][0] + x0, r["bbox"][1] + y0, r["bbox"][2] + x0, r["bbox"][3] + y0]
                    rejected.append(r)
    final = []
    for d, x0, y0 in sorted(kept, key=lambda t: -t[0].score):
        b = d.bbox()
        full_box = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
        if any(cid == d.cls.id and _box_iou(full_box, fb) >= 0.5 for cid, fb, _ in final):
            continue
        final.append((d.cls.id, full_box, (d, x0, y0)))
    out = []
    for _, full_box, (d, x0, y0) in final:
        canvas = np.zeros((H, W), bool)
        h, w = d.mask.shape
        canvas[y0:y0 + h, x0:x0 + w] = d.mask
        r = record(d, _rle(canvas))
        r["bbox"] = full_box
        out.append(r)
    return {"frame": "native", "size": [W, H], "tile_size": tile_size, "tiles": len(boxes),
            "distresses": out, "rejected": rejected}


def process_photo(proc, photo: Path, lib, max_side: int, tile: bool, tile_size: int,
                  photo_host=None) -> dict:
    import torch
    from PIL import Image, ImageOps

    from .model import detect, inference_context
    from .postprocess import postprocess, record, render_overlay

    full = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    result = {"photo": photo.name, "native_size": list(full.size)}
    for attempt, side in enumerate((max_side, int(max_side * 0.75))):
        img = full.copy()
        img.thumbnail((side, side))
        scale = img.size[0] / full.size[0]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with inference_context("cuda"):
                raw = detect(proc, img, lib.prompt_list())
            res = postprocess(raw, lib, photo_host=photo_host)
            tiles = tiled_pass(proc, full, res, scale, lib, tile_size, photo_host) if tile else None
            status = "ok" if attempt == 0 else "ok_reduced"
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    else:
        result.update(status="needs_batch_gpu", detail="out of GPU memory at both sizes")
        return result

    result.update(
        status=status, processed_size=list(img.size), scale=round(scale, 5),
        sec=round(time.time() - t0, 1),
        peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2),
        counts=res.counts(),
        components=[record(d, _rle(d.mask)) for d in res.components],
        distresses=[record(d, _rle(d.mask)) for d in res.distresses],
        suppressors=[record(d) for d in res.suppressors],
        rejected=[record(d) for d in res.rejected],
        distress_status="proposed, unverified (no evaluation baseline yet)",
        _overlay=render_overlay(img, res),
    )
    if tiles is not None:
        result["tiled"] = tiles
    return result


def run(photos: List[Path], outdir: Path, prompts: Path, max_side: int, tile: bool, tile_size: int,
        resume: bool, dtype: str, budget_gb: float, tags: Optional[Path] = None) -> dict:
    from .model import load_model, make_processor
    from .postprocess import load_prompt_library

    lib = load_prompt_library(prompts)
    photo_tags = load_tags(tags, {c.sp35_ref: c.id for c in lib.of_kind("component")})
    outdir.mkdir(parents=True, exist_ok=True)
    model, info = load_model(dtype=dtype, budget_gb=budget_gb)
    proc = make_processor(model, "cuda", lib.min_threshold)
    manifest = outdir / "manifest.jsonl"
    done, summary = 0, {"ok": 0, "ok_reduced": 0, "needs_batch_gpu": 0, "error": 0, "skipped": 0}
    for photo in photos:
        out_json = outdir / f"{photo.stem}.json"
        if resume and out_json.exists():
            try:
                if json.loads(out_json.read_text(encoding="utf-8")).get("status") in ("ok", "ok_reduced"):
                    summary["skipped"] += 1
                    continue
            except json.JSONDecodeError:
                pass
        tag = photo_tags.get(photo.name)
        try:
            result = process_photo(proc, photo, lib, max_side, tile, tile_size, tag)
        except Exception as e:  # one bad photo must not stop the folder
            result = {"photo": photo.name, "status": "error", "detail": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc(limit=5)}
        overlay = result.pop("_overlay", None)
        if overlay is not None:
            overlay.save(outdir / f"{photo.stem}_overlay.png")
        if tag:
            result["photo_tag"] = {"label": tag[0], "sp35": tag[1]}
        result.update(prompt_library={"version": lib.version, "sha12": lib.sha12},
                      model={k: info.get(k) for k in ("sam3", "checkpoint_sha256_prefix", "dtype", "gpu")},
                      created=dt.datetime.now().isoformat(timespec="seconds"))
        out_json.write_text(json.dumps(result), encoding="utf-8")
        line = {k: result.get(k) for k in ("photo", "status", "sec", "peak_alloc_gib", "counts", "detail")}
        if "tiled" in result:
            line["tiled_distresses"] = len(result["tiled"]["distresses"])
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        summary[result["status"]] += 1
        done += 1
    summary.update(photos=len(photos), processed=done, model=info)
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SAM 3.1 inferencer (one photo or a folder)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photo")
    src.add_argument("--folder")
    src.add_argument("--manifest", help="build_pilot manifest.csv: run every photo marked keep=yes")
    ap.add_argument("--src", default="../yugmi_images", help="folder holding the photos named in --manifest")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N photos (trial runs)")
    ap.add_argument("--outdir", default="data/pilot/infer_out")
    ap.add_argument("--prompts", default="kb/prompts/sam3_prompts.yaml")
    ap.add_argument("--tags", default="data/pilot/tags.csv",
                    help="tags.csv with an sp35_hint column; the tagged component hosts distresses SAM can't place")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--tile", action="store_true", help="also run the native-resolution tiled pass")
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--resume", action="store_true", help="skip photos that already have an ok result")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--budget-gb", type=float, default=3.4)
    args = ap.parse_args(argv)
    if args.photo:
        photos = [Path(args.photo)]
    elif args.manifest:
        photos = photos_from_manifest(Path(args.manifest), Path(args.src))
    else:
        photos = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in PHOTO_EXT)
    if args.limit > 0:
        photos = photos[:args.limit]
    if not photos:
        raise SystemExit("no photos found")
    print(f"{len(photos)} photo(s) -> {args.outdir}", flush=True)
    summary = run(photos, Path(args.outdir), Path(args.prompts), args.max_side, args.tile, args.tile_size,
                  args.resume, args.dtype, args.budget_gb, Path(args.tags) if args.tags else None)
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "model"}))
    return 0 if summary["error"] == 0 and summary["needs_batch_gpu"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
