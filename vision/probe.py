"""VRAM probe (skill vram-probe, plan task 0.4).

Loads the model once with vision.model.load_model, runs every prompt of the
library on each probe photo, then one tiled native-resolution pass, and writes
a JSON + Markdown report. The pass/fail decision is computed here, not by hand.

    .venv-win\\Scripts\\python.exe -m vision.probe --photos data/pilot/probe10 --tile-photo IMG_4231.jpg
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

GIB = 2 ** 30


def _peak():
    import torch
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photos", default="data/pilot/probe10")
    ap.add_argument("--prompts", default="kb/prompts/sam3_prompts.yaml")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--budget-gb", type=float, default=3.4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tile-photo", default="IMG_4231.jpg", help="photo for the native-resolution tiled pass ('' to skip)")
    ap.add_argument("--tile-size", type=int, default=1008)
    ap.add_argument("--label", default="rtx2050")
    ap.add_argument("--out", default="docs/reports")
    args = ap.parse_args(argv)

    import torch
    from PIL import Image, ImageOps

    from .model import detect, inference_context, load_model, make_processor
    from .postprocess import load_prompt_library
    from .tiling import tile_boxes

    lib = load_prompt_library(Path(args.prompts))
    prompts = lib.prompt_list()
    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        raise SystemExit(f"no photos in {args.photos}")
    model, info = load_model(dtype=args.dtype, budget_gb=args.budget_gb)
    proc = make_processor(model, "cuda", lib.min_threshold)
    print("load", json.dumps(info), flush=True)

    rows, oom = [], False
    for ph in photos:
        img = ImageOps.exif_transpose(Image.open(ph)).convert("RGB")
        native = img.size
        img.thumbnail((args.max_side, args.max_side))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        row = {"photo": ph.name, "native": list(native), "processed": list(img.size)}
        try:
            with inference_context("cuda"):
                dets = detect(proc, img, prompts)
            row.update(masks=len(dets), oom=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row.update(masks=0, oom=True)
            oom = True
        alloc, reserved = _peak()
        row.update(sec=round(time.time() - t0, 1), sec_per_prompt=round((time.time() - t0) / len(prompts), 3),
                   peak_alloc_gb=round(alloc / 1e9, 2), peak_reserved_gb=round(reserved / 1e9, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()

    tiled = None
    if args.tile_photo:
        ph = Path(args.photos) / args.tile_photo
        img = ImageOps.exif_transpose(Image.open(ph)).convert("RGB")
        boxes = tile_boxes(*img.size, size=args.tile_size)
        tile_prompts = lib.prompt_list(kinds=("distress",), tile_only=True)
        torch.cuda.reset_peak_memory_stats()
        t0, n, tile_oom = time.time(), 0, False
        try:
            with inference_context("cuda"):
                for b in boxes:
                    n += len(detect(proc, img.crop(b), tile_prompts, frame="tile"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tile_oom = oom = True
        alloc, reserved = _peak()
        tiled = {"photo": ph.name, "native": list(img.size), "tiles": len(boxes), "tile_size": args.tile_size,
                 "prompts": len(tile_prompts), "masks": n, "oom": tile_oom, "sec": round(time.time() - t0, 1),
                 "peak_alloc_gb": round(alloc / 1e9, 2), "peak_reserved_gb": round(reserved / 1e9, 2)}
        print("tiled", json.dumps(tiled), flush=True)

    peaks = [r["peak_alloc_gb"] for r in rows] + ([tiled["peak_alloc_gb"]] if tiled else [])
    reserved = [r["peak_reserved_gb"] for r in rows] + ([tiled["peak_reserved_gb"]] if tiled else [])
    worst_alloc, worst_reserved = max(peaks), max(reserved)
    free_gb = info["free_at_start_gib"] * GIB / 1e9
    verdict = {
        "budget_gb": args.budget_gb,
        "worst_peak_alloc_gb": worst_alloc,
        "worst_peak_reserved_gb": worst_reserved,
        "free_at_start_gb": round(free_gb, 2),
        "within_budget": worst_alloc <= args.budget_gb,
        "no_oom": not oom,
        "no_shared_memory_spill": worst_reserved <= free_gb,
        "median_sec_per_photo": sorted(r["sec"] for r in rows)[len(rows) // 2],
    }
    verdict["pass_3_4gb"] = verdict["within_budget"] and verdict["no_oom"] and verdict["no_shared_memory_spill"]
    report = {"date": dt.date.today().isoformat(), "label": args.label, "max_side": args.max_side,
              "prompt_library": {"version": lib.version, "sha12": lib.sha12, "prompts": len(prompts)},
              "model": info, "photos": rows, "tiled_pass": tiled, "verdict": verdict}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"vram_probe_{args.label}_{report['date']}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / f"{stem}.md").write_text(_markdown(report), encoding="utf-8")
    print("PROBE", json.dumps(verdict), flush=True)
    return 0 if verdict["pass_3_4gb"] else 1


def _markdown(r: dict) -> str:
    m, v = r["model"], r["verdict"]
    lines = [
        f"# VRAM probe — {m.get('gpu', '?')} ({r['date']})",
        "",
        "Generated by `python -m vision.probe`; the verdict below is computed, not hand-written.",
        "",
        f"- Environment: torch {m['torch']} (CUDA {m['cuda_build']}), sam3 {m['sam3']}, weights `{Path(m['checkpoint']).name}` "
        f"(sha256 {m['checkpoint_sha256_prefix']}…), dtype {m['dtype']} ({m['bf16_params_m']} M params in bf16).",
        f"- GPU memory: {m['total_gib']} GiB total, {m['free_at_start_gib']} GiB free at start, process cap {m['cap_gib']} GiB.",
        f"- Model load: {m['load_sec']} s, {m['alloc_after_load_gib']} GiB allocated; checkpoint keys: {m['missing_keys']} known-missing, "
        f"{m['unused_tracker_keys']} unused tracker-neck tensors (checked).",
        f"- Prompts: {r['prompt_library']['prompts']} from `kb/prompts/sam3_prompts.yaml` v{r['prompt_library']['version']} "
        f"(sha {r['prompt_library']['sha12']}); photos downscaled to {r['max_side']} px long side.",
        "",
        "| photo | native | processed | sec | s/prompt | peak alloc GB | peak reserved GB | masks | OOM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in r["photos"]:
        lines.append(f"| {p['photo']} | {p['native'][0]}x{p['native'][1]} | {p['processed'][0]}x{p['processed'][1]} | "
                     f"{p['sec']} | {p['sec_per_prompt']} | {p['peak_alloc_gb']} | {p['peak_reserved_gb']} | {p['masks']} | "
                     f"{'yes' if p['oom'] else 'no'} |")
    t = r["tiled_pass"]
    if t:
        lines += ["", f"Tiled native pass: {t['photo']} {t['native'][0]}x{t['native'][1]}, {t['tiles']} tiles of {t['tile_size']} px, "
                      f"{t['prompts']} tile prompts, {t['sec']} s, peak alloc {t['peak_alloc_gb']} GB, reserved {t['peak_reserved_gb']} GB, "
                      f"{t['masks']} masks, OOM {'yes' if t['oom'] else 'no'}."]
    lines += ["", "## Verdict", "",
              f"- Worst peak allocated **{v['worst_peak_alloc_gb']} GB** vs budget {v['budget_gb']} GB → {'within' if v['within_budget'] else 'OVER'} budget.",
              f"- Worst peak reserved {v['worst_peak_reserved_gb']} GB vs {v['free_at_start_gb']} GB free → "
              f"{'no shared-memory spill' if v['no_shared_memory_spill'] else 'SPILLS into shared memory'}.",
              f"- Out-of-memory: {'none' if v['no_oom'] else 'YES'}. Median {v['median_sec_per_photo']} s per photo.",
              "", f"**{'PASS' if v['pass_3_4gb'] else 'FAIL'}** (pass_3_4gb = {str(v['pass_3_4gb']).lower()})", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
