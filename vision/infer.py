"""One-click SAM 3.1 inferencer.

Photo in -> masks JSON (COCO-RLE) + overlay PNG out. Prompts only from
kb/prompts/sam3_prompts.yaml. Model loads once per process (local weight,
cuda, bf16 autocast). OOM -> empty cache -> mark needs batch GPU, never crash.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

CKPT = "data/weights/sam3.1/sam3.1_multiplex.pt"
MAX_SIDE = 1024


def load_prompts(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="One-click SAM 3.1 inferencer")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--outdir", default="data/pilot/infer_out")
    ap.add_argument("--prompts", default="kb/prompts/sam3_prompts.yaml")
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args()
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image
    import numpy as np
    from pycocotools import mask as cocomask

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lib = load_prompts(Path(args.prompts))
    photo = Path(args.photo)
    img = Image.open(photo).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    W, H = img.size

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = build_sam3_image_model(checkpoint_path=CKPT, device="cuda", load_from_HF=False)
        proc = Sam3Processor(model)
        overlay = np.array(img).astype(np.int32)
        dets = []
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for cls in lib.get("components", []) + lib.get("distresses", []):
                for text in cls.get("prompts", []):
                    out = proc.set_text_prompt(state=state, prompt=text)
                    ms = out.get("masks", []) if isinstance(out, dict) else []
                    sc = out.get("scores", [0] * len(ms)) if isinstance(out, dict) else []
                    for m, s in zip(ms, sc):
                        try:
                            sv = float(s.max() if hasattr(s, "max") else s)
                        except Exception:
                            sv = 0.0
                        if sv < args.score:
                            continue
                        m_np = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m))
                        if m_np.ndim > 2:
                            m_np = m_np[0]
                        mh, mw = m_np.shape[-2:]
                        full = np.zeros((H, W), dtype=np.uint8)
                        full[:mh, :mw] = (m_np > 0.5).astype(np.uint8)
                        rle = cocomask.encode(np.asfortranarray(full))
                        rle["counts"] = rle["counts"].decode()
                        dets.append({"class": cls.get("id"), "sp35": cls.get("sp35"),
                                     "prompt": text, "score": round(sv, 3), "rle": rle})
                        overlay[full > 0] = (overlay[full > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.int32)
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        status, dets = "needs batch GPU", []
        overlay = np.array(img).astype(np.int32)
    Image.fromarray(overlay.astype("uint8")).save(outdir / (photo.stem + "_overlay.png"))
    result = {"photo": photo.name, "size": [W, H], "sec": round(time.time() - t0, 1),
              "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
              "status": status, "detections": dets}
    (outdir / (photo.stem + ".json")).write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "detections"},
                     indent=1) + f"\ndetections: {len(dets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
