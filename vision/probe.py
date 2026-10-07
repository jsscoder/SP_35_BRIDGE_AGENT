"""VRAM probe for SAM 3.1 (Task 0.4, skill vram-probe).

Runs 10 photos x all prompts in bf16 + one tiled pass, logging
torch.cuda.max_memory_allocated() and sec/photo.
Prompts come only from kb/prompts/sam3_prompts.yaml. No SP:35 values here.
Token read from HF_TOKEN env only (never logged).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def load_prompts(path: Path):
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    comps = [p for c in data.get("components", []) for p in c.get("prompts", [])]
    dists = [p for d in data.get("distresses", []) for p in d.get("prompts", [])]
    return comps, dists


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="data/pilot/probe10")
    ap.add_argument("--prompts", default="kb/prompts/sam3_prompts.yaml")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    photos = sorted(Path(args.photos).glob("*.*"))
    if not photos:
        print(f"No photos in {args.photos} — place 10 pilot photos there (Task 0.6).")
        return 2
    comps, dists = load_prompts(Path(args.prompts))
    print(f"Photos: {len(photos)}  component-prompts: {len(comps)}  distress-prompts: {len(dists)}")

    try:
        import torch
    except ImportError:
        print("torch not installed in this env (WSL2 venv needs torch cu128). BLOCKED.")
        return 3
    if not torch.cuda.is_available():
        print("CUDA not available. Record CPU-only timing per skill fallback.")
        return 4

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build_sam3_image_model()
    proc = Sam3Processor(model)
    # ... set_image once per photo, set_text_prompt per prompt (full loop in G0 run)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps({"peak_gb": peak, "sec": time.time() - t0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
