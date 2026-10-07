"""Full VRAM probe (skill vram-probe): 10 photos x all prompts bf16 + 1 tiled pass."""
import json, time
from pathlib import Path
import torch
import yaml
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
lib = yaml.safe_load(Path("kb/prompts/sam3_prompts.yaml").read_text(encoding="utf-8"))
prompts = [(c.get("id"), t) for c in lib.get("components", []) + lib.get("distresses", [])
           for t in c.get("prompts", [])]
print("prompt_count", len(prompts))
photos = sorted(Path("data/pilot/probe10").glob("*.*"))[:10]
print("photos", len(photos))

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = build_sam3_image_model(checkpoint_path="data/weights/sam3.1/sam3.1_multiplex.pt",
                               device="cuda", load_from_HF=False)
MAX_SIDE = 1024
load_peak = torch.cuda.max_memory_allocated() / 1e9
print("load", json.dumps({"sec": round(time.time() - t0, 1), "peak_gb": round(load_peak, 2)}))
proc = Sam3Processor(model)
rows = []
for ph in photos:
    img = Image.open(ph).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(img)
        n = 0
        for cid, text in prompts:
            out = proc.set_text_prompt(state=state, prompt=text)
            m = out.get("masks", []) if isinstance(out, dict) else []
            n += len(m)
    rows.append({"photo": ph.name, "size": list(img.size), "sec": round(time.time() - t1, 1),
                 "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "masks": n})
    print(json.dumps(rows[-1]))
worst = max(r["peak_gb"] for r in rows)
print("PROBE", json.dumps({"worst_peak_gb": worst, "pass_3_4gb": worst <= 3.4,
                           "total_masks": sum(r["masks"] for r in rows)}))
