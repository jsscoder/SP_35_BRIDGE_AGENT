"""First-light: load local sam3.1 weight on cuda, 1 photo x 1 prompt."""
import json, time
from pathlib import Path
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

ckpt = Path("data/weights/sam3.1/sam3.1_multiplex.pt")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = build_sam3_image_model(checkpoint_path=str(ckpt), device="cuda", load_from_HF=False)
print("load_sec", round(time.time() - t0, 1), "peak_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))
proc = Sam3Processor(model)
img = Image.open("data/pilot/probe10/14930adc-d817-4c24-8136-045d9616902f.jpg").convert("RGB")
print("img", img.size)
t1 = time.time()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    state = proc.set_image(img)
    out = proc.set_text_prompt(state=state, prompt="bridge pier")
t2 = time.time()
print(json.dumps({"infer_sec": round(t2 - t1, 1),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  "keys": sorted(out.keys()) if isinstance(out, dict) else type(out).__name__}))
