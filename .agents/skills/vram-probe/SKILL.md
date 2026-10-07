---
name: vram-probe
description: Measure SAM 3.1 peak VRAM and speed on the current GPU — gate G0 on the RTX 2050 laptop, and later on the RTX 3060 / RTX 5070 Ti. Use when asked about memory, speed or out-of-memory errors, or before changing input size, dtype or the number of prompts.
---

# VRAM probe

1. Record the GPU: `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv`.
2. In the worker environment run:
   `uv run python -m vision.probe --photos data/pilot/probe10 --prompts kb/prompts/sam3_prompts.yaml --dtype bf16`
   If `vision/probe.py` does not exist yet, create it: load the image model once, run 10 photos × all component and distress prompts (one `set_image` per photo, one `set_text_prompt` per prompt), then one tiled pass on one photo.
3. Record for each attempt: GPU, driver, torch and CUDA versions, input size, dtype, peak `torch.cuda.max_memory_allocated()`, peak reserved, seconds per photo (image encode vs per prompt), out-of-memory yes/no.
4. Pass mark on the RTX 2050: peak allocated ≤ 3.4 GB with no out-of-memory error.
   If it fails, try in this order and record each: smaller input → sequential prompts with `torch.cuda.empty_cache()` → text encoder on CPU → CPU-only timing → remote worker on the RTX 3060.
5. Write `docs/reports/vram_probe_<gpu>_<yyyy-mm-dd>.md` with the table and a one-paragraph recommendation for G0.
