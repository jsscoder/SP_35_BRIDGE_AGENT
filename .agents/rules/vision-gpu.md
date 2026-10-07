---
trigger: glob
globs: vision/**
description: GPU, memory and prompt rules for the SAM 3.1 vision worker.
---

# Vision worker rules (SAM 3.1)

- Use the official repo API:
  `from sam3.model_builder import build_sam3_image_model`
  `from sam3.model.sam3_image_processor import Sam3Processor`
  `state = processor.set_image(img)` once per photo, then `processor.set_text_prompt(state=state, prompt=...)` per prompt → `masks`, `boxes`, `scores`.
- Checkpoints: Hugging Face `facebook/sam3.1` (gated). Read the token from the environment (`HF_TOKEN`); never write it to code, logs or the repo.
- Desk app loads the image model only (no video predictor), once per process, under bf16 autocast.
- VRAM budget on the RTX 2050 (4 GB, ~0.25 GB used by Windows): peak allocated ≤ 3.4 GB. Log `torch.cuda.max_memory_allocated()` per job.
- On out-of-memory: empty the cache → retry at a smaller input → CPU fallback → mark the job "needs batch GPU". Never crash the app.
- Prompts come only from `kb/prompts/sam3_prompts.yaml` (text, synonyms, threshold, min area, tiling flag, SP:35 mapping). No prompt strings in code.
- A distress mask counts only inside a component mask. Subtract suppression masks (cables, ropes, posters, people, vehicles) first.
- Tiled pass (native resolution, 20% overlap) only for classes with `tile: true` and only on photos the user selects or the job marks.
- Outputs per photo: COCO-RLE JSON + overlay PNG. Job manifest in JSONL; jobs resume after interruption.
- The same code runs in WSL2 on the RTX 2050 (sm_86), and in Docker on the RTX 3060 (sm_86) and the RTX 5070 Ti (sm_120, CUDA 12.8+ builds only).
