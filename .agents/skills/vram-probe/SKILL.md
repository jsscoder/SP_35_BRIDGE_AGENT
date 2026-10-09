---
name: vram-probe
description: Measure SAM 3.1 peak VRAM and speed on the current GPU — gate G0 on the RTX 2050 laptop, and later on the RTX 3060 / RTX 5070 Ti. Use when asked about memory, speed or out-of-memory errors, or before changing input size, dtype, the prompt library or the loader.
---

# VRAM probe

1. Check the environment: `.venv-win\Scripts\python.exe tools\env_check.py` (add `--full` to hash the checkpoint).
2. From the repo root run:
   `.venv-win\Scripts\python.exe -m vision.probe --photos data/pilot/probe10 --tile-photo IMG_4231.jpg --label <gpu>`
   It loads the model once via `vision.model.load_model`, runs every prompt of `kb/prompts/sam3_prompts.yaml` on each probe photo, then one tiled native-resolution pass, and writes `docs/reports/vram_probe_<gpu>_<date>.json` and `.md`.
3. The verdict is computed by the probe, never edited by hand: `pass_3_4gb` requires peak allocated ≤ budget, no OOM, and peak reserved ≤ free VRAM at start (no shared-memory spill).
4. If it fails, try in order and record each run: smaller `--max-side` → fewer prompts (synonyms) → text encoder on CPU → remote worker on the RTX 3060.
5. Commit both report files; `tools/verify_phase0.py` reads the JSON.
