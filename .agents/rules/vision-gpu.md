---
trigger: glob
globs: vision/**
description: GPU, memory and prompt rules for the SAM 3.1 vision worker.
---

# Vision worker rules (SAM 3.1)

- Load the model only with `vision.model.load_model()`. It builds the `sam3` image model, loads `data/weights/sam3.1/sam3.1_multiplex.pt`, asserts the known key mismatch (exactly 4 missing `convs.3.*` tensors, discarded by `scalp=1`; only tracker-neck tensors unused) and raises `CheckpointMismatch` on anything else.
- bf16: only the floating-point parameters of the vision trunk and text encoder are cast (95% of weights). Never call `model.to(torch.bfloat16)` or `.to(bfloat16)` on the trunk: the complex64 RoPE buffers lose their imaginary part and detections collapse (audit R2).
- Run inference inside `vision.model.inference_context()` (inference_mode + bf16 autocast) and with `vision.model.detect()`: one `set_image` per photo, one `set_text_prompt` per prompt; each prompt replaces the previous one in the processor state.
- `Sam3Processor` drops results below its `confidence_threshold`; it is set to the library's lowest class threshold, and per-class thresholds are applied in `postprocess`.
- VRAM: `load_model` caps the process below the free VRAM (`budget_gb`, default 3.4 GB) so an overflow raises OutOfMemoryError instead of paging into shared memory. Measured on the RTX 2050: peak 2.57 GB, ~16.5 s per photo for 87 prompts. On OOM: empty cache → retry at 0.75× size → mark "needs batch GPU". Never crash the folder run.
- Checkpoint: Hugging Face `facebook/sam3.1` (gated), sha256 `0567debe…`; download only with `tools/fetch_weights.py` (token from `HF_TOKEN`, never in a URL, command line or log).
- Prompts come only from `kb/prompts/sam3_prompts.yaml`. Components carry `sp35` (C-code), distresses `sp35_row` (Table 7.2/7.3 row). No prompt strings in code.
- Post-processing rules live in `vision/postprocess.py` and are unit-tested: synonym merge, suppression of cables/ropes/posters/people/vehicles, distress must sit inside a component (or the photo's tagged component), whole-component re-detections rejected, cross-class duplicates resolved. Rejected detections keep their reason.
- Distress output is a proposal ("proposed, unverified") until the Phase 2 evaluation baseline exists; nothing feeds the SP:35 engine automatically.
- Tiled pass (`--tile`, native resolution, 20% overlap) only for `tile: true` classes.
- Outputs: per-photo JSON (COCO-RLE, bbox, area, SP:35 codes, host) + overlay PNG; `manifest.jsonl`; runs resume with `--resume`.
- Same code on native Windows (RTX 2050, sm_86) and in Docker on the RTX 3060 (sm_86) / RTX 5070 Ti (sm_120 → CUDA 12.8+ builds only).
