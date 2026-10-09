"""Load SAM 3.1 once, the same way everywhere (audit A2, A8).

- Builds the SAM 3 image model from the installed `sam3` package (pinned in
  requirements/vision-win.txt) and loads `sam3.1_multiplex.pt` itself, so the
  key mismatch can be checked: exactly the 4 tensors of the 0.5x neck level are
  missing (that level is discarded by `scalp=1`, so detections are unaffected)
  and only tracker-neck tensors are unused. Anything else raises.
- Casts the floating-point weights of the vision trunk and text encoder (95% of
  parameters) to bf16. It casts parameters one by one: `Module.to(bfloat16)`
  would also turn the trunk's complex64 RoPE buffers real and break detection.
  Measured on the RTX 2050: peak 2.34 GiB instead of 4.1 GB, ~4x faster,
  mask IoU 0.955 against the fp32 run.
- Caps this process's CUDA memory below the free VRAM, so an overflow raises
  OutOfMemoryError instead of silently paging into shared system memory.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from importlib import metadata
from pathlib import Path
from typing import Tuple

CHECKPOINT = Path("data/weights/sam3.1/sam3.1_multiplex.pt")
# Git LFS object id of sam3.1_multiplex.pt in facebook/sam3.1 at revision daa63191845a
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
CHECKPOINT_BYTES = 3_502_755_717
EXPECTED_MISSING = frozenset({
    "backbone.vision_backbone.convs.3.conv_1x1.weight",
    "backbone.vision_backbone.convs.3.conv_1x1.bias",
    "backbone.vision_backbone.convs.3.conv_3x3.weight",
    "backbone.vision_backbone.convs.3.conv_3x3.bias",
})
ALLOWED_UNUSED_PREFIXES = ("backbone.vision_backbone.interactive_convs.",
                           "backbone.vision_backbone.propagation_convs.")
DEFAULT_BUDGET_GB = 3.4
GIB = 2 ** 30


class CheckpointMismatch(RuntimeError):
    """The checkpoint and the installed sam3 code do not pair as expected."""


def sha256_file(path: Path, chunk: int = 16 * 2 ** 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(path: Path = CHECKPOINT, full: bool = False) -> Tuple[bool, str]:
    """Size check always; full sha256 (about 15 s) when asked."""
    path = Path(path)
    if not path.exists():
        return False, f"missing {path}"
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        return False, f"size {size} != expected {CHECKPOINT_BYTES}"
    if full:
        digest = sha256_file(path)
        if digest != CHECKPOINT_SHA256:
            return False, f"sha256 {digest[:12]} != expected {CHECKPOINT_SHA256[:12]}"
        return True, f"sha256 {digest[:12]} ok"
    return True, "size ok (sha256 not checked)"


def load_model(checkpoint: Path = CHECKPOINT, device: str = "cuda", dtype: str = "bf16",
               budget_gb: float = DEFAULT_BUDGET_GB, safety_mib: int = 192):
    """Return (model, info). `dtype` is "bf16" (default) or "fp32"."""
    import torch
    from sam3.model_builder import build_sam3_image_model

    if dtype not in ("bf16", "fp32"):
        raise ValueError("dtype must be bf16 or fp32")
    ok, detail = verify_checkpoint(checkpoint)
    if not ok:
        raise CheckpointMismatch(detail)
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "sam3": metadata.version("sam3"), "checkpoint": str(checkpoint),
            "checkpoint_sha256_prefix": CHECKPOINT_SHA256[:12], "dtype": dtype, "device": device}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; pass device='cpu'")
        free, total = torch.cuda.mem_get_info()
        cap = min(budget_gb * 1e9, free - safety_mib * 2 ** 20)
        if cap <= 0:
            raise RuntimeError(f"only {free / GIB:.2f} GiB free on the GPU")
        torch.cuda.set_per_process_memory_fraction(cap / total)
        info.update(gpu=torch.cuda.get_device_name(0), free_at_start_gib=round(free / GIB, 2),
                    total_gib=round(total / GIB, 2), cap_gib=round(cap / GIB, 2))

    t0 = time.time()
    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, device="cpu", eval_mode=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    detector = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    missing, unexpected = model.load_state_dict(detector, strict=False)
    stray = [k for k in unexpected if not k.startswith(ALLOWED_UNUSED_PREFIXES)]
    if set(missing) != EXPECTED_MISSING or stray:
        raise CheckpointMismatch(
            f"unexpected pairing of sam3 {info['sam3']} and {checkpoint.name}: "
            f"missing={sorted(set(missing) ^ EXPECTED_MISSING)} stray_unused={stray[:5]}")
    del ckpt, detector

    cast = 0
    if dtype == "bf16":
        for sub in (model.backbone.vision_backbone.trunk, model.backbone.language_backbone.encoder):
            for p in sub.parameters():
                if p.is_floating_point():
                    p.data = p.data.to(torch.bfloat16)
                    cast += p.numel()
    model = model.to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
        info["alloc_after_load_gib"] = round(torch.cuda.memory_allocated() / GIB, 2)
    info.update(load_sec=round(time.time() - t0, 1), bf16_params_m=round(cast / 1e6, 1),
                missing_keys=len(missing), unused_tracker_keys=len(unexpected),
                params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
    return model, info


def inference_context(device: str = "cuda"):
    """inference_mode + bf16 autocast on CUDA."""
    import torch

    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    if device == "cuda":
        stack.enter_context(torch.amp.autocast("cuda", dtype=torch.bfloat16))
    return stack


def make_processor(model, device: str, confidence_threshold: float):
    from sam3.model.sam3_image_processor import Sam3Processor

    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def detect(processor, image, prompt_list, frame: str = "photo"):
    """Run every (class, prompt) on one image; return raw Detections (numpy masks)."""
    from .postprocess import Detection

    state = processor.set_image(image)
    out = []
    try:
        for cls, text in prompt_list:
            res = processor.set_text_prompt(state=state, prompt=text)
            masks, scores = res.get("masks"), res.get("scores")
            if masks is None or len(masks) == 0:
                continue
            masks = masks.squeeze(1).cpu().numpy()
            for m, s in zip(masks, scores.float().cpu().tolist()):
                out.append(Detection(cls=cls, prompt=text, score=float(s), mask=m, frame=frame))
    finally:
        del state
    return out
