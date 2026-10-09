# Runtime decision (Task 0.3)

Date: 2026-10-07. GPU: NVIDIA GeForce RTX 2050 4GB, driver 616.92,
CUDA UMD 13.4 (Windows `nvidia-smi`); 313MiB already used at idle.
WSL2: Ubuntu + docker-desktop both Running (v2); WSL sees same GPU
(`nvidia-smi` inside WSL: RTX 2050, 4096MiB, 616.92).

## Checks
- Windows: uv-managed CPython 3.12.13 available. Torch not installed.
  Native Windows given max 1h per plan; SAM 3 recommendation + known
  Triton-on-Windows issues favour WSL2, so no full native install attempted.
- WSL2 Ubuntu: system python 3.14.4; torch not installed yet.
  GPU passthrough confirmed working.

## Decision (updated 2026-10-07, user direction)
Standard: **native Windows + Python 3.12 (`sp35-bridge-ai/.venv-win`) + PyTorch 2.11 cu128 (CUDA 12.8)**
for the GPU worker. WSL2 Ubuntu venv (`~/sp35wsl`, torch 2.11 cu128, CUDA True)
kept as fallback. Gated access to `facebook/sam3.1` verified OK (sha daa63191845a);
token injected per-command from User registry, never logged.

## Next (Task 0.4)
In WSL2: create `uv` venv with Python 3.12 + `torch>=2.7` cu128,
then run `vram-probe` skill (10 photos x all prompts bf16 + 1 tiled pass,
pass <= 3.4GB). Fallbacks per skill if OOM.
No tokens or SP:35 values in this report.

## Update — 8 Oct 2026 (audit remediation)
- **Environment pinned.** The working `.venv-win` (Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0+cu128, sam3 0.1.4, triton-windows 3.8.0.post29, 41 packages) is frozen in `requirements/vision-win.txt`; the file resolves cleanly with `uv pip compile`. Rebuild with `tools/setup_vision_env.ps1` (refuses to touch an existing env without `-Force`). Check with `tools/env_check.py`.
- **Loader.** `vision/model.py` casts only the floating-point weights of the vision trunk and text encoder to bf16 and keeps the 32 complex64 RoPE buffers; it asserts the known checkpoint mismatch (4 missing `convs.3.*` tensors, discarded by `scalp=1`; 36 unused tracker-neck tensors). Result on the RTX 2050: load 1.75 GiB, peak 2.57 GB, ~16.5 s per photo for 87 prompts (`docs/reports/vram_probe_rtx2050_2026-10-08.md`).
- **No silent shared-memory paging.** The loader caps the process below the free VRAM, so an overflow raises OutOfMemoryError. Optional extra safeguard (person to set): NVIDIA Control Panel → Manage 3D settings → Program settings → `.venv-win\Scripts\python.exe` → "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".
- **Weights.** `tools/fetch_weights.py` downloads `sam3.1_multiplex.pt` at revision `daa63191845a` with the token from `HF_TOKEN` (never in a URL) and verifies sha256 `0567debe…cb6`. The old `git clone` with the token in the URL is removed.
- **Supply chain.** `triton-windows` is a community build, pinned to an exact version; review before upgrading.
- **sam3 vs SAM 3.1.** `sam3` 0.1.4 predates SAM 3.1 (27 Mar 2026). Image detection is unaffected (verified), but video tracking (Phase 6) needs the official repo at a later commit, pinned by SHA.
