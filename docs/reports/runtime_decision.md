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
