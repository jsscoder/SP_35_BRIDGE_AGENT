---
name: gpu-batch-day
description: Prepare, run and close a booked day on a company GPU (RTX 3060 or RTX 5070 Ti) so the rare window is fully used — batch SAM 3.1 inference, pre-labelling, prompt sweeps, evaluation runs or Cosmos 3 spikes.
---

# GPU batch day

## Before the day (on the laptop)
- Freeze the job list in `jobs/<yyyy-mm-dd>_<gpu>.jsonl`, one JSON object per job: `id`, `type`, `inputs`, `config_hash`, `expected_minutes`, `vram_budget_gb`.
- Build and tag the worker image `sp35-worker:<git-sha>` (CUDA 12.8, PyTorch cu128; the 5070 Ti needs sm_120 support). Smoke-test it on 2 photos.
- Copy inputs to the encrypted SSD with `manifest.sha256`.
- Order jobs: anything a decision depends on first, then longest first.

## On the day
- Check `nvidia-smi`, then run the jobs file with `docker run --gpus all ...`. The runner must resume after an interruption.
- Every 30 minutes append progress, VRAM peak and errors to `jobs/<yyyy-mm-dd>_<gpu>.log.md`.
- If a job fails twice, skip it and continue. Keep the GPU busy.

## After
- Copy results back and verify checksums.
- Run the evaluation harness locally.
- Write a short summary: what ran, what failed, what needs re-booking.
