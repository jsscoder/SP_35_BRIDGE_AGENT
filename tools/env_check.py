"""Environment check for the GPU worker (replaces ingest/cuda_check.py, torch_info.py,
sam3_check.py, builder_sig.py, first_light.py).

    .venv-win\\Scripts\\python.exe tools\\env_check.py            # quick
    .venv-win\\Scripts\\python.exe tools\\env_check.py --full     # + checkpoint sha256 (~15 s)
"""
import argparse
import os
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also hash the 3.5 GB checkpoint")
    args = ap.parse_args()
    ok = True
    import torch
    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu {torch.cuda.get_device_name(0)}: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB")
    else:
        ok = False
    for pkg in ("sam3", "triton-windows", "pycocotools", "pillow", "pyyaml", "numpy"):
        try:
            print(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"{pkg} MISSING")
            ok = ok and pkg == "triton-windows"  # triton-windows only needed on native Windows
    from vision.model import CHECKPOINT, verify_checkpoint
    good, detail = verify_checkpoint(ROOT / CHECKPOINT, full=args.full)
    print(f"checkpoint {CHECKPOINT}: {detail}")
    ok = ok and good
    print("HF_TOKEN in environment:", "present" if os.environ.get("HF_TOKEN") else "absent (only needed to download)")
    print("ENV", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
