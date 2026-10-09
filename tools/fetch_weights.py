"""Download the SAM 3.1 checkpoint without putting the token anywhere it can leak
(replaces ingest/clone_model.ps1 and ingest/win_fetch.py, audit A1).

The token is read from the HF_TOKEN environment variable and passed to the
Hugging Face client; it never appears in a URL, a command line, git config or
a log. The file is verified against the published sha256 after download.

    (set HF_TOKEN as a Windows User environment variable, then open a new terminal)
    .venv-win/Scripts/python.exe tools/fetch_weights.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.model import CHECKPOINT, verify_checkpoint  # noqa: E402

REPO_ID = "facebook/sam3.1"
REVISION = "daa63191845a41281374e725f4c9e51c7a824460"


def main() -> int:
    target = ROOT / CHECKPOINT
    ok, detail = verify_checkpoint(target, full=True) if target.exists() else (False, "missing")
    if ok:
        print(f"{target} already present: {detail}")
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set in this shell; see the docstring.")
        return 2
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT.name, revision=REVISION,
                           local_dir=target.parent, token=token)
    ok, detail = verify_checkpoint(Path(path), full=True)
    print(f"downloaded {path}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
