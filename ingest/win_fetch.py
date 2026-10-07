"""Windows Phase-0 runner: cuda check -> gated access check -> hf download.
Token injected per-command from User registry, never printed or logged.
Usage (PowerShell): $env:HF_TOKEN=[Environment]::GetEnvironmentVariable('HF_TOKEN','User'); uv run --with huggingface_hub python ingest/win_fetch.py
"""
import os, sys
tok = os.environ.get("HF_TOKEN", "")
print("hf_token_chars", len(tok))
if len(tok) < 5:
    print("BLOCKED: HF_TOKEN missing in process env. Set with $env:HF_TOKEN=[Environment]::GetEnvironmentVariable('HF_TOKEN','User')")
    sys.exit(2)
try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info("facebook/sam3.1")
    print("gated-access OK:", info.id, "sha:", (info.sha or "")[:12])
except Exception as e:
    print("gated-access FAIL:", type(e).__name__, str(e)[:200])
    sys.exit(3)
print("ready for hf download")
