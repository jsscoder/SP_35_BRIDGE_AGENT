"""Phase-0 verification suite: oracle + KB + inference artifacts + repo hygiene."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:
        fails.append(name)
        print(f"FAIL {name}: {e}")


def oracle():
    r = subprocess.run([sys.executable, "tools/sp35_reference_check.py"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert "ALL PASS" in r.stdout, r.stdout[-500:]


def kb_status():
    import yaml
    for f in (ROOT / "kb/sp35_2024").glob("*.yaml"):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        if "status" in d:
            assert d["status"] in ("draft-for-signoff", "signed"), f


def infer_artifacts():
    d = ROOT / "data/pilot/infer_out/IMG_4151.json"
    assert d.exists(), "run vision/infer.py on IMG_4151 first"
    r = json.loads(d.read_text())
    assert r["status"] == "ok" and len(r["detections"]) > 0
    assert (ROOT / "data/pilot/infer_out/IMG_4151_overlay.png").exists()
    for det in r["detections"]:
        assert set(det["rle"]) >= {"counts", "size"}


def probe_report():
    reps = list((ROOT / "docs/reports").glob("vram_probe_*.md"))
    assert reps, "no probe report"
    txt = reps[0].read_text()
    assert "worst" in txt.lower() or "peak" in txt.lower()


def repo_hygiene():
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT)
    listed = r.stdout
    for bad in (".env", ".pt", ".safetensors", "probe10", "infer_out", "weights"):
        assert bad not in listed, bad
    assert "HF_TOKEN" not in subprocess.run(
        ["git", "grep", "-l", "hf_", "--", ":!*.md"], capture_output=True,
        text=True, cwd=ROOT).stdout


for n, f in [("oracle", oracle), ("kb-status", kb_status),
             ("infer-artifacts", infer_artifacts), ("probe-report", probe_report),
             ("repo-hygiene", repo_hygiene)]:
    check(n, f)
print("VERIFY", "ALL PASS" if not fails else f"FAILURES={fails}")
sys.exit(1 if fails else 0)
