"""Phase 0 verification (audit A3). Every check reports PASS, FAIL, SKIP or PENDING.

- FAIL: something is wrong; exit code 1.
- SKIP: the evidence is local-only (data/ is gitignored), e.g. on a CI runner.
- PENDING: waiting on a person (e.g. engineer sign-off). Fails only with --strict,
  which is what the G0 gate uses.

    uv run --with pyyaml python tools/verify_phase0.py            # CI / day to day
    uv run --with pyyaml python tools/verify_phase0.py --strict   # G0 gate
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RX = r"hf_[A-Za-z0-9]{30,}"
results = []


class Skip(Exception):
    pass


class Pending(Exception):
    pass


def check(name, fn):
    try:
        detail = fn() or ""
        results.append(("PASS", name, detail))
    except Skip as e:
        results.append(("SKIP", name, str(e)))
    except Pending as e:
        results.append(("PENDING", name, str(e)))
    except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
        results.append(("FAIL", name, f"{type(e).__name__}: {e}"))


def oracle():
    r = subprocess.run([sys.executable, "tools/sp35_reference_check.py"], capture_output=True,
                       text=True, cwd=ROOT, timeout=300)
    assert "ALL PASS" in r.stdout, r.stdout[-400:]
    return "book examples reproduced; KB weights and bands consistent"


def kb_signed():
    import yaml
    statuses = {}
    for f in sorted((ROOT / "kb/sp35_2024").glob("*.yaml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        if "status" in d:
            assert d["status"] in ("draft-for-signoff", "signed"), f"{f.name}: bad status {d['status']!r}"
            statuses[f.name] = d["status"]
    register = (ROOT / "docs/plan/SP35_CH7_INTERPRETATIONS.md").read_text(encoding="utf-8")
    rows = [line for line in register.splitlines() if line.startswith("| Bridge engineer")]
    signed_register = bool(rows) and len([c for c in rows[0].split("|")[2:5] if c.strip()]) >= 3
    if not all(s == "signed" for s in statuses.values()) or not signed_register:
        raise Pending("E1-E14 not signed by the bridge engineer; KB files still draft-for-signoff")
    return "KB signed"


def probe_report():
    reps = sorted((ROOT / "docs/reports").glob("vram_probe_*.json"))
    assert reps, "no machine-readable probe report (run: python -m vision.probe)"
    rep = json.loads(reps[-1].read_text(encoding="utf-8"))
    v = rep["verdict"]
    assert v["pass_3_4gb"] is True, f"{reps[-1].name}: probe FAILED {v}"
    assert rep.get("tiled_pass") and not rep["tiled_pass"]["oom"], "tiled pass missing or OOM"
    assert (reps[-1].with_suffix(".md")).exists(), "markdown report missing"
    return (f"{reps[-1].name}: peak {v['worst_peak_alloc_gb']} GB <= {v['budget_gb']} GB, "
            f"no spill, no OOM, {v['median_sec_per_photo']} s/photo")


def pilot_set():
    manifest = ROOT / "data/pilot/manifest.csv"
    if not manifest.exists():
        raise Skip("data/pilot is local-only")
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    kept = [r for r in rows if r["keep"] == "yes"]
    assert len(rows) == 383, f"{len(rows)} files in manifest"
    assert all(r["category"] == "iphone" for r in kept), "non-bridge files in the pilot set"
    assert all(r["reason"] for r in rows if r["keep"] == "no"), "an exclusion has no reason"
    assert len(kept) == 167, f"{len(kept)} kept, plan says 167"
    probe = sorted(p.name for p in (ROOT / "data/pilot/probe10").iterdir())
    assert len(probe) == 10 and set(probe) <= {r["file"] for r in kept}, "probe10 must be 10 pilot photos"
    tags = list(csv.DictReader(open(ROOT / "data/pilot/tags.csv", encoding="utf-8")))
    assert len(tags) == len(kept), "tags.csv must list every pilot photo"
    return f"383 -> {len(kept)} kept, every exclusion reasoned, probe10 = 10 pilot photos"


def inference_outputs():
    out = ROOT / "data/pilot/infer_out"
    files = sorted(out.glob("*.json")) if out.exists() else []
    files = [f for f in files if f.name != "run_summary.json"]
    if not files:
        raise Skip("data/pilot/infer_out is local-only")
    n = 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["status"] in ("ok", "ok_reduced"), f"{f.name}: {d['status']}"
        for key in ("components", "distresses"):
            for det in d[key]:
                assert det["sp35"], f"{f.name}: {key} without SP:35 code"
                assert {"counts", "size"} <= set(det["rle"]) and len(det["bbox"]) == 4
                n += 1
        for det in d["distresses"]:
            host = det.get("host", {})
            # the one unhosted case allowed: an orphan kept explicitly for the engineer to place
            orphan = host.get("source") == "orphan_rescue" and det.get("review_required") is True
            assert det["sp35"].startswith("T7.") and (host.get("sp35") or orphan), f"{f.name}: distress unmapped"
        assert (out / f"{f.stem}_overlay.png").exists(), f"overlay missing for {f.stem}"
    return f"{len(files)} photos, {n} detections, all mapped to SP:35"


def repo_hygiene():
    listed = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    assert listed, "not a git checkout"
    bad = [p for p in listed if p.startswith(("data/", ".venv")) or re.search(r"\.(pt|pth|safetensors|bin|onnx|env)$", p)
           or p.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
    assert not bad, f"tracked files that must not be in git: {bad[:5]}"
    hits = subprocess.run(["git", "grep", "-nIE", TOKEN_RX], capture_output=True, text=True, cwd=ROOT).stdout
    assert not hits.strip(), "token-like string in tracked files: " + hits.split(":")[0]
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True,
                               text=True, cwd=ROOT).stdout.splitlines()
    leak = [p for p in untracked if re.search(TOKEN_RX, (ROOT / p).read_text(encoding="utf-8", errors="ignore"))
            if (ROOT / p).is_file() and (ROOT / p).stat().st_size < 2_000_000]
    assert not leak, f"token-like string in untracked files: {leak[:3]}"
    return f"{len(listed)} tracked files; no weights, photos, data or tokens"


def tests_present():
    tests = [p for p in (ROOT / "tests").rglob("test_*.py")]
    assert len(tests) >= 3, "too few test files"
    assert not (ROOT / "tests/test_placeholder.py").exists(), "placeholder test still present"
    return f"{len(tests)} test files"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="PENDING counts as failure (gate review)")
    args = ap.parse_args()
    for name, fn in [("oracle", oracle), ("kb-signed", kb_signed), ("probe-report", probe_report),
                     ("pilot-set", pilot_set), ("inference-outputs", inference_outputs),
                     ("repo-hygiene", repo_hygiene), ("tests-present", tests_present)]:
        check(name, fn)
    for status, name, detail in results:
        print(f"{status:<8} {name:<18} {detail}")
    failed = [n for s, n, _ in results if s == "FAIL" or (args.strict and s == "PENDING")]
    print("VERIFY", "ALL PASS" if not failed else f"FAILURES={failed}",
          f"(skipped {sum(s == 'SKIP' for s, *_ in results)}, pending {sum(s == 'PENDING' for s, *_ in results)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
