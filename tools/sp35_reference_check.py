"""Independent oracle for IRC:SP:35-2024 Chapter 7.

Recomputes the book's worked examples from the printed inputs, with its own
copy of the weights, so the engine (sp35/) and the knowledge base (kb/) can be
checked against something that shares no code with them.

    uv run --with pyyaml python tools/sp35_reference_check.py

Every line must end in PASS. Lines marked INFO are reference values only.
If PyYAML or the kb/ files are missing, the KB cross-checks are skipped.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

S_K = {"I": 0.0, "II": 0.25, "III": 0.5, "IV": 0.75, "V": 1.0}

# Table 7.1 (pp. 102-105): main weight and sub-component weights (column 3 and column 5)
TABLE_7_1 = {
    "C1": (3.6, {"C1.1": 16.0, "C1.2": 5.8, "C1.3": 12.9, "C1.4": 15.2, "C1.5": 9.7, "C1.6": 9.9,
                 "C1.7": 10.3, "C1.8": 20.1}),
    "C2": (18.0, {"C2.1": 14.2, "C2.2": 11.1, "C2.3": 5.2, "C2.4": 4.5, "C2.5": 7.1, "C2.6": 9.0,
                  "C2.7": 21.8, "C2.8": 14.6, "C2.9": 12.5}),
    "C3": (5.8, {"C3.1": 4.0, "C3.2": 13.0, "C3.3": 8.4, "C3.4": 15.0, "C3.5": 7.6, "C3.6": 12.0,
                 "C3.7": 11.3, "C3.8": 14.7, "C3.9": 14.0}),
    "C4": (32.7, {"C4.1": 66.0, "C4.2": 13.9, "C4.3": 20.1}),
    "C5": (21.7, {"C5.1": 35.0, "C5.2": 14.2, "C5.3": 12.7, "C5.4": 15.1, "C5.5": 13.2, "C5.6": 9.8}),
    "C6": (4.1, {"C6.1": 7.9, "C6.2": 7.3, "C6.3": 17.8, "C6.4": 18.7, "C6.5": 4.8, "C6.6": 4.6,
                 "C6.7": 3.7, "C6.8": 7.4, "C6.9": 11.9, "C6.10": 7.6, "C6.11": 4.3, "C6.12": 3.9}),
    "C7": (14.0, {"C7.1": 50.0, "C7.2": 25.4, "C7.3": 24.6}),
}

# Table 7.6 (pp. 133-135), Yamuna bridge. Condition states per noticed distress; [] = Nil, NA or not visible.
# Two printed CS_j values disagree with their own states (E4): guide bunds and light posts.
YAMUNA_STATES = {
    "C1.1": [], "C1.2": ["II"], "C1.3": [], "C1.4": ["V"], "C1.5": [], "C1.6": [], "C1.7": ["II"], "C1.8": ["II"],
    "C2.1": [], "C2.2": [], "C2.3": [], "C2.4": [], "C2.5": [], "C2.6": [], "C2.7": [],
    "C2.8": ["II", "II", "II", "III"], "C2.9": [],
    "C3.1": ["II", "III"], "C3.2": ["II"], "C3.3": [], "C3.4": [], "C3.5": [], "C3.6": [], "C3.7": [], "C3.8": [],
    "C3.9": [],
    "C4.1": [], "C4.2": [], "C4.3": [],
    "C5.1": [], "C5.2": [], "C5.3": [], "C5.4": [], "C5.5": [], "C5.6": [],
    "C6.1": ["II"], "C6.2": [], "C6.3": ["II"], "C6.4": [], "C6.5": [], "C6.6": [], "C6.7": ["II"], "C6.8": ["II"],
    "C6.9": [], "C6.10": [], "C6.11": [], "C6.12": [],
    "C7.1": ["II"], "C7.2": ["II"], "C7.3": [],
}
YAMUNA_PRINTED_CS_OVERRIDES = {"C3.2": 100.0, "C6.7": 50.0}  # as printed in Table 7.6

# Table 7.9 (pp. 139-143): weights and priority ranking values for bridge 1 and bridge 2
TABLE_7_9 = {
    "F1": (4.38, [(16.27, 100, 100), (10.67, 75, 75), (42.85, 75, 75), (30.21, 0, 0)]),
    "F2": (8.67, [(5.78, 0, 0), (30.73, 25, 25), (12.70, 100, 100), (9.92, 0, 0), (40.87, 50, 25)]),
    "F3": (5.33, [(15.15, 25, 25), (20.22, 75, 75), (8.29, 50, 50), (24.63, 0, 0), (31.69, 0, 0)]),
    "F4": (8.05, [(10.48, 50, 25), (30.41, 100, 0), (15.31, 75, 25), (43.81, 75, 75)]),
    "F5": (22.22, [(8.82, 100, 100), (17.63, 50, 50), (73.55, 0, 0)]),
    "F6": (18.58, [(34.47, 50, 50), (11.35, 50, 50), (10.97, 25, 25), (43.21, 75, 75)]),
    "F7": (29.68, [(100.0, 50, 25)]),
    "F8": (3.09, [(100.0, 50, 50)]),
}

failures = 0


def check(label, value, expected, tol):
    global failures
    ok = abs(value - expected) <= tol
    failures += 0 if ok else 1
    print(f"{label:<62} {value:9.2f}  expected {expected:.2f} +/-{tol:<5}  {'PASS' if ok else 'FAIL'}")


def info(label, value):
    print(f"{label:<62} {value:9.2f}  INFO")


def cs(states):
    """Eq. 3."""
    if not states:
        return 100.0
    return sum(100 - 100 * S_K[s] for s in states) / len(states)


def wavg(pairs):
    """Eq. 1, 2, 4, 5."""
    return sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)


def star(bhi):
    for lo, stars in ((80, 5), (60, 4), (40, 3), (20, 2)):
        if bhi > lo:
            return stars
    return 1


def bhi_from(cs_by_sub):
    cis = {c: wavg([(w, cs_by_sub[s]) for s, w in subs.items()]) for c, (_, subs) in TABLE_7_1.items()}
    return wavg([(TABLE_7_1[c][0], ci) for c, ci in cis.items()]), cis


print("== Eq. 3 sub-component scores")
check("p.130 deck slab, states II III II I (book prints 50.0, E1)", cs(["II", "III", "II", "I"]), 75.0, 1e-9)
check("p.132 pier cap, states II II II III", cs(["II", "II", "II", "III"]), 68.75, 1e-9)

print("== Table 7.6 Yamuna bridge BHI")
strict = {s: cs(v) for s, v in YAMUNA_STATES.items()}
as_printed = dict(strict, **YAMUNA_PRINTED_CS_OVERRIDES)
bhi_p, cis_p = bhi_from(as_printed)
bhi_s, cis_s = bhi_from(strict)
for c, printed_ci in {"C1": 75.65, "C2": 95.03, "C3": 98.50, "C4": 100, "C5": 100, "C6": 89.77, "C7": 81.15}.items():
    check(f"CI {c} (printed CS values)", cis_p[c], printed_ci, 0.5)
check("BHI with printed CS values", bhi_p, 95.06, 0.2)
check("BHI scored strictly by state (E4)", bhi_s, 95.06, 0.2)
check("Star rating", star(bhi_s), 5, 0)
critical = sorted(s for s, v in YAMUNA_STATES.items() if "V" in v)
print(f"{'Critical list (state V)':<62} {', '.join(critical):>9}  expected C1.4       {'PASS' if critical == ['C1.4'] else 'FAIL'}")
failures += 0 if critical == ["C1.4"] else 1

print("== Table 7.9 MPI")
for b, expected in ((0, 44.89), (1, 33.21)):
    fr = {f: wavg([(w, p[b]) for w, *p in subs]) for f, (_, subs) in TABLE_7_9.items()}
    check(f"MPI bridge {b + 1} (book PR values)", wavg([(TABLE_7_9[f][0], v) for f, v in fr.items()]), expected, 0.2)
    fr["F7"] = (25, 0)[b]  # E9: F7 from star bands (BHI 75.69 -> Good, 85.66 -> Excellent)
    info(f"MPI bridge {b + 1} with F7 from star bands (E9)", wavg([(TABLE_7_9[f][0], v) for f, v in fr.items()]))


# ---------------- knowledge-base cross-checks ----------------
def parse_interval(text):
    text = str(text).strip()
    if text == "=0":
        return (0.0, 0.0, True, True)
    m = re.fullmatch(r"([\[(])\s*([-\d.]+)\s*,\s*([-\d.]+|inf)\s*([\])])", text)
    if not m:
        raise ValueError(f"bad interval {text!r}")
    lo, hi = float(m.group(2)), math.inf if m.group(3) == "inf" else float(m.group(3))
    return (lo, hi, m.group(1) == "[", m.group(4) == "]")


def contains(iv, x):
    lo, hi, lo_in, hi_in = iv
    return (x > lo or (lo_in and x == lo)) and (x < hi or (hi_in and x == hi))


def check_bands(row_id, bands):
    ivs = {k: parse_interval(v) for k, v in bands.items() if v is not None}
    finite = [b for iv in ivs.values() for b in iv[:2] if b != math.inf]
    top = max(finite) * 1.5 + 1
    xs = sorted({0.0, *finite, *[b + 1e-6 for b in finite], *[b - 1e-6 for b in finite if b > 0],
                 *[i * top / 2000 for i in range(2001)]})
    bad = [x for x in xs if x >= 0 and sum(contains(iv, x) for iv in ivs.values()) != 1]
    return bad[:3]


try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

kb = Path(__file__).resolve().parents[1] / "kb" / "sp35_2024"
if yaml is None or not kb.exists():
    print("== KB cross-checks skipped (install pyyaml and keep kb/ next to tools/)")
else:
    print("== KB cross-checks")
    comp = yaml.safe_load((kb / "components.yaml").read_text(encoding="utf-8"))
    mism = []
    for c in comp["components"]:
        w, subs = TABLE_7_1[c["code"]]
        if c["weight"] != w:
            mism.append(c["code"])
        mism += [s["code"] for s in c["subcomponents"] if subs.get(s["code"]) != s["weight"]]
        if {s["code"] for s in c["subcomponents"]} != set(subs):
            mism.append(c["code"] + " set")
    ok = not mism
    failures += 0 if ok else 1
    print(f"{'components.yaml matches Table 7.1':<62} {len(comp['components']):9d}  {'PASS' if ok else 'FAIL ' + str(mism)}")

    mpi = yaml.safe_load((kb / "mpi.yaml").read_text(encoding="utf-8"))
    mism = []
    for f in mpi["factors"]:
        w, subs = TABLE_7_9[f["code"]]
        if f["weight"] != w or [s["weight"] for s in f["subfactors"]] != [s[0] for s in subs]:
            mism.append(f["code"])
    ok = not mism
    failures += 0 if ok else 1
    print(f"{'mpi.yaml matches Table 7.9 weights':<62} {len(mpi['factors']):9d}  {'PASS' if ok else 'FAIL ' + str(mism)}")

    for name, expected_rows in (("distress_bridge.yaml", 28), ("distress_approach.yaml", 34)):
        data = yaml.safe_load((kb / name).read_text(encoding="utf-8"))
        rows = data["distresses"]
        problems = []
        for r in rows:
            if set(r["printed"]) != {"I", "II", "III", "IV", "V"}:
                problems.append(f"{r['id']} printed keys")
            for crit in r.get("criteria") or [{"bands": r["bands"]}]:
                bad = check_bands(r["id"], crit["bands"])
                if bad:
                    problems.append(f"{r['id']} gap/overlap at {bad}")
        ok = len(rows) == expected_rows and not problems
        failures += 0 if ok else 1
        print(f"{name + ': rows, every value maps to one state':<62} {len(rows):9d}  {'PASS' if ok else 'FAIL ' + str(problems[:4])}")

print("== " + ("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"))
sys.exit(1 if failures else 0)
