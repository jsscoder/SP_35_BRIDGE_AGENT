from pathlib import Path
import csv
from PIL import Image
import imagehash
src = Path("C:/Users/omkar/Desktop/sp-35 reasoner/yugmi_images")
mani = list(csv.DictReader(open("data/pilot/manifest.csv", encoding="utf-8")))
kept = [r for r in mani if not r["exclude_reason"]]
print("prelim kept", len(kept))
hashes = {}
for r in kept:
    try:
        h = imagehash.dhash(Image.open(src / r["file"]).convert("L"), hash_size=8)
    except Exception:
        continue
    hashes[r["file"]] = h
groups, used = [], set()
for f, h in hashes.items():
    if f in used:
        continue
    g = [f]
    for f2, h2 in hashes.items():
        if f2 in used or f2 == f:
            continue
        if h - h2 <= 8:
            g.append(f2)
    if len(g) > 1:
        groups.append(sorted(g))
        used.update(g)
print("near-dupe groups (hamming<=8):", len(groups))
for g in groups[:25]:
    print(len(g), g[:4])
