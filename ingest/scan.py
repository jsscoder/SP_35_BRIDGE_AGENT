from pathlib import Path
import hashlib, csv
from PIL import Image
import imagehash
src=Path("C:/Users/omkar/Desktop/sp-35 reasoner/yugmi_images")
out=Path("data/pilot"); out.mkdir(parents=True,exist_ok=True)
probe=Path("data/pilot/probe10"); probe.mkdir(parents=True,exist_ok=True)
files=sorted(src.iterdir())
rows=[]
seen_sha={}
for f in files:
    h=hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        dh=str(imagehash.dhash(Image.open(f).convert("L"),hash_size=8))
    except Exception as e:
        dh="ERR"
    dup_sha=seen_sha.get(h)
    rows.append((f.name,f.suffix.lower(),f.stat().st_size,h[:12],dh,"EXACT-DUPE of "+dup_sha if dup_sha else ""))
    if not dup_sha: seen_sha[h]=f.name
# HEIC twin marking
names=set(r[0] for r in rows)
for i,r in enumerate(rows):
    n,ext,sz,sh,dh,reason=list(r)
    if ".HEIC" in n and not reason:
        rows[i]=(n,ext,sz,sh,dh,"HEIC-conversion twin (keep earliest original)")
    elif n.endswith("_1.jpg") or n.endswith("_1.JPG") or n.endswith("_1.png"):
        if not reason: rows[i]=(n,ext,sz,sh,dh,"resaved _1 copy (review)")
# DJI thermal + screenshot heuristic
for i,r in enumerate(rows):
    n,ext,sz,sh,dh,reason=list(r)
    if n.startswith("DJI_") and not reason:
        rows[i]=(n,ext,sz,sh,dh,"DJI thermal frame (non-bridge? review)")
with open(out/"manifest.csv","w",newline="",encoding="utf-8") as fo:
    w=csv.writer(fo); w.writerow(["file","ext","bytes","sha12","dhash","exclude_reason"])
    w.writerows(rows)
kept=[r for r in rows if not r[5]]
print(f"total={len(rows)} kept_prelim={len(kept)} excluded_prelim={len(rows)-len(kept)}")
# probe10: first 10 kept copied
import shutil
for r in kept[:10]:
    shutil.copy2(src/r[0], probe/r[0])
print("probe10:",len(list(probe.glob('*'))))
# tags skeleton
with open(out/"tags.csv","w",newline="",encoding="utf-8") as fo:
    w=csv.writer(fo); w.writerow(["file","span","component","note"])
    for r in kept: w.writerow([r[0],"","",""])
print("tags rows:",len(kept))
