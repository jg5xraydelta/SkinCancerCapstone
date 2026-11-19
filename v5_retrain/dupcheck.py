"""Find exact and near-duplicate images within/between train and test (perceptual hash)."""
import os, sys, hashlib, json
from PIL import Image
import imagehash
root = sys.argv[1]
recs = []
for split in ["train", "test"]:
    for cls in ["benign", "malignant"]:
        d = os.path.join(root, "data", split, cls)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            im = Image.open(p).convert("RGB")
            recs.append(dict(split=split, cls=cls, f=f, md5=hashlib.md5(open(p,'rb').read()).hexdigest(),
                             ph=imagehash.phash(im), size=im.size))
print("images:", len(recs), "sizes:", {str(r['size']) for r in recs})
# exact
from collections import defaultdict
by = defaultdict(list)
for r in recs: by[r['md5']].append(r)
ex = [v for v in by.values() if len(v) > 1]
print("exact-duplicate groups:", len(ex))
def kind(g):
    s = {r['split'] for r in g}; c = {r['cls'] for r in g}
    return ("cross-split" if len(s) > 1 else "within-" + g[0]['split']) + (", CONFLICTING LABELS" if len(c) > 1 else "")
from collections import Counter
print(Counter(kind(g) for g in ex))
# near duplicates: phash distance <= 4
tr = [r for r in recs if r['split']=='train']; te = [r for r in recs if r['split']=='test']
near = []
for a in te:
    best = min(tr, key=lambda b: a['ph'] - b['ph'])
    d = a['ph'] - best['ph']
    if d <= 4: near.append((a["cls"], a["f"], best["cls"], best["f"], int(d)))
print("test images with a near-duplicate in train (phash<=4):", len(near))
print(Counter((n[0], n[2]) for n in near))
print(near[:10])
json.dump({"exact_groups": [[(r['split'], r['cls'], r['f']) for r in g] for g in ex],
           "near": near}, open("/home/claude/scripts/dups.json", "w"))
