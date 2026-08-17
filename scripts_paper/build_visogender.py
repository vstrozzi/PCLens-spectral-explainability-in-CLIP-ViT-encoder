"""Fetch VisoGender as an occupation-classification set with gender as the spurious attribute.

VisoGender (Hall et al., 2023) ships as a TSV of image URLs plus annotator-perceived gender, so the
first thing this does is report what actually downloaded - link rot is the standing risk with any
URL-distributed benchmark, and a silently shrunken cell would corrupt the worst-group metric.

Measured before building (2026-08): OO 230 rows / 100% of a 60-URL sample live, OP 460 rows / 97%.
The OO split is perfectly balanced by construction - 23 occupations x 5 masculine x 5 feminine.

    task      occupation (23-way)          <- retain
    shortcut  perceived gender             <- remove

**Granularity caveat, stated because it limits what the numbers can say:** 5 images per
(occupation, gender) cell means a per-cell accuracy can only take the values 0, 20, 40, 60, 80, 100.
Worst-group over 46 such cells is therefore coarse and noisy; report it with that caveat, and prefer
the gender GAP averaged over occupations as the headline, which pools 115 images per side.

Images land in datasets/visogender/<occupation>/ so the existing ImageFolder extraction path works,
with meta.json carrying the perceived gender per file.

  python -m scripts_paper.build_visogender --splits OO
"""
import argparse
import csv
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from scripts_paper.pclens_core import ROOT

OUT = os.path.join(ROOT, "datasets", "visogender")
BASE = "https://raw.githubusercontent.com/oxai/visogender/main/data/visogender_data"
FILES = {"OO": f"{BASE}/OO/OO_Visogender_10052025.tsv",
         "OP": f"{BASE}/OP/OP_Visogender_11012024.tsv"}
UA = {"User-Agent": "Mozilla/5.0"}
URL_COL = "URL type (Type NA if can't find)"


def _tsv(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8")), delimiter="\t"))


def _fetch(job):
    idx, occ, gender, url, split = job
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        return dict(idx=idx, ok=False, err=str(e)[:80])
    d = os.path.join(OUT, occ.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{idx}.jpg")
    im.save(p, quality=92)
    return dict(idx=idx, ok=True, file=os.path.relpath(p, OUT), occupation=occ,
                gender=gender, split=split)


def main(splits, workers):
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for sp in splits:
        for row in _tsv(FILES[sp]):
            occ = (row.get("Occupation") or "").strip()
            gen = (row.get("Occupation_perceived_gender") or "").strip()
            url = (row.get(URL_COL) or "").strip()
            if occ and gen in ("masculine", "feminine") and url.startswith("http"):
                jobs.append((row["IDX"], occ, gen, url, sp))
    print(f"{len(jobs)} annotated URLs across {len(splits)} split(s)")

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_fetch, jobs))
    ok = [r for r in res if r["ok"]]
    bad = [r for r in res if not r["ok"]]
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(ok, f)

    print(f"downloaded {len(ok)}/{len(jobs)} ({100 * len(ok) / max(len(jobs), 1):.1f}%); "
          f"{len(bad)} failed")
    # the cell census is the number that decides whether worst-group is computable at all
    cells = {}
    for r in ok:
        cells[(r["occupation"], r["gender"])] = cells.get((r["occupation"], r["gender"]), 0) + 1
    occs = sorted({o for o, _ in cells})
    empty = [(o, g) for o in occs for g in ("masculine", "feminine") if cells.get((o, g), 0) == 0]
    thin = [(o, g, n) for (o, g), n in sorted(cells.items()) if n < 3]
    print(f"{len(occs)} occupations; {len(empty)} EMPTY cells; {len(thin)} cells with <3 images")
    if empty:
        print("  empty:", empty[:10])
    if thin:
        print("  thin :", thin[:10])
    print("  per-occupation (m/f):",
          ", ".join(f"{o}:{cells.get((o, 'masculine'), 0)}/{cells.get((o, 'feminine'), 0)}"
                    for o in occs[:12]))
    # class-name file for the text tower, index = sorted occupation order
    names = [o.replace("_", " ") for o in occs]
    from scripts_paper.pclens_core import ACT_DIR
    os.makedirs(ACT_DIR, exist_ok=True)
    with open(os.path.join(ACT_DIR, "visogender_classnames.txt"), "w") as f:
        f.write("\n".join(f"a photo of a {n}" for n in names) + "\n")
    print(f"wrote {len(names)} class prompts")


def get_args_parser():
    p = argparse.ArgumentParser("fetch VisoGender", add_help=False)
    p.add_argument("--splits", nargs="+", default=["OO"], choices=["OO", "OP"])
    p.add_argument("--workers", default=16, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.splits, a.workers)
