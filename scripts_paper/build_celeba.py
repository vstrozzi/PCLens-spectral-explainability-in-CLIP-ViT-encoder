"""CelebA blond-hair x gender - the canonical spurious correlation.

This is the benchmark every group-robustness paper reports (GroupDRO, JTT, DFR), which is the point:
unlike VisoGender it contains a REAL shortcut, and our numbers become directly comparable to
published ones.

    task      hair colour, blond vs not-blond   <- retain
    shortcut  gender                            <- remove

The correlation is in the data, not imposed: blond hair is overwhelmingly female in CelebA, so
"blond" and "female" are confusable and the rare cell - blond men - is where a shortcut-reliant
model fails. That cell is what worst-group accuracy measures.

VisoGender failed for the opposite reason and it is worth recording why: it is gender-BALANCED by
construction, so gender never helps predict occupation and an occupation-only objective has no
shortcut to expose. A spurious correlation has to be present in the calibration signal for a
loss-driven selector to find it.

Sampling is deliberately NOT the natural distribution: with ~2k images the rare cell would hold a
few dozen examples and worst-group would be unmeasurable. Cells are drawn as evenly as the source
allows and the realised census is printed, so the imbalance that remains is visible rather than
assumed.

The parquet shards are read directly rather than through `datasets.load_dataset`, because this
repository has its own top-level `datasets/` directory which shadows the library on the import path.

  python -m scripts_paper.build_celeba --n 2000
Writes datasets/celeba/<label>/ + meta.json and the class-name file for the text tower.
"""
import argparse
import io
import json
import os
from collections import Counter

from scripts_paper.pclens_core import ACT_DIR, ROOT

OUT = os.path.join(ROOT, "datasets", "celeba")
REPO = "flwrlabs/celeba"
CONFIG = "img_align+identity+attr"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{CONFIG}"
CLASSES = ["a photo of a person with dark hair", "a photo of a person with blond hair"]


def _shards(split, k):
    """Stream rows out of the parquet shards, one shard at a time."""
    import urllib.request
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)          # the login cgroup caps processes; pyarrow's pool trips it
    pa.set_io_thread_count(1)
    for i in range(k):
        url = f"{BASE}/{split}-{i:05d}-of-{k:05d}.parquet"
        local = os.path.join("/tmp", f"celeba_{split}_{i}.parquet")
        if not os.path.exists(local):
            print(f"  fetching shard {i}...", flush=True)
            urllib.request.urlretrieve(url, local)
        f = pq.ParquetFile(local)
        for batch in f.iter_batches(batch_size=256,
                                    columns=["image", "Blond_Hair", "Male"],
                                    use_threads=False):
            for row in batch.to_pylist():
                yield row


def main(n, seed, split, n_shards=3):
    ds = _shards(split, n_shards)

    per_cell = max(1, n // 4)
    want = {(y, g): per_cell for y in (0, 1) for g in (0, 1)}
    got = Counter()
    os.makedirs(OUT, exist_ok=True)
    meta = []
    seen = 0
    for row in ds:
        seen += 1
        y = int(bool(row["Blond_Hair"]))
        g = int(bool(row["Male"]))
        if got[(y, g)] >= want[(y, g)]:
            if sum(got.values()) >= sum(want.values()):
                break
            continue
        d = os.path.join(OUT, f"{y:02d}")
        os.makedirs(d, exist_ok=True)
        j = sum(got.values())
        p = os.path.join(d, f"{j:05d}.jpg")
        from PIL import Image
        im = row["image"]
        if isinstance(im, dict):          # parquet stores {bytes, path}
            im = im["bytes"]
        if isinstance(im, (bytes, bytearray)):
            im = Image.open(io.BytesIO(im))
        im.convert("RGB").save(p, quality=92)
        meta.append(dict(file=os.path.relpath(p, OUT), blond=y, male=g))
        got[(y, g)] += 1
        if j and j % 250 == 0:
            print(f"  {j} images ({seen} scanned)", flush=True)
        if seen > 400000:
            break

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(ACT_DIR, "celeba_classnames.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    names = {(0, 0): "dark/female", (0, 1): "dark/male",
             (1, 0): "blond/female", (1, 1): "blond/male"}
    print(f"\nwrote {len(meta)} images from {seen} scanned")
    print("cell census (the rare cell is the one worst-group depends on):")
    for k, lab in names.items():
        print(f"   {lab:14s} {got[k]:5d}")
    thin = [lab for k, lab in names.items() if got[k] < 50]
    if thin:
        print(f"   WARNING: {thin} under 50 images - worst-group will be very coarse there")


def get_args_parser():
    p = argparse.ArgumentParser("build CelebA blond x gender", add_help=False)
    p.add_argument("--n", default=2000, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--split", default="test")
    p.add_argument("--n_shards", default=3, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.n, a.seed, a.split, a.n_shards)
