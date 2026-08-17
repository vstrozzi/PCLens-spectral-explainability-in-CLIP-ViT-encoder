"""Counting, 1 to 10, on rendered scenes with exact ground truth.

COCO's own captions cannot carry this: only 543 of 5,000 contain a number word and 70% of those are
"two", so counts above four are essentially absent. The set is therefore composed - n instances of
one object type on a plain canvas, n balanced over 1..10.

The objects are RENDERED shapes, not pasted photographs, and that choice is forced. Pasting whole
ImageNet crops does not give a countable scene: a "banana" val image is a market shelf holding
dozens of them, so an image labelled "one banana" shows a pile and the ground truth is simply
wrong. Rendered shapes are the only way to make the count exact, which is the one property this
task cannot do without.

The cost is that the images sit outside CLIP's training distribution, so the BASELINE has to be
read before anything else: if the full model is at chance (10%), the set says nothing about
counting and no selector result on it should be believed. That number is reported first.

Layout is a jittered grid, so instances seldom occlude and the task stays about counting rather
than segmentation. Text side is `a photo of {number word} {colour} {shape}s` for every
(object, count) pair, so the count can be scored with the object held fixed - chance exactly 10%.

  python -m scripts_paper.build_counting --n 2000
Writes datasets/counting/<count>/ + meta.json, and the class-name file for the text tower.
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

from scripts_paper.pclens_core import ACT_DIR, ROOT

OUT = os.path.join(ROOT, "datasets", "counting")
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
CANVAS = 336
COLORS = {"red": (211, 47, 47), "blue": (25, 88, 210), "green": (46, 145, 62),
          "yellow": (240, 190, 30), "black": (28, 28, 30)}
SHAPES = ["circle", "square", "triangle", "star"]
# object = (colour, shape); the count is what varies within an object
CLASSES = [f"{c} {s}" for c in COLORS for s in SHAPES]


def _draw(d, shape, box, fill):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    elif shape == "triangle":
        d.polygon([(x0 + (x1 - x0) / 2, y0), (x0, y1), (x1, y1)], fill=fill)
    else:
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
        pts = []
        for k in range(10):
            import math
            a = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=fill)


def compose(obj, n, rng):
    """n instances of one (colour, shape) on a jittered grid."""
    colour, shape = obj.split(" ", 1)
    im = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 243))
    d = ImageDraw.Draw(im)
    cols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 9 else 4))
    rows = (n + cols - 1) // cols
    cw, ch = CANVAS // cols, CANVAS // rows
    size = int(min(cw, ch) * rng.uniform(0.55, 0.8))
    slots = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    for k in range(n):
        c, r = slots[k]
        jx = rng.randint(0, max(0, cw - size))
        jy = rng.randint(0, max(0, ch - size))
        x0, y0 = c * cw + jx, r * ch + jy
        _draw(d, shape, (x0, y0, x0 + size, y0 + size), COLORS[colour])
    return im


def main(n_images, seed, data_path):
    rng = random.Random(seed)
    names = CLASSES
    print(f"using {len(names)} objects: {', '.join(names)}")
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for j in range(n_images):
        cname = names[j % len(names)]                 # balanced over objects
        cnt = 1 + (j // len(names)) % 10              # and balanced over 1..10
        d = os.path.join(OUT, f"{cnt:02d}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{j:05d}.jpg")
        compose(cname, cnt, rng).save(p, quality=95)
        meta.append(dict(file=os.path.relpath(p, OUT), count=cnt, obj=cname,
                         cls=names.index(cname)))
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{n_images}", flush=True)

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(dict(items=meta, classes=names), f)
    # text side: every (object, count) pair; row index = object_index * 10 + (count - 1)
    lines = ["a photo of {} {}{}".format(WORDS[k], c, "" if k == 0 else "s")
             for c in names for k in range(10)]
    with open(os.path.join(ACT_DIR, "counting_classnames.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(meta)} images under {OUT} and {len(lines)} sentences")
    from collections import Counter
    print("count distribution:", dict(sorted(Counter(m["count"] for m in meta).items())))


def get_args_parser():
    p = argparse.ArgumentParser("build counting set", add_help=False)
    p.add_argument("--n", default=2000, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--data_path", default=os.path.join(ROOT, "datasets"))
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.n, a.seed, a.data_path)
