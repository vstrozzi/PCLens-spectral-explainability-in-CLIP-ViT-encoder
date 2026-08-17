"""Build matched caption PAIRS on the COCO images we already have activations for.

Each item is (image i, positive caption, negative caption). The two captions differ in exactly one
respect, so picking the wrong one isolates a single known CLIP weakness rather than general
retrieval difficulty. Chance is 50% by construction.

  negation - "a photo of a dog" vs "a photo with no dog", on images that do and do not contain the
             object. Web captions almost never say what is absent, so the word "dog" fires the dog
             concept whichever way the sentence is built.
  count    - the caption's own number word swapped for a different one.
  order    - the two noun phrases around a relation word swapped ("a man riding a horse" ->
             "a horse riding a man"). Bag-of-words matching cannot tell these apart.

Writes utils/text_descriptions/coco_hardneg_{task}.txt (2N lines, positive then negative per item)
and output_dir/.../coco_hardneg_{task}_meta.json with the image index and label per item.

  python -m scripts_paper.build_hardneg --max_items 2000
"""
import argparse
import json
import os
import random
import re

from scripts_paper.pclens_core import ACT_DIR, ROOT

CAPS = os.path.join(ROOT, "utils", "text_descriptions", "coco_karpathy_test.txt")
TXT_DIR = os.path.join(ROOT, "utils", "text_descriptions")

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM_RE = re.compile(r"\b(" + "|".join(NUM) + r")\b", re.I)

# COCO-ish object vocabulary: concrete, visually checkable, common in these captions
OBJECTS = ["dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird",
           "car", "bus", "truck", "train", "motorcycle", "bicycle", "boat", "airplane",
           "pizza", "cake", "sandwich", "banana", "apple", "clock", "umbrella", "kite",
           "laptop", "phone", "television", "toilet", "bed", "couch", "chair", "table"]

# action verbs first: swapping around them keeps the sentence grammatical, so the negative cannot
# be rejected on syntax alone - only the relation direction has changed
RELS = ["riding", "holding", "wearing", "eating", "chasing", "watching", "carrying", "pulling",
        "next to", "in front of", "behind", "under", "above", "on top of", "near", "beside"]


def _articled(w):
    return f"an {w}" if w[0] in "aeiou" else f"a {w}"


def build_negation(caps, rng, max_items):
    """Balanced present/absent items for the same object word."""
    low = [c.lower() for c in caps]
    items = []
    for obj in OBJECTS:
        pat = re.compile(rf"\b{obj}s?\b")
        present = [i for i, c in enumerate(low) if pat.search(c)]
        # "absent" must be safely absent: no mention of the object or its plural anywhere
        absent = [i for i, c in enumerate(low) if not pat.search(c)]
        if len(present) < 5:
            continue
        n = min(len(present), 40)
        rng.shuffle(present)
        pick_abs = rng.sample(absent, n)
        pos_txt = f"a photo of {_articled(obj)}"
        neg_txt = f"a photo with no {obj}"
        for i in present[:n]:
            items.append(dict(img=i, pos=pos_txt, neg=neg_txt, label=1, group=obj))
        for i in pick_abs:
            # for an image WITHOUT the object the correct caption is the negated one
            items.append(dict(img=i, pos=neg_txt, neg=pos_txt, label=0, group=obj))
    rng.shuffle(items)
    return items[:max_items]


def build_count(caps, rng, max_items):
    items = []
    for i, c in enumerate(caps):
        m = NUM_RE.search(c)
        if not m:
            continue
        w = m.group(1).lower()
        alts = [k for k in NUM if k != w]
        alt = rng.choice(alts)
        neg = c[:m.start()] + (alt.capitalize() if m.group(1)[0].isupper() else alt) + c[m.end():]
        items.append(dict(img=i, pos=c, neg=neg, label=1, group=w))
    rng.shuffle(items)
    return items[:max_items]


def build_order(caps, rng, max_items):
    """Swap the noun phrases either side of a relation word."""
    items = []
    for i, c in enumerate(caps):
        cl = c.rstrip(". ")
        for rel in RELS:
            m = re.search(rf"^(.{{4,}}?)\s+{re.escape(rel)}\s+(.{{4,}})$", cl, re.I)
            if not m:
                continue
            a, b = m.group(1).strip(), m.group(2).strip()
            # keep both sides short, else the swap reads as a garbled sentence and the model can
            # reject it on fluency rather than on the relation
            if a.lower() == b.lower() or len(a.split()) > 6 or len(b.split()) > 6:
                continue
            a, b = a[0].lower() + a[1:], b[0].lower() + b[1:]
            items.append(dict(img=i, pos=cl[0].lower() + cl[1:], neg=f"{b} {rel} {a}",
                              label=1, group=rel))
            break
    rng.shuffle(items)
    return items[:max_items]


BUILDERS = dict(negation=build_negation, count=build_count, order=build_order)


def main(tasks, max_items, seed):
    caps = [l.strip() for l in open(CAPS)]
    for task in tasks:
        rng = random.Random(seed)
        items = BUILDERS[task](caps, rng, max_items)
        if not items:
            print(f"[{task}] no items built")
            continue
        lines = []
        for it in items:
            lines += [it["pos"], it["neg"]]
        p = os.path.join(TXT_DIR, f"coco_hardneg_{task}.txt")
        with open(p, "w") as f:
            f.write("\n".join(l.replace("\n", " ") for l in lines) + "\n")
        os.makedirs(ACT_DIR, exist_ok=True)
        with open(os.path.join(ACT_DIR, f"coco_hardneg_{task}_meta.json"), "w") as f:
            json.dump(items, f)
        pos = sum(it["label"] for it in items)
        print(f"[{task}] {len(items)} items ({pos} label=1), {len(lines)} sentences -> {p}")
        for it in items[:3]:
            print(f"    img {it['img']:4d} [{it['group']}]  + {it['pos']}\n"
                  f"                        - {it['neg']}")


def get_args_parser():
    p = argparse.ArgumentParser("build hard-negative caption pairs", add_help=False)
    p.add_argument("--tasks", nargs="+", default=list(BUILDERS))
    p.add_argument("--max_items", default=2000, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.tasks, a.max_items, a.seed)
