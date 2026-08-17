"""Render the typographic attack: paste a WRONG class name onto real ImageNet photographs.

CLIP reads text in images and will often prefer the written word to what is depicted - the
"multimodal neuron" failure. That makes it a clean test of whether component selection can separate
the two routes, because each image now supports two defensible answers and only one is correct:

    true label     what the photograph shows
    attack label   the word printed on it

Two numbers follow, and they are not redundant: accuracy on the TRUE label, and the ATTACK SUCCESS
RATE (share of images assigned the printed word). A change that lowers both is the model losing the
task; a change that lowers only the attack rate is the model ignoring the text.

Images are written as an ImageFolder under datasets/typographic/<true_class_idx>/ so the existing
extraction path handles them, with a sidecar json recording the pasted label per file.

  python -m scripts_paper.build_typographic --n 2000
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageNet

from scripts_paper.pclens_core import ROOT

OUT = os.path.join(ROOT, "datasets", "typographic")
CLASSES = os.path.join(ROOT, "utils", "datasets_constants")


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(im, word, rng):
    """White word on a black band, placed over the image - the standard form of the attack."""
    im = im.convert("RGB").resize((336, 336), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    size = max(18, int(336 * 0.11))
    f = _font(size)
    box = d.textbbox((0, 0), word, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    while w > 320 and size > 12:                 # shrink until the word fits the frame
        size -= 2
        f = _font(size)
        box = d.textbbox((0, 0), word, font=f)
        w, h = box[2] - box[0], box[3] - box[1]
    x = rng.randint(4, max(5, 336 - w - 4))
    y = rng.randint(4, max(5, 336 - h - 14))
    d.rectangle([x - 5, y - 4, x + w + 5, y + h + 10], fill=(0, 0, 0))
    d.text((x - box[0], y - box[1]), word, font=f, fill=(255, 255, 255))
    return im


def main(n, seed, data_path, clean=False):
    """`clean=True` renders the SAME images with no word pasted on - the control that says how much
    of the post-intervention gap is the attack and how much is the images being hard.

    Alignment is not re-derived, it is READ from the attacked set's meta.json when that exists. The
    attacked set was built when render() drew from the shared generator, so regenerating the labels
    would silently produce a different pairing - and the attacked activations are already extracted
    for every model, so they are the source of truth."""
    rng = random.Random(seed)          # image order + attack labels
    rng_draw = random.Random(seed + 1)  # placement jitter only, so the two sets stay aligned
    ds = ImageNet(root=os.path.join(data_path, "imagenet"), split="val")
    names = [c[0].split(",")[0].strip() for c in ds.classes]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]

    out_dir = OUT + ("_clean" if clean else "")
    os.makedirs(out_dir, exist_ok=True)
    ref = {}
    ref_p = os.path.join(OUT, "meta.json")
    if clean and os.path.exists(ref_p):
        ref = {int(os.path.basename(m["file"])[:-4]): m for m in json.load(open(ref_p))}
        print(f"aligning to the existing attacked set ({len(ref)} images) - labels come from it")
    meta = []
    for j, i in enumerate(idx):
        im, y = ds[i]
        # the pasted word must name a DIFFERENT class, else the attack has nothing to measure
        a = rng.randrange(len(names) - 1)
        a = a if a < y else a + 1
        if j in ref:                       # the attacked set decides the pairing, not this draw
            y, a = ref[j]["true"], ref[j]["attack"]
        d = os.path.join(out_dir, f"{y:04d}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{j:05d}.jpg")
        # placement uses its own generator, so skipping render() cannot shift the attack labels:
        # the clean set is image-for-image and label-for-label aligned with the attacked one
        img = (im.convert("RGB").resize((336, 336), Image.BICUBIC) if clean
               else render(im, names[a], rng_draw))
        img.save(p, quality=92)
        meta.append(dict(file=os.path.relpath(p, out_dir), true=y, attack=a,
                         true_name=names[y], attack_name=names[a]))
        if (j + 1) % 250 == 0:
            print(f"  {j + 1}/{len(idx)}", flush=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote {len(meta)} images under {out_dir}")
    for m in meta[:5]:
        print(f"    {m['file']}  shows '{m['true_name']}', printed '{m['attack_name']}'")


def get_args_parser():
    p = argparse.ArgumentParser("build typographic attack set", add_help=False)
    p.add_argument("--n", default=2000, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--data_path", default=os.path.join(ROOT, "datasets"))
    p.add_argument("--clean", action="store_true",
                   help="render the same images WITHOUT the pasted word (the control ceiling)")
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.n, a.seed, a.data_path, a.clean)
