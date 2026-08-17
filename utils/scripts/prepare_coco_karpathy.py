"""Build the COCO Karpathy-test 1-to-1 retrieval set (paper section 6.2).

Reads the `yerevann/coco-karpathy` test parquet (5000 val2014 images, ~5 captions
each), picks EXACTLY ONE caption per image (rng seeded), and writes:

  datasets/coco/coco_karpathy_test_manifest.json   ordered records:
      {"row": i, "cocoid": ..., "filename": ..., "caption": ..., "caption_idx": ...}
  utils/text_descriptions/coco_karpathy_test.txt   one caption per line, SAME order
      as the image manifest -> image i and text i are the positive pair.

The image activation run (--dataset coco) iterates the manifest in order with
label = row index, so `{dataset}_labels_*` of both towers are aligned by construction.

Usage:
  python -m utils.scripts.prepare_coco_karpathy [--seed 69]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def get_args_parser():
    p = argparse.ArgumentParser("Prepare COCO Karpathy test split", add_help=False)
    p.add_argument("--seed", default=69, type=int, help="rng seed for the 1-caption-per-image choice")
    p.add_argument("--coco_dir", default="./datasets/coco", type=str)
    p.add_argument("--text_dir", default="./utils/text_descriptions", type=str)
    return p


def main(args):
    parquet = hf_hub_download("yerevann/coco-karpathy", "data/test-00000-of-00001.parquet",
                              repo_type="dataset")
    df = pd.read_parquet(parquet)
    assert len(df) == 5000 and (df["filepath"] == "val2014").all()

    rng = np.random.default_rng(args.seed)
    records, captions = [], []
    for row, rec in enumerate(df.itertuples(index=False)):
        sents = [" ".join(str(s).split()) for s in rec.sentences]  # collapse whitespace/newlines
        cap_idx = int(rng.integers(len(sents)))
        records.append({"row": row, "cocoid": int(rec.cocoid), "filename": rec.filename,
                        "caption": sents[cap_idx], "caption_idx": cap_idx})
        captions.append(sents[cap_idx])

    os.makedirs(args.coco_dir, exist_ok=True)
    manifest = os.path.join(args.coco_dir, "coco_karpathy_test_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"seed": args.seed, "records": records}, f, indent=1)

    txt = os.path.join(args.text_dir, "coco_karpathy_test.txt")
    with open(txt, "w") as f:
        f.write("\n".join(captions) + "\n")

    # sanity: images present on disk (if val2014 already extracted)
    img_dir = os.path.join(args.coco_dir, "val2014")
    if os.path.isdir(img_dir):
        missing = [r["filename"] for r in records
                   if not os.path.exists(os.path.join(img_dir, r["filename"]))]
        print(f"images missing on disk: {len(missing)}")
        assert not missing, missing[:5]
    else:
        print(f"[warn] {img_dir} not extracted yet; rerun to verify images")

    print(f"wrote {manifest} ({len(records)} pairs) and {txt}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
