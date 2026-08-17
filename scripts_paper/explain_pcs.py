"""Label the PC units the Waterbirds search added or dropped, in both directions.

For every selected unit (tower, layer, component, PC index) this recovers the PC direction, then
reports the top-M and bottom-M cosine-similar text descriptions and images - the PCLens labelling,
applied to the units the loss-driven search actually chose.

Both poles are reported because a PC is an AXIS: the two ends carry different concepts and only
their combination explains what keeping or removing the unit does.

  python -m scripts_paper.explain_pcs --model ViT-L-14 --variant forward --top 15
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task
from scripts_paper.exp_waterbirds import TASK
from scripts_paper.exp_waterbirds_pc import build_units
from scripts_paper.pclens_core import ACT_DIR, RES_DIR

TEXT_EMB = "top_1500_nouns_5_sentences_imagenet_clean"
TEXT_TXT = "utils/text_descriptions/top_1500_nouns_5_sentences_imagenet_clean.txt"


def load_probes(model, device, seed=69):
    te = torch.from_numpy(np.load(os.path.join(ACT_DIR, f"{TEXT_EMB}_{model}.npy"))).to(device)
    sents = [l.strip() for l in open(TEXT_TXT)]
    ie = torch.from_numpy(np.load(os.path.join(
        ACT_DIR, f"imagenet_embeddings_{model}_seed_{seed}.npy"))).to(device)
    meta = json.load(open(os.path.join(ACT_DIR, f"imagenet_idx_to_class_seed_{seed}.json")))
    names = [r["class_name"] for r in meta]
    # mean-centre the image side: it compensates the modality gap, as in PCLens
    return (torch.nn.functional.normalize(te, dim=1), sents,
            torch.nn.functional.normalize(ie - ie.mean(0), dim=1), names)


def describe(p, te, sents, ie, names, m=5):
    ts = te @ p
    ims = ie @ p
    ti = ts.argsort(descending=True)
    ii = ims.argsort(descending=True)
    return dict(
        text_pos=[sents[j] for j in ti[:m].tolist()],
        text_neg=[sents[j] for j in ti[-m:].flip(0).tolist()],
        img_pos=[names[j] for j in ii[:m].tolist()],
        img_neg=[names[j] for j in ii[-m:].flip(0).tolist()])


def _acts(task, model, device):
    """The activations whose PC bases the search ran over."""
    if task == "bias":
        from scripts_paper.exp_bias_audit import load_audit
        av, mv_, at, mt, _, _ = load_audit(model, device)
        return av, mv_, at, mt
    av, mv_, at, mt, _ = _load_task(model, TASK, device)
    return av, mv_, at, mt


def main(model, variant, regime, rep, top, m, device, task="waterbirds"):
    stem = "bias_pc_moves" if task == "bias" else "waterbirds_pc_moves"
    mv_path = os.path.join(RES_DIR, "6_2", f"{stem}_{model}.csv")
    if not os.path.exists(mv_path):
        print(f"[missing] {mv_path} - run the PC-level experiment first")
        return
    mvs = pd.read_csv(mv_path)
    mvs = mvs[(mvs.variant == variant) & (mvs.regime == regime) & (mvs.rep == rep)]
    if mvs.empty:
        print(f"[empty] no moves for {variant}/{regime}/rep{rep}")
        return

    av, mv_, at, mt = _acts(task, model, device)
    units, alpha, P, _ = build_units(model, av, mv_, at, mt, 8, device)
    te, sents, ie, names = load_probes(model, device)

    for action in ("add", "drop"):
        sub = mvs[mvs.action == action].head(top)
        if sub.empty:
            continue
        print(f"\n{'=' * 100}\n{model}  {variant}/{regime}/rep{rep}  -  first {len(sub)} "
              f"units {'ADDED (kept)' if action == 'add' else 'DROPPED (mean-ablated)'}"
              f"\n{'=' * 100}")
        for n, r in enumerate(sub.itertuples(), 1):
            key = (r.tower, (r.kind, int(r.layer), int(r.head)))
            if key not in P or int(r.pc) >= P[key].shape[0]:
                continue
            p = P[key][int(r.pc)]
            d = describe(p, te, sents, ie, names, m)
            comp = f"MLP {r.layer}" if r.kind == "mlp" else f"layer {r.layer} head {r.head}"
            print(f"\n[{n:2d}] {r.tower.upper():6s} {comp:22s} PC{r.pc}   "
                  f"(step {r.step}, loss {r.loss:.4f})")
            print(f"     + texts : {' | '.join(d['text_pos'])}")
            print(f"     + images: {', '.join(d['img_pos'])}")
            print(f"     - texts : {' | '.join(d['text_neg'])}")
            print(f"     - images: {', '.join(d['img_neg'])}")


def get_args_parser():
    p = argparse.ArgumentParser("explain selected PCs", add_help=False)
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--variant", default="forward", choices=["forward", "backward"])
    p.add_argument("--task", default="waterbirds", choices=["waterbirds", "bias"])
    p.add_argument("--regime", default="class", choices=["class", "group", "bias"])
    p.add_argument("--rep", default=0, type=int)
    p.add_argument("--top", default=15, type=int, help="how many moves to label")
    p.add_argument("--m", default=5, type=int, help="descriptions per pole")
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    main(a.model, a.variant, a.regime, a.rep, a.top, a.m, a.device, a.task)
