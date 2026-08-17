"""Typographic attack: can component selection make CLIP look at the photo instead of the caption?

Every image is a real ImageNet photograph with the name of a DIFFERENT class printed on it, so two
answers are defensible and only one is right. The classifier is the full 1,000 ImageNet class-name
text tower, already extracted - the attack label is one of the 1,000, not a special case.

Two numbers, always reported together:
    acc_true      share assigned the class the photograph shows
    attack_rate   share assigned the class printed on it

They are not complements (998 other classes are available), and only their separation says
anything: a selector that drops both is destroying the model, one that drops `attack_rate` while
holding `acc_true` is genuinely suppressing the text-reading route.

The calibration objective is the zero-shot cross-entropy toward the TRUE label on a handful of
images; nothing tells the selector which label was printed.

  python -m scripts_paper.exp_typographic --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/typographic_{eval,trace}_{model}.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import keep_only
from scripts_paper.exp_waterbirds import VARIANTS, local_search
from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, ROOT, component_index, embed,
                                       logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
DS = "typographic"
TEXT = "imagenet_classnames"
IMG_DIR = os.path.join(ROOT, "datasets", "typographic")


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_typo(model, device, seed=69):
    """Activations plus the (true, attack) label pair per image, in ImageFolder order."""
    av = _npy(f"{DS}_attn_{model}_seed_{seed}.npy", device)
    mv = _npy(f"{DS}_mlp_{model}_seed_{seed}.npy", device)
    at = _npy(f"{TEXT}_attn_text_{model}_seed_{seed}.npy", device)
    mt = _npy(f"{TEXT}_mlp_text_{model}_seed_{seed}.npy", device)
    meta = json.load(open(os.path.join(IMG_DIR, "meta.json")))
    # ImageFolder walks sorted class dirs then sorted filenames - reproduce that exact order
    meta = sorted(meta, key=lambda m: (os.path.dirname(m["file"]), os.path.basename(m["file"])))
    assert len(meta) == av.shape[0], f"{len(meta)} meta rows vs {av.shape[0]} decomposed images"
    y = torch.tensor([m["true"] for m in meta], device=device)
    a = torch.tensor([m["attack"] for m in meta], device=device)
    return av, mv, at, mt, y, a


@torch.no_grad()
def scores(Ev, Et, y, a):
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    p = L.argmax(1)
    return dict(acc_true=float((p == y).float().mean() * 100),
                attack_rate=float((p == a).float().mean() * 100),
                acc_other=float(((p != y) & (p != a)).float().mean() * 100))


def run_model(model, device="cpu", seed=69, n_cal=64, max_steps=10 ** 6, reps=3, n_random=3,
              seed_topk=32):
    av, mv, at, mt, y, a = load_typo(model, device, seed)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    full = scores(embed(av, mv), embed(at, mt), y, a)
    print(f"[{model}] {av.shape[0]} attacked images, {at.shape[0]} classes, {len(comps)} "
          f"components\n    FULL MODEL true={full['acc_true']:.2f}%  "
          f"ATTACK={full['attack_rate']:.2f}%  neither={full['acc_other']:.2f}%", flush=True)

    rows = [dict(model=model, variant="full model", method="full model", rep=-1,
                 size=len(comps), **full)]
    traces = []
    for rep in range(reps):
        g = torch.Generator().manual_seed(seed + rep)
        cal = torch.randperm(av.shape[0], generator=g)[:n_cal].to(device)
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal], max_steps=max_steps,
                                    seed_topk=seed_topk, verbose=False, **cfg)
            Ev, Et = keep_only(av, mv, at, mt, pool)
            r = scores(Ev[held], Et, y[held], a[held])
            rows.append(dict(model=model, variant=variant, method="selected", rep=rep,
                             size=len(pool), **r))
            traces.append(pd.DataFrame([dict(model=model, variant=variant, rep=rep,
                                             step=t.step, loss=t.loss, n_kept=t.n_kept)
                                        for t in tr.itertuples()]))
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):4d}/{len(comps)} "
                  f"true={r['acc_true']:.2f}%  ATTACK={r['attack_rate']:.2f}%", flush=True)
            rng = np.random.default_rng(seed + rep)
            for k in range(n_random):
                sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                Ev, Et = keep_only(av, mv, at, mt, sel)
                rows.append(dict(model=model, variant=variant, method="random pool", rep=k,
                                 size=len(sel), **scores(Ev[held], Et, y[held], a[held])))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"typographic_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"typographic_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote typographic_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("typographic attack selection", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=64, type=int, help="calibration images (labels only)")
    p.add_argument("--max_steps", default=10 ** 6, type=int,
                   help="cap; default = run to convergence")
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.max_steps, a.reps, seed_topk=a.seed_topk)
