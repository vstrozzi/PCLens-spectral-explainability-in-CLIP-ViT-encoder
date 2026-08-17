"""VisoGender: occupation is the task, perceived gender is the shortcut.

Third benchmark in the same structure as Waterbirds - retain the class, remove the spurious
attribute - but the shortcut is now demographic and the task is 23-way rather than binary. Same
algorithm, no per-dataset tuning: that transfer is the point.

    task      occupation (23-way)     <- retain
    shortcut  perceived gender        <- remove

**Two reporting decisions forced by the data, both stated rather than buried:**

1. 5 images per (occupation, gender) cell means a per-cell accuracy can only be 0/20/40/60/80/100.
   Worst-group over 46 such cells is therefore very coarse. It is reported, but the HEADLINE is the
   GENDER GAP averaged over occupations - |acc(masculine) - acc(feminine)| - which pools 115 images
   per side and moves smoothly.
2. The set is gender-balanced by construction, so overall accuracy cannot be raised by exploiting a
   class prior. A drop in the gap therefore has to come from the intervention.

`gender_readout` is the diagnostic that separates a real fix from a lucky one: how well the frozen
embeddings still support zero-shot gender classification after the intervention. A method that
closes the gap while leaving gender fully decodable has re-weighted, not removed.

  python -m scripts_paper.exp_visogender --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/visogender_{eval,trace}_{model}.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import keep_only
from scripts_paper.exp_waterbirds import VARIANTS, local_search
from scripts_paper.pclens_core import (ACT_DIR, MODEL_PRETRAINED, MODELS, RES_DIR, ROOT,
                                       component_index, embed, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
DS = "visogender"
IMG_DIR = os.path.join(ROOT, "datasets", DS)
GENDER_PROMPTS = ["a photo of a man", "a photo of a woman"]   # fixed, not tuned


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_visogender(model, device, seed=69):
    av = _npy(f"{DS}_attn_{model}_seed_{seed}.npy", device)
    mv = _npy(f"{DS}_mlp_{model}_seed_{seed}.npy", device)
    at = _npy(f"{DS}_classnames_attn_text_{model}_seed_{seed}.npy", device)
    mt = _npy(f"{DS}_classnames_mlp_text_{model}_seed_{seed}.npy", device)
    meta = json.load(open(os.path.join(IMG_DIR, "meta.json")))
    items = sorted(meta, key=lambda m: (os.path.dirname(m["file"]), os.path.basename(m["file"])))
    assert len(items) == av.shape[0], f"{len(items)} meta rows vs {av.shape[0]} images"
    occs = sorted({m["occupation"].replace(" ", "_") for m in items})
    y = torch.tensor([occs.index(m["occupation"].replace(" ", "_")) for m in items], device=device)
    g = torch.tensor([0 if m["gender"] == "masculine" else 1 for m in items], device=device)
    return av, mv, at, mt, y, g, occs


@torch.no_grad()
def scores(Ev, Et, y, g, T_gender=None):
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    ok = (L.argmax(1) == y).float()
    am = float(ok[g == 0].mean() * 100) if (g == 0).any() else float("nan")
    af = float(ok[g == 1].mean() * 100) if (g == 1).any() else float("nan")
    # per-occupation gap, then averaged: an overall gap can hide occupations that cancel out
    gaps = []
    for c in y.unique():
        m = y == c
        a = ok[m & (g == 0)]
        b = ok[m & (g == 1)]
        if a.numel() and b.numel():
            gaps.append(abs(float(a.mean() - b.mean()) * 100))
    cells = [float(ok[(y == c) & (g == s)].mean() * 100)
             for c in y.unique() for s in (0, 1) if ((y == c) & (g == s)).any()]
    out = dict(total=float(ok.mean() * 100), acc_masc=am, acc_fem=af,
               gap_overall=abs(am - af), gap_per_occ=float(np.mean(gaps)) if gaps else float("nan"),
               worst_cell=min(cells) if cells else float("nan"))
    if T_gender is not None:
        out["gender_readout"] = float(
            (torch.nn.functional.normalize(Ev, dim=1) @ T_gender.T).argmax(1).eq(g).float().mean()
            * 100)
    return out


def _gender_probe(model, device):
    from utils.models.factory import create_model_and_transforms, get_tokenizer
    m, _, _ = create_model_and_transforms(model, pretrained=MODEL_PRETRAINED[model],
                                          precision="fp32")
    m.eval().to(device)
    tok = get_tokenizer(model)
    with torch.no_grad():
        e = m.encode_text(tok(GENDER_PROMPTS).to(device))
    del m
    return torch.nn.functional.normalize(e.float(), dim=1)


def run_model(model, device="cpu", seed=69, n_cal=1, reps=5, n_random=3, max_steps=10 ** 6,
              seed_topk=32):
    av, mv, at, mt, y, g, occs = load_visogender(model, device, seed)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    T_gender = _gender_probe(model, device)
    full = scores(embed(av, mv), embed(at, mt), y, g, T_gender)
    print(f"[{model}] {av.shape[0]} images, {len(occs)} occupations, {len(comps)} components\n"
          f"    FULL MODEL total={full['total']:.2f}%  masc={full['acc_masc']:.2f} "
          f"fem={full['acc_fem']:.2f}  GAP(per-occ)={full['gap_per_occ']:.2f}  "
          f"worst-cell={full['worst_cell']:.2f}  gender-readout={full['gender_readout']:.2f}%",
          flush=True)

    rows = [dict(model=model, variant="full model", method="full model", rep=-1,
                 size=len(comps), **full)]
    traces = []
    for rep in range(reps):
        gen = torch.Generator().manual_seed(seed + 977 * rep)
        # one image per occupation: the one-shot specification, at 23-way scale
        cal = []
        for c in range(len(occs)):
            w = (y == c).nonzero(as_tuple=True)[0]
            if w.numel():
                cal += w[torch.randperm(w.numel(), generator=gen)[:n_cal].to(w.device)].tolist()
        cal = torch.tensor(sorted(cal), device=device)
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal], max_steps=max_steps,
                                    seed_topk=seed_topk, verbose=False, **cfg)
            Ev, Et = keep_only(av, mv, at, mt, pool)
            r = scores(Ev[held], Et, y[held], g[held], T_gender)
            rows.append(dict(model=model, variant=variant, method="selected", rep=rep,
                             size=len(pool), **r))
            traces.append(pd.DataFrame([dict(model=model, variant=variant, rep=rep, step=t.step,
                                             loss=t.loss, n_kept=t.n_kept)
                                        for t in tr.itertuples()]))
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):4d} total={r['total']:.2f} "
                  f"GAP={r['gap_per_occ']:.2f} worst-cell={r['worst_cell']:.2f} "
                  f"gender-readout={r['gender_readout']:.2f}", flush=True)
            rng = np.random.default_rng(seed + rep)
            for q in range(n_random):
                sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                Ev, Et = keep_only(av, mv, at, mt, sel)
                rows.append(dict(model=model, variant=variant, method="random pool", rep=q,
                                 size=len(sel),
                                 **scores(Ev[held], Et, y[held], g[held], T_gender)))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"visogender_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"visogender_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote visogender_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("VisoGender occupation vs gender", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=1, type=int, help="calibration images per occupation")
    p.add_argument("--reps", default=5, type=int)
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.reps, max_steps=a.max_steps,
                  seed_topk=a.seed_topk)
