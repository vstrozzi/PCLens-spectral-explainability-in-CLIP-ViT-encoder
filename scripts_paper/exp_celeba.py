"""CelebA blond x gender - the canonical spurious correlation, and the control VisoGender was not.

    task      hair colour (blond / dark)   <- retain
    shortcut  gender                       <- remove

Blond hair is overwhelmingly female in CelebA, so a model that keys on gender scores well overall
and fails on blond men. Worst-group accuracy over the four (hair, gender) cells is the metric that
moves; total accuracy is reported beside it because a method can lift the worst cell by flattening
a strong one.

This is the same structure as Waterbirds with a demographic attribute in place of a background, run
by the same four selectors with no per-dataset tuning - which is the transfer claim.

`gender_readout` is the non-degeneracy check: if the gap closes while gender stays fully decodable
from the embeddings, the intervention re-weighted rather than removed. On VisoGender that readout
never moved, which is what identified it as a task-fit failure rather than a method failure.

  python -m scripts_paper.exp_celeba --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/celeba_{eval,trace}_{model}.csv
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
DS = "celeba"
IMG_DIR = os.path.join(ROOT, "datasets", DS)
GENDER_PROMPTS = ["a photo of a woman", "a photo of a man"]      # fixed, not tuned
CELL = {(0, 0): "dark_female", (0, 1): "dark_male",
        (1, 0): "blond_female", (1, 1): "blond_male"}


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_celeba(model, device, seed=69):
    av = _npy(f"{DS}_attn_{model}_seed_{seed}.npy", device)
    mv = _npy(f"{DS}_mlp_{model}_seed_{seed}.npy", device)
    at = _npy(f"{DS}_classnames_attn_text_{model}_seed_{seed}.npy", device)
    mt = _npy(f"{DS}_classnames_mlp_text_{model}_seed_{seed}.npy", device)
    meta = sorted(json.load(open(os.path.join(IMG_DIR, "meta.json"))),
                  key=lambda m: (os.path.dirname(m["file"]), os.path.basename(m["file"])))
    assert len(meta) == av.shape[0], f"{len(meta)} meta rows vs {av.shape[0]} images"
    y = torch.tensor([m["blond"] for m in meta], device=device)
    g = torch.tensor([m["male"] for m in meta], device=device)
    return av, mv, at, mt, y, g


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


@torch.no_grad()
def scores(Ev, Et, y, g, T_gender=None):
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    ok = (L.argmax(1) == y).float()
    out = {"total": float(ok.mean() * 100)}
    for (yy, gg), name in CELL.items():
        m = (y == yy) & (g == gg)
        out[name] = float(ok[m].mean() * 100) if m.any() else float("nan")
    cells = [out[n] for n in CELL.values() if not np.isnan(out[n])]
    out["worst_group"] = min(cells) if cells else float("nan")
    out["worst_class"] = min(float(ok[y == c].mean() * 100) for c in (0, 1) if (y == c).any())
    if T_gender is not None:
        out["gender_readout"] = float(
            (torch.nn.functional.normalize(Ev, dim=1) @ T_gender.T).argmax(1).eq(g)
            .float().mean() * 100)
    return out


def run_model(model, device="cpu", seed=69, n_cal=1, reps=5, n_random=3, max_steps=10 ** 6,
              seed_topk=32, regimes=("class", "group")):
    av, mv, at, mt, y, g = load_celeba(model, device, seed)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    T_gender = _gender_probe(model, device)
    full = scores(embed(av, mv), embed(at, mt), y, g, T_gender)
    print(f"[{model}] {av.shape[0]} images, {len(comps)} components\n"
          f"    FULL MODEL total={full['total']:.2f} worst-group={full['worst_group']:.2f} "
          f"(blond_male={full['blond_male']:.2f})  gender-readout={full['gender_readout']:.2f}",
          flush=True)

    rows = [dict(model=model, regime="-", variant="full model", method="full model", rep=-1,
                 size=len(comps), **full)]
    traces = []
    for regime in regimes:
        strata = y if regime == "class" else (2 * y + g)
        for rep in range(reps):
            gen = torch.Generator().manual_seed(seed + 977 * rep)
            cal = []
            for s in strata.unique():
                w = (strata == s).nonzero(as_tuple=True)[0]
                cal += w[torch.randperm(w.numel(), generator=gen)[:n_cal].to(w.device)].tolist()
            cal = torch.tensor(sorted(cal), device=device)
            held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
            held[cal] = False
            for variant, cfg in VARIANTS.items():
                pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal],
                                        max_steps=max_steps, seed_topk=seed_topk,
                                        verbose=False, **cfg)
                Ev, Et = keep_only(av, mv, at, mt, pool)
                r = scores(Ev[held], Et, y[held], g[held], T_gender)
                base = dict(model=model, regime=regime, variant=variant, rep=rep)
                rows.append(dict(**base, method="selected", size=len(pool), **r))
                traces.append(pd.DataFrame([dict(**base, step=t.step, loss=t.loss,
                                                 n_kept=t.n_kept) for t in tr.itertuples()]))
                print(f"  [{regime} rep{rep} {variant:14s}] kept={len(pool):4d} "
                      f"total={r['total']:.2f} worst-group={r['worst_group']:.2f} "
                      f"gender-readout={r['gender_readout']:.2f}", flush=True)
                rng = np.random.default_rng(seed + rep)
                for q in range(n_random):
                    sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                    Ev, Et = keep_only(av, mv, at, mt, sel)
                    rows.append(dict(**{**base, "rep": q}, method="random pool", size=len(sel),
                                     **scores(Ev[held], Et, y[held], g[held], T_gender)))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"celeba_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"celeba_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote celeba_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("CelebA blond x gender", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=1, type=int, help="calibration images per stratum")
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
