"""Counting 1..10: can selection recover a count signal CLIP barely uses?

Two scores, and the first one decides whether the second is worth reading:

    count_acc   the object is held FIXED and only the number word varies, so the 10 candidates
                differ in exactly one token. Chance is 10%.
    full_acc    all (object, count) pairs compete at once - a 200-way problem that also tests
                whether the object is recognised at all.

The images are rendered (see build_counting for why that is forced), so they sit outside CLIP's
training distribution. If `count_acc` for the FULL model is at chance, the set cannot support any
claim about counting and the selector numbers on it mean nothing - that is reported, not hidden.

  python -m scripts_paper.exp_counting --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/counting_{eval,trace}_{model}.csv
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
DS = "counting"
IMG_DIR = os.path.join(ROOT, "datasets", DS)


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_counting(model, device, seed=69):
    """Activations plus (object, count) per image, in ImageFolder order."""
    av = _npy(f"{DS}_attn_{model}_seed_{seed}.npy", device)
    mv = _npy(f"{DS}_mlp_{model}_seed_{seed}.npy", device)
    at = _npy(f"{DS}_classnames_attn_text_{model}_seed_{seed}.npy", device)
    mt = _npy(f"{DS}_classnames_mlp_text_{model}_seed_{seed}.npy", device)
    meta = json.load(open(os.path.join(IMG_DIR, "meta.json")))
    items = sorted(meta["items"],
                   key=lambda m: (os.path.dirname(m["file"]), os.path.basename(m["file"])))
    assert len(items) == av.shape[0], f"{len(items)} meta rows vs {av.shape[0]} images"
    obj = torch.tensor([m["cls"] for m in items], device=device)
    cnt = torch.tensor([m["count"] - 1 for m in items], device=device)   # 0..9
    return av, mv, at, mt, obj, cnt, meta["classes"]


@torch.no_grad()
def scores(Ev, Et, obj, cnt):
    """Count accuracy with the object fixed, and the joint 200-way accuracy."""
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    joint = L.argmax(1)
    tgt = obj * 10 + cnt
    # restrict to the 10 rows of the true object: row index = object * 10 + (count - 1)
    rows = (obj.unsqueeze(1) * 10 + torch.arange(10, device=L.device).unsqueeze(0))
    pick = L.gather(1, rows).argmax(1)
    out = dict(count_acc=float((pick == cnt).float().mean() * 100),
               full_acc=float((joint == tgt).float().mean() * 100),
               obj_acc=float((joint.div(10, rounding_mode="floor") == obj).float().mean() * 100),
               mae=float((pick - cnt).abs().float().mean()))
    for c in range(10):
        m = cnt == c
        out[f"count_{c + 1}"] = float((pick[m] == cnt[m]).float().mean() * 100) if m.any() else float("nan")
    return out


def run_model(model, device="cpu", seed=69, n_cal=4, max_steps=10 ** 6, reps=3, n_random=3,
              seed_topk=32):
    av, mv, at, mt, obj, cnt, classes = load_counting(model, device, seed)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    full = scores(embed(av, mv), embed(at, mt), obj, cnt)
    print(f"[{model}] {av.shape[0]} images, {len(classes)} objects x 10 counts, "
          f"{len(comps)} components", flush=True)
    print(f"[{model}] FULL MODEL count_acc={full['count_acc']:.2f}% (chance 10)  "
          f"full_acc={full['full_acc']:.2f}%  obj_acc={full['obj_acc']:.2f}%  "
          f"MAE={full['mae']:.2f}", flush=True)
    if full["count_acc"] < 13:
        print(f"[{model}] NOTE: the full model is at chance on counting - selector numbers on "
              f"this set do not support a claim about counting.", flush=True)

    rows = [dict(model=model, variant="full model", method="full model", rep=-1,
                 size=len(comps), **full)]
    traces = []
    tgt = obj * 10 + cnt
    for rep in range(reps):
        g = torch.Generator().manual_seed(seed + rep)
        cal = []
        for c in range(10):                              # calibrate balanced over the counts
            w = (cnt == c).nonzero(as_tuple=True)[0]
            cal += w[torch.randperm(w.numel(), generator=g)[:n_cal].to(w.device)].tolist()
        cal = torch.tensor(sorted(cal), device=device)
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search(av, mv, at, mt, tau_inv, cal, tgt[cal], max_steps=max_steps,
                                    seed_topk=seed_topk, verbose=False, **cfg)
            Ev, Et = keep_only(av, mv, at, mt, pool)
            r = scores(Ev[held], Et, obj[held], cnt[held])
            rows.append(dict(model=model, variant=variant, method="selected", rep=rep,
                             size=len(pool), **r))
            traces.append(pd.DataFrame([dict(model=model, variant=variant, rep=rep,
                                             step=t.step, loss=t.loss, n_kept=t.n_kept)
                                        for t in tr.itertuples()]))
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):4d}/{len(comps)} "
                  f"count={r['count_acc']:.2f}%  full={r['full_acc']:.2f}%  "
                  f"MAE={r['mae']:.2f}", flush=True)
            rng = np.random.default_rng(seed + rep)
            for k in range(n_random):
                sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                Ev, Et = keep_only(av, mv, at, mt, sel)
                rows.append(dict(model=model, variant=variant, method="random pool", rep=k,
                                 size=len(sel), **scores(Ev[held], Et, obj[held], cnt[held])))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"counting_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"counting_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote counting_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("counting 1..10", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=4, type=int, help="calibration images per count")
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.max_steps, a.reps, seed_topk=a.seed_topk)
