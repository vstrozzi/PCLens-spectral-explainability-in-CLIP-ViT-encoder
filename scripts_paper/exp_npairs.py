"""Q1 - how many examples does the intervention actually need?

The claim under test is that ONE example per class is enough to find the components a frozen CLIP
uses as a shortcut. The pair is not hidden supervision: it is the input, a one-shot specification of
the decision rule the user wants, in the same sense a prompt is.

Two arms, because "which example" and "how many examples" are different questions:

  informative   the example per class comes from a MINORITY group (waterbird-on-land,
                landbird-on-water) - the user picks a case that exposes the failure;
  random        the example per class is drawn at random - the user picks carelessly.

The gap prices how much the CHOICE matters. If random nearly matches informative, the method
survives a careless user; if not, "pick an example that exposes the failure" is a documented usage
instruction rather than a hidden assumption.

Calibration images are excluded from evaluation at every n, and a random pool of the same size is
scored alongside, so "more components changed" is never mistaken for "better components chosen".

  python -m scripts_paper.exp_npairs --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/npairs_{eval,trace}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task, keep_only
from scripts_paper.exp_waterbirds import TASK, VARIANTS, accuracies, groups, local_search
from scripts_paper.pclens_core import MODELS, RES_DIR, component_index, embed, logit_scale

OUT = os.path.join(RES_DIR, "6_2")
NPAIRS = [1, 2, 5, 10, 25, 100]


def _minority(y, place):
    """The two cells where the background contradicts the usual correlation."""
    return (y == 0) & (place == 1), (y == 1) & (place == 0)


def sample_cal(y, place, n, arm, gen, device):
    """n calibration images per class, drawn from the minority cells or at random."""
    lo, hi = _minority(y, place)
    pools = ([lo, hi] if arm == "informative" else [y == 0, y == 1])
    cal = []
    for m in pools:
        w = m.nonzero(as_tuple=True)[0]
        if w.numel() == 0:
            return None
        take = min(n, w.numel())
        cal += w[torch.randperm(w.numel(), generator=gen)[:take].to(w.device)].tolist()
    return torch.tensor(sorted(set(cal)), device=device)


def run_model(model, device="cpu", seed=69, reps=5, n_random=3, npairs=NPAIRS,
              max_steps=10 ** 6, seed_topk=32):
    av, mv, at, mt, _ = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    full = accuracies(embed(av, mv), embed(at, mt), y, place)
    print(f"[{model}] {av.shape[0]} images, {len(comps)} components; full model "
          f"total={full['total']:.2f} worst-group={full['worst_group']:.2f}", flush=True)

    rows = [dict(model=model, arm="-", n_pairs=0, variant="full model", method="full model",
                 rep=-1, size=len(comps), n_cal=0,
                 **{k: v for k, v in full.items() if k != "groups"})]
    traces = []
    for arm in ("informative", "random"):
        for n in npairs:
            for rep in range(reps):
                gen = torch.Generator().manual_seed(seed + 100 * rep + n)
                cal = sample_cal(y, place, n, arm, gen, device)
                if cal is None or cal.numel() < 2:
                    continue
                held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
                held[cal] = False
                for variant, cfg in VARIANTS.items():
                    pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal],
                                            max_steps=max_steps, seed_topk=seed_topk,
                                            verbose=False, **cfg)
                    Ev, Et = keep_only(av, mv, at, mt, pool)
                    r = accuracies(Ev[held], Et, y[held], place[held])
                    base = dict(model=model, arm=arm, n_pairs=n, variant=variant, rep=rep,
                                n_cal=int(cal.numel()))
                    rows.append(dict(**base, method="selected", size=len(pool),
                                     **{k: v for k, v in r.items() if k != "groups"}))
                    # the ordered move list is what Q2 (stability) reads back
                    traces.append(pd.DataFrame(
                        [dict(**base, step=t.step, loss=t.loss, n_kept=t.n_kept,
                              move=str(t.applied)) for t in tr.itertuples()]))
                    rng = np.random.default_rng(seed + rep)
                    for q in range(n_random):
                        sel = [comps[i] for i in
                               rng.choice(len(comps), size=len(pool), replace=False)]
                        Ev, Et = keep_only(av, mv, at, mt, sel)
                        rr = accuracies(Ev[held], Et, y[held], place[held])
                        rows.append(dict(**{**base, "rep": q}, method="random pool",
                                         size=len(sel),
                                         **{k: v for k, v in rr.items() if k != "groups"}))
                print(f"  [{arm:11s} n={n:3d}] "
                      f"worst-group " + "  ".join(
                          f"{v[:3]}={np.mean([r['worst_group'] for r in rows if r['method'] == 'selected' and r['arm'] == arm and r['n_pairs'] == n and r['variant'] == v]):.1f}"
                          for v in VARIANTS), flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"npairs_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"npairs_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote npairs_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("Q1: how many examples?", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reps", default=5, type=int)
    p.add_argument("--npairs", nargs="+", type=int, default=NPAIRS)
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.reps, npairs=a.npairs, max_steps=a.max_steps,
                  seed_topk=a.seed_topk)
