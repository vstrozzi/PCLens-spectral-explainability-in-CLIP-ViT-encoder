"""Context-aware component ORDERINGS for the section-6.2 curves.

Section 6.2 orders components by a per-component score and then ablates cumulatively. That score
is marginal: it asks what a component is worth on its own, never what it adds given the ones
already picked. Redundant components therefore crowd the top of the list. This module builds the
ordering with the same greedy, context-aware rule as the pool selector in `exp_greedy_pool`:

  forward   start from everything mean-ablated; repeatedly ADD the component that most reduces the
            loss GIVEN the current set. The addition order is the ranking. Matched to the
            keep-only (reconstruction) curve.
  backward  start from the intact model; repeatedly REMOVE the component whose removal most
            INCREASES the loss given what is already removed. Matched to the ablation curve.

Both are run per tower (`vision`, `text`) and jointly, exactly like section 6.2. For a single-tower
ordering the OTHER tower is left intact, which is what the 6.2 vision-only / text-only curves do.

The greedy needs O(n^2 / 2) loss evaluations, so selection runs on a subsample of `n_samples`
COCO pairs while every reported curve is evaluated on the full 5,000 - the selection set is small
and separate, so the curves are not read off the same data the order was tuned on.

  python -m scripts_paper.exp_greedy_order --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/greedy_order_{order,ablation}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_loss_ranking import curve, infonce
from scripts_paper.pclens_core import (MODELS, RES_DIR, comp_delta, component_index, embed,
                                       load_tower, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")


@torch.no_grad()
def _greedy(av, mv, at, mt, tau_inv, idx, towers, mode, verbose=True, seed_topk=48):
    """Greedy context-aware ordering over the components of `towers`.

    mode="forward": components of `towers` start mean-ablated and are added back one at a time,
    each step taking the component that minimises the loss.
    mode="backward": they start intact and are removed one at a time, each step taking the
    component whose removal maximises the loss.
    Components of a tower not in `towers` are always left intact."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    tgt = torch.arange(idx.numel(), device=av.device)
    pool = [(t, c) for t in towers for c in component_index(*acts[t])]
    delta = {k: comp_delta(*acts[k[0]], k[1], *mean[k[0]])[idx] for k in pool}

    cur = {t: embed(*acts[t])[idx] for t in acts}
    if mode == "forward":                      # strip the ordered towers down to their mean
        for k in pool:
            cur[k[0]] = cur[k[0]] - delta[k]

    order, remaining = [], list(pool)
    if mode == "forward" and len(towers) == 2:
        # Both towers start at their mean, so all images are identical AND all texts are
        # identical: no SINGLE component can create any discrimination and the loss sits at
        # exactly log(M) whatever we add. The joint forward search has to be seeded with one
        # component from each tower at once, exactly as the pool selector does.
        cvs = [k for k in pool if k[0] == "vision"]
        cts = [k for k in pool if k[0] == "text"]
        if seed_topk:      # prefilter by solo score against the intact partner tower
            full = {t: embed(*acts[t])[idx] for t in acts}
            solo = {k: infonce(cur["vision"] + delta[k] if k[0] == "vision" else full["vision"],
                               cur["text"] + delta[k] if k[0] == "text" else full["text"], tau_inv)
                    for k in pool}
            cvs = sorted(cvs, key=lambda k: solo[k])[:seed_topk]
            cts = sorted(cts, key=lambda k: solo[k])[:seed_topk]
        best, best_l = None, None
        for a in cvs:
            ev = cur["vision"] + delta[a]
            for b in cts:
                l = infonce(ev, cur["text"] + delta[b], tau_inv)
                if best_l is None or l < best_l:
                    best, best_l = (a, b), l
        for k in best:
            cur[k[0]] = cur[k[0]] + delta[k]
            order.append((k[0], k[1], best_l))
            remaining.remove(k)
        if verbose:
            print(f"    seed pair {best[0][1]} + {best[1][1]} loss={best_l:.4f}", flush=True)
    while remaining:
        best, best_l = None, None
        for k in remaining:
            t = k[0]
            trial = cur[t] + delta[k] if mode == "forward" else cur[t] - delta[k]
            ev = trial if t == "vision" else cur["vision"]
            et = trial if t == "text" else cur["text"]
            l = infonce(ev, et, tau_inv)
            # forward wants the biggest drop, backward the biggest rise
            if best_l is None or (l < best_l if mode == "forward" else l > best_l):
                best, best_l = k, l
        cur[best[0]] = (cur[best[0]] + delta[best] if mode == "forward"
                        else cur[best[0]] - delta[best])
        order.append((best[0], best[1], best_l))
        remaining.remove(best)
        if verbose and len(order) % 50 == 0:
            print(f"    {mode} {len(order)}/{len(pool)} loss={best_l:.4f}", flush=True)
    return order


def _steps(n):
    s = set(np.round(np.linspace(0, n, 20)).astype(int).tolist())
    return s | {x for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64) if x <= n}


def run_model(model, device="cpu", n_samples=256, seed=69, n_random=3, seed_topk=48):
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    tau_inv = logit_scale(model)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(av.shape[0], generator=g)[:min(n_samples, av.shape[0])].to(device)
    mean = {"vision": (av.mean(0), None if mv is None else mv.mean(0)),
            "text": (at.mean(0), None if mt is None else mt.mean(0))}
    print(f"[{model}] greedy orderings on {idx.numel()} pairs, 1/tau={tau_inv:.1f}", flush=True)

    rng = np.random.default_rng(seed)
    ord_rows, ab_rows = [], []
    for tname, towers in (("vision", ["vision"]), ("text", ["text"]),
                          ("joint", ["vision", "text"])):
        orders = {}
        for mode in ("forward", "backward"):
            o = _greedy(av, mv, at, mt, tau_inv, idx, towers, mode, seed_topk=seed_topk)
            orders[f"greedy_{mode}"] = [[(t, c) for t, c, _ in o]]
            ord_rows += [dict(model=model, tower=tname, mode=mode, rank=r, comp_tower=t,
                              kind=c[0], layer=c[1], head=c[2], loss=l)
                         for r, (t, c, l) in enumerate(o)]
            print(f"  [{model}/{tname}] {mode} done ({len(o)} components)", flush=True)
        # marginal (context-free) baseline from the loss-restricted score, if it exists
        p = os.path.join(OUT, f"loss_scores_{model}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d = d[d.tower.isin(towers)].sort_values("score")
            orders["lossB_static"] = [[(r.tower, (r.kind, int(r.layer), int(r.head)))
                                       for r in d.itertuples()]]
        pool = orders["greedy_forward"][0]
        orders["random"] = [[pool[i] for i in rng.permutation(len(pool))] for _ in range(n_random)]

        steps = _steps(len(pool))
        for name, seqs in orders.items():
            for rep, s in enumerate(seqs):
                for cmode in ("ablate", "keep"):
                    for r in curve(av, mv, at, mt, s, mean, steps, cmode, tau_inv):
                        ab_rows.append(dict(model=model, tower=tname, order=name, mode=cmode,
                                            rep=rep, frac=r["step"] / max(len(s), 1), **r))
    pd.DataFrame(ord_rows).to_csv(os.path.join(OUT, f"greedy_order_order_{model}.csv"), index=False)
    df = pd.DataFrame(ab_rows)
    df.to_csv(os.path.join(OUT, f"greedy_order_ablation_{model}.csv"), index=False)
    print(f"[{model}] wrote greedy_order_{{order,ablation}} ({len(df)} rows)", flush=True)
    return df


def get_args_parser():
    p = argparse.ArgumentParser("context-aware greedy orderings", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", default=256, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--seed_topk", default=48, type=int,
                   help="prefilter each tower before the joint seed-pair search (0 = exhaustive)")
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.n_samples, a.seed, seed_topk=a.seed_topk)
