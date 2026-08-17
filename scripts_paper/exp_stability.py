"""Q2 - do independent one-shot examples discover the SAME components?

If two unrelated pairs of images select the same internal components, the method is finding a
property of the model. If they do not, it is fitting the images. This is the experiment that can
falsify the mechanistic claim outright, so it is built to be hard to fool.

The trap it is built around: the converged pools are large (~150 of 266 components), and two RANDOM
subsets of that size already overlap at Jaccard ~= 0.43. Raw Jaccard on full pools therefore looks
impressive while meaning almost nothing. Every number here is reported against its own chance level:

  prefix k=10, 15   the first k components the search commits to. Chance Jaccard ~= 0.02 at k=10 of
                    266, so real agreement is unmistakable. THIS is the number that carries the claim.
  plateau prefix    the prefix at which held-out worst-group accuracy first reaches within `tol` of
                    the run's best - the effective set, the fairest choice of k because it is where
                    the method has already done its work.
  full pool         reported for completeness and explicitly labelled near-uninformative.

Chance is computed two ways, and both are printed: a closed-form hypergeometric expectation, and an
empirical baseline from random subsets of the same sizes (which also captures any size asymmetry
between the two runs being compared).

  python -m scripts_paper.exp_stability --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/stability_{pairs,summary}_{model}.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task, keep_only
from scripts_paper.exp_npairs import sample_cal
from scripts_paper.exp_waterbirds import TASK, VARIANTS, accuracies, groups, local_search
from scripts_paper.pclens_core import MODELS, RES_DIR, component_index, embed, logit_scale

OUT = os.path.join(RES_DIR, "6_2")
PREFIXES = [5, 10, 15, 25, 50]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def expected_jaccard(n_a, n_b, N):
    """Chance Jaccard for two independent uniform subsets of sizes n_a, n_b from N components.

    E|A n B| = n_a*n_b/N exactly; |A u B| = n_a + n_b - |A n B|. Using the expectation inside the
    ratio is an approximation, but it is accurate here and, more importantly, it is the number a
    reader can check by hand."""
    inter = n_a * n_b / N
    return inter / (n_a + n_b - inter) if (n_a + n_b - inter) > 0 else float("nan")


def moves_in_order(trace, init):
    """The components the search committed to, in the order it committed to them.

    Direction matters: a forward run ADDS components, so its decisions are the additions; a backward
    run starts from everything and REMOVES, so its decisions are the drops. Reading additions from a
    backward run yields an empty list and a meaningless zero overlap."""
    want = 1.0 if init == "empty" else -1.0
    out = []
    for r in trace:
        for k, sign in r:
            if sign == want and k not in out:
                out.append(k)
    return out


def plateau_k(curve, tol=1.0):
    """First prefix within `tol` points of the best held-out value on the curve."""
    if not curve:
        return 0
    best = max(curve)
    for i, v in enumerate(curve):
        if v >= best - tol:
            return i + 1
    return len(curve)


def run_model(model, device="cpu", seed=69, n_runs=20, n_pairs=1, arm="informative",
              max_steps=10 ** 6, seed_topk=32, eval_every=5, tol=1.0, n_null=20):
    av, mv, at, mt, _ = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    N = len(comps)
    print(f"[{model}] {N} components; {n_runs} independent {arm} draws of {n_pairs} pair(s)",
          flush=True)

    runs = {v: [] for v in VARIANTS}          # variant -> list of ordered move lists
    pools = {v: [] for v in VARIANTS}
    plateau = {v: [] for v in VARIANTS}
    for r in range(n_runs):
        gen = torch.Generator().manual_seed(seed + 977 * r)
        cal = sample_cal(y, place, n_pairs, arm, gen, device)
        if cal is None:
            continue
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal], max_steps=max_steps,
                                    seed_topk=seed_topk, verbose=False, **cfg)
            order = moves_in_order([t.applied for t in tr.itertuples()], cfg["init"])
            # held-out curve along the prefix, to locate the plateau
            curve, acc_pool = [], []
            for k in range(eval_every, min(len(order), 60) + 1, eval_every):
                Ev, Et = keep_only(av, mv, at, mt, order[:k])
                curve.append(accuracies(Ev[held], Et, y[held], place[held])["worst_group"])
                acc_pool.append(k)
            runs[variant].append(order)
            pools[variant].append(pool)
            plateau[variant].append(acc_pool[plateau_k(curve, tol) - 1] if curve else 0)
        if (r + 1) % 5 == 0:
            print(f"  {r + 1}/{n_runs} draws done", flush=True)

    rng = np.random.default_rng(seed)
    rows, summ = [], []
    for variant in VARIANTS:
        orders, pl = runs[variant], pools[variant]
        if len(orders) < 2:
            continue
        ks = [("prefix", k) for k in PREFIXES] + [("plateau", None), ("full", None)]
        for kind, k in ks:
            obs, chance_emp, chance_hyp = [], [], []
            for i, j in itertools.combinations(range(len(orders)), 2):
                if kind == "prefix":
                    a, b = orders[i][:k], orders[j][:k]
                elif kind == "plateau":
                    a = orders[i][:max(plateau[variant][i], 1)]
                    b = orders[j][:max(plateau[variant][j], 1)]
                else:
                    a, b = pl[i], pl[j]
                if not a or not b:
                    continue
                obs.append(jaccard(a, b))
                chance_hyp.append(expected_jaccard(len(a), len(b), N))
                # one random draw is very noisy at small k, so average a handful
                ce = [jaccard(rng.choice(N, size=len(a), replace=False).tolist(),
                              rng.choice(N, size=len(b), replace=False).tolist())
                      for _ in range(n_null)]
                chance_emp.append(float(np.mean(ce)))
                rows.append(dict(model=model, variant=variant, kind=kind, k=k or -1,
                                 i=i, j=j, jaccard=obs[-1], chance_emp=chance_emp[-1],
                                 chance_hyp=chance_hyp[-1], n_a=len(a), n_b=len(b)))
            if obs:
                summ.append(dict(model=model, variant=variant, kind=kind, k=k or -1,
                                 n_comp=N, mean_size=float(np.mean([len(orders[i][:k])
                                                                    if kind == "prefix" else 0
                                                                    for i in range(len(orders))]))
                                 if kind == "prefix" else np.nan,
                                 jaccard=float(np.mean(obs)), jaccard_sd=float(np.std(obs)),
                                 chance_emp=float(np.mean(chance_emp)),
                                 chance_hyp=float(np.mean(chance_hyp)),
                                 excess=float(np.mean(obs) - np.mean(chance_emp))))
                s = summ[-1]
                lab = f"{kind}{'' if k is None else f' k={k}'}"
                print(f"  [{variant:14s} {lab:12s}] Jaccard={s['jaccard']:.3f} "
                      f"chance={s['chance_emp']:.3f}  EXCESS={s['excess']:+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"stability_pairs_{model}.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, f"stability_summary_{model}.csv"), index=False)
    print(f"[{model}] wrote stability_{{pairs,summary}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("Q2: component stability", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_runs", default=20, type=int, help="independent example draws")
    p.add_argument("--n_pairs", default=1, type=int, help="pairs per draw (1 = one-shot)")
    p.add_argument("--arm", default="informative", choices=["informative", "random"])
    p.add_argument("--tol", default=1.0, type=float, help="plateau tolerance in accuracy points")
    p.add_argument("--n_null", default=20, type=int, help="random draws averaged for the null")
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_runs, a.n_pairs, a.arm, a.max_steps, a.seed_topk,
                  tol=a.tol, n_null=a.n_null)
