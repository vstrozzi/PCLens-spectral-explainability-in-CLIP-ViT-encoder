"""Waterbirds selection at PC granularity: the unit is (component, principal component).

Every method so far treats a component as atomic. Writing a component through its own spectral
basis splits it further,

    c_a(m) = mean_a + sum_k alpha_k^a(m) p_k^a + resid_a(m),

so the natural unit becomes the pair (component a, PC k). The loss terms carry over unchanged - the
pair interaction simply factorises,

    S_(a,i),(b,j)(m, n) = alpha_i^a(m) * alpha_j^b(n) * <p_i^a, p_j^b>,

and mean-ablating a unit means replacing alpha by its dataset mean, which is zero on the
mean-centred basis. That is exactly PCSelection, now driven by the loss instead of by a query.

Everything that is NOT selectable (component means, plus the residual beyond the K kept PCs) forms
a fixed scaffold that is always present, so
    embedding = scaffold + sum over selected units,
which keeps the decomposition exact whatever K is.

Basis estimation differs per tower, and this matters:
  vision  PCs from the task's own 2,000 image activations - plenty of samples.
  text    Waterbirds has only TWO class names, so a basis estimated there would be rank <= 1 after
          centring. The text PCs are therefore taken from the broad text set (the same corpus
          PCLens uses), and the class-name activations are projected onto it.

  python -m scripts_paper.exp_waterbirds_pc --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/waterbirds_pc_{eval,trace}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task
from scripts_paper.exp_waterbirds import TASK, VARIANTS, _ce, accuracies, groups
from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_vec, component_index, embed,
                                       logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
TEXT_SET = "top_1500_nouns_5_sentences_imagenet_bias_clean"


@torch.no_grad()
def pc_basis(X, k, var=0.99):
    """Top-k right singular vectors of the mean-centred activations, plus the mean."""
    mu = X.mean(0)
    Xc = X - mu
    q = min(k, min(Xc.shape) - 1)
    if q < 1:
        return mu, torch.zeros(0, X.shape[1], device=X.device)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    keep = min(q, int((torch.cumsum(S ** 2, 0) / (S ** 2).sum() < var).sum().item()) + 1)
    return mu, Vh[:keep]


@torch.no_grad()
def build_units(model, av, mv, at, mt, k, device, seed=69):
    """Per-component PC bases and the coefficients of the target data on them.

    Returns (units, alpha, P, scaffold_v, scaffold_t) where alpha[(t, c)] is [N, K] and
    P[(t, c)] is [K, d]."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    # text basis from the broad corpus: two class names cannot define one
    ta = torch.from_numpy(np.load(os.path.join(
        ACT_DIR, f"{TEXT_SET}_attn_text_{model}_seed_{seed}.npy"))).to(device)
    p = os.path.join(ACT_DIR, f"{TEXT_SET}_mlp_text_{model}_seed_{seed}.npy")
    tm = torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None
    basis_src = {"vision": (av, mv), "text": (ta, tm)}

    units, alpha, P = [], {}, {}
    for t in ("vision", "text"):
        for c in component_index(*acts[t]):
            mu, Pk = pc_basis(comp_vec(*basis_src[t], c), k)
            if Pk.shape[0] == 0:
                continue
            X = comp_vec(*acts[t], c)
            alpha[(t, c)] = (X - X.mean(0)) @ Pk.T          # [N, K]
            P[(t, c)] = Pk
            units += [(t, c, j) for j in range(Pk.shape[0])]
    # scaffold = what the units cannot express (component means + residual beyond K PCs)
    scaf = {}
    for t in ("vision", "text"):
        E = embed(*acts[t]).clone()
        for c in component_index(*acts[t]):
            if (t, c) in alpha:
                E -= alpha[(t, c)] @ P[(t, c)]
        scaf[t] = E
    return units, alpha, P, scaf


@torch.no_grad()
def local_search_pc(units, alpha, P, scaf, tau_inv, cal, cal_y, n_cls, init="empty",
                    allow_swap=False, max_steps=10 ** 6, seed_topk=64, verbose=False,
                    max_undo=100):
    """Same add / drop / swap search as the component-level version, over PC units.

    `max_undo` budgets moves against the primary direction, so a swap-enabled run cannot spend its
    steps oscillating; it still runs to convergence."""
    def d(u, idx):
        t, c, j = u
        return alpha[(t, c)][idx, j].unsqueeze(1) * P[(t, c)][j]

    tix = torch.arange(n_cls, device=cal.device)
    dl = {u: d(u, cal if u[0] == "vision" else tix) for u in units}
    base_v, base_t = scaf["vision"][cal], scaf["text"]

    kept = {u: init == "full" for u in units}
    cur_v, cur_t = base_v.clone(), base_t.clone()
    for u in units:
        if kept[u]:
            if u[0] == "vision":
                cur_v = cur_v + dl[u]
            else:
                cur_t = cur_t + dl[u]

    if init == "empty":
        uv = [u for u in units if u[0] == "vision"]
        ut = [u for u in units if u[0] == "text"]
        fv, ft = base_v + sum(dl[u] for u in uv), base_t + sum(dl[u] for u in ut)
        solo = {u: (_ce(base_v + dl[u], ft, tau_inv, cal_y) if u[0] == "vision"
                    else _ce(fv, base_t + dl[u], tau_inv, cal_y)) for u in units}
        uv = sorted(uv, key=lambda u: solo[u])[:seed_topk]
        ut = sorted(ut, key=lambda u: solo[u])[:seed_topk]
        best, cur = None, None
        for a in uv:
            ev = base_v + dl[a]
            for b in ut:
                l = _ce(ev, base_t + dl[b], tau_inv, cal_y)
                if cur is None or l < cur:
                    best, cur = (a, b), l
        for u in best:
            kept[u] = True
            if u[0] == "vision":
                cur_v = cur_v + dl[u]
            else:
                cur_t = cur_t + dl[u]
        applied0 = [(u, 1.0) for u in best]
    else:
        cur = _ce(cur_v, cur_t, tau_inv, cal_y)
        applied0 = []

    trace = [dict(step=0, loss=cur, n_kept=sum(kept.values()), applied=applied0)]
    undo_left = max_undo if allow_swap else 0
    for step in range(1, max_steps + 1):
        cand, cand_l, cand_add = None, cur, None
        for u in units:
            add = not kept[u]
            if ((add and init == "full") or (not add and init == "empty")) and undo_left <= 0:
                continue
            s = 1.0 if add else -1.0
            l = (_ce(cur_v + s * dl[u], cur_t, tau_inv, cal_y) if u[0] == "vision"
                 else _ce(cur_v, cur_t + s * dl[u], tau_inv, cal_y))
            if l < cand_l:
                cand, cand_l, cand_add = u, l, add
        if cand is None:
            break
        if (cand_add and init == "full") or (not cand_add and init == "empty"):
            undo_left -= 1
        kept[cand] = cand_add
        s = 1.0 if cand_add else -1.0
        if cand[0] == "vision":
            cur_v = cur_v + s * dl[cand]
        else:
            cur_t = cur_t + s * dl[cand]
        cur = cand_l
        trace.append(dict(step=step, loss=cur, n_kept=sum(kept.values()), applied=[(cand, s)]))
        if verbose and step % 50 == 0:
            print(f"    step {step} loss={cur:.4f} kept={sum(kept.values())}", flush=True)
    return [u for u in units if kept[u]], pd.DataFrame(trace)


@torch.no_grad()
def replay_pc(units, alpha, P, scaf, trace, init, y, place, held):
    """Held-out accuracy after each move, replayed on the full dataset."""
    n = scaf["vision"].shape[0]
    tix = torch.arange(scaf["text"].shape[0], device=scaf["text"].device)

    def d(u):
        t, c, j = u
        idx = slice(None) if t == "vision" else tix
        return alpha[(t, c)][idx, j].unsqueeze(1) * P[(t, c)][j]

    Ev, Et = scaf["vision"].clone(), scaf["text"].clone()
    if init == "full":
        for u in units:
            if u[0] == "vision":
                Ev = Ev + d(u)
            else:
                Et = Et + d(u)
    out = []
    for r in trace.itertuples():
        for u, s in r.applied:
            if u[0] == "vision":
                Ev = Ev + s * d(u)
            else:
                Et = Et + s * d(u)
        out.append(dict(step=r.step, **{a: b for a, b in
                                        accuracies(Ev[held], Et, y[held], place[held]).items()
                                        if a != "groups"}))
    return pd.DataFrame(out)


def run_model(model, device="cpu", seed=69, n_cal=8, k=8, max_steps=10 ** 6, reps=2,
              n_random=5):
    av, mv, at, mt, y_lab = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    tau_inv = logit_scale(model)
    units, alpha, P, scaf = build_units(model, av, mv, at, mt, k, device, seed)
    n_comp = len(component_index(av, mv)) + len(component_index(at, mt))
    full = accuracies(embed(av, mv), embed(at, mt), y, place)
    print(f"[{model}] {len(units)} PC units from {n_comp} components (K<={k}); "
          f"full model worst-group={full['worst_group']:.2f} total={full['total']:.2f}", flush=True)

    rows = [dict(model=model, regime="-", variant="full model", rep=-1, size=len(units),
                 **{a: b for a, b in full.items() if a != "groups"})]
    traces, moves = [], []
    for regime in ("class", "group"):
        strata = y if regime == "class" else (2 * y + place)
        for rep in range(reps):
            g = torch.Generator().manual_seed(seed + rep)
            cal = []
            for s in strata.unique():
                w = (strata == s).nonzero(as_tuple=True)[0]
                cal += w[torch.randperm(w.numel(), generator=g)[:n_cal].to(w.device)].tolist()
            cal = torch.tensor(cal, device=device)
            held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
            held[cal] = False
            for variant, cfg in VARIANTS.items():
                pool, tr = local_search_pc(units, alpha, P, scaf, tau_inv, cal, y[cal],
                                           at.shape[0], max_steps=max_steps, **cfg)
                tj = replay_pc(units, alpha, P, scaf, tr, cfg["init"], y, place, held).merge(
                    tr[["step", "loss", "n_kept"]], on="step")
                bi = int(tj.worst_group.idxmax())
                base = dict(model=model, regime=regime, variant=variant, rep=rep)
                rows.append(dict(method="selected", size=len(pool), **base,
                                 **tj.iloc[-1].drop("step").to_dict()))
                rows.append(dict(method="ORACLE best worst-group", size=int(tj.loc[bi, "n_kept"]),
                                 **base, **tj.loc[bi].drop("step").to_dict()))
                traces.append(tj.assign(**base))
                # the ordered move list: which unit, added or dropped, and the loss after it
                for r in tr.itertuples():
                    for (t, c, j), sign in r.applied:
                        moves.append(dict(**base, step=r.step, tower=t, kind=c[0], layer=c[1],
                                          head=c[2], pc=j, action="add" if sign > 0 else "drop",
                                          loss=r.loss))
                print(f"  [{regime} rep{rep} {variant:14s}] kept={len(pool):5d}/{len(units)} "
                      f"total={tj.iloc[-1].total:.2f} "
                      f"worst-group={tj.iloc[-1].worst_group:.2f}", flush=True)
                rng = np.random.default_rng(seed + rep)
                for r in range(n_random):
                    sel = [units[i] for i in rng.choice(len(units), size=len(pool), replace=False)]
                    Ev, Et = scaf["vision"].clone(), scaf["text"].clone()
                    tix = torch.arange(at.shape[0], device=device)
                    for u in sel:
                        t, c, j = u
                        idx = slice(None) if t == "vision" else tix
                        dd = alpha[(t, c)][idx, j].unsqueeze(1) * P[(t, c)][j]
                        if t == "vision":
                            Ev = Ev + dd
                        else:
                            Et = Et + dd
                    rows.append(dict(model=model, regime=regime, variant=variant,
                                     method="random units", rep=r, size=len(pool),
                                     **{a: b for a, b in
                                        accuracies(Ev[held], Et, y[held], place[held]).items()
                                        if a != "groups"}))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"waterbirds_pc_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"waterbirds_pc_trace_{model}.csv"), index=False)
    pd.DataFrame(moves).to_csv(os.path.join(OUT, f"waterbirds_pc_moves_{model}.csv"), index=False)
    print(f"[{model}] wrote waterbirds_pc_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("waterbirds PC-level selection", add_help=False)
    p.add_argument("--models", nargs="+", default=["ViT-B-32"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", default=8, type=int, help="PCs kept per component")
    p.add_argument("--n_cal", default=8, type=int)
    p.add_argument("--max_steps", default=10 ** 6, type=int, help="cap; default = run to convergence")
    p.add_argument("--reps", default=2, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.k, a.max_steps, a.reps)
