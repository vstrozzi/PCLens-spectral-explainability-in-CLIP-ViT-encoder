"""Specialise on 10 ImageNet classes - then measure what it costs on the other 990.

A selector tuned on a narrow task will improve that task. The question this asks is what happens
OUTSIDE it: the same kept set is re-scored on the FULL 1,000-way problem, so specialisation and
collateral damage are read off together. Reporting only the subset gain would be meaningless.

Everything is held fixed except which components (or which (component, PC) units) stay un-ablated:

  * means and PC bases always come from the FULL ImageNet activations, never from the 10-class
    slice. Mean-ablation is defined relative to a data distribution, so if the subset defined it
    the "same" model would differ between the two evaluations and the comparison would be void.
  * the calibration objective is 10-way cross-entropy over the subset's class names only;
  * calibration images are excluded from BOTH evaluations.

Sample sizes are small and stated rather than hidden: the decomposed ImageNet set holds 5 images
per class, so a 10-class subset is 50 images - `n_cal` per class calibrate and the rest are
held out. Numbers are averaged over `--subsets` independent class draws x `--reps` calibration
draws for that reason.

  python -m scripts_paper.exp_imagenet_subset --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/insub_{eval,trace}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_waterbirds_pc import pc_basis
from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_delta, comp_vec,
                                       component_index, embed, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
TASK = "imagenet"
TEXT = "imagenet_classnames"
VARIANTS = {"forward": dict(init="empty", allow_swap=False),
            "forward+swap": dict(init="empty", allow_swap=True),
            "backward": dict(init="full", allow_swap=False),
            "backward+swap": dict(init="full", allow_swap=True)}


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_imagenet(model, device, seed=69):
    av = _npy(f"{TASK}_attn_{model}_seed_{seed}.npy", device)
    mv = _npy(f"{TASK}_mlp_{model}_seed_{seed}.npy", device)
    at = _npy(f"{TEXT}_attn_text_{model}_seed_{seed}.npy", device)
    mt = _npy(f"{TEXT}_mlp_text_{model}_seed_{seed}.npy", device)
    y = _npy(f"{TASK}_labels_{model}_seed_{seed}.npy", device).long()
    return av, mv, at, mt, y


@torch.no_grad()
def _ce(Ev, Et, tau_inv, tgt):
    L = (torch.nn.functional.normalize(Ev, dim=1) @
         torch.nn.functional.normalize(Et, dim=1).T) * tau_inv
    return float(torch.nn.functional.cross_entropy(L, tgt))


@torch.no_grad()
def _acc(Ev, Et, tgt):
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    return float((L.argmax(1) == tgt).float().mean() * 100)


# --------------------------------------------------------------------- component level
@torch.no_grad()
def _deltas(av, mv, at, mt):
    """Per-component deltas against the FULL-dataset mean, for both towers."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    scaf_v = mean["vision"][0].sum((0, 1)).clone()
    if mv is not None:
        scaf_v += mean["vision"][1].sum(0)
    scaf_t = mean["text"][0].sum((0, 1)).clone()
    if mt is not None:
        scaf_t += mean["text"][1].sum(0)
    d = {}
    for t in acts:
        for c in component_index(*acts[t]):
            d[(t, c)] = comp_delta(*acts[t], c, *mean[t])
    return d, scaf_v, scaf_t


@torch.no_grad()
def search(units, dv, dt, scaf_v, scaf_t, cal, tsub, cal_y, tau_inv, init="empty",
           allow_swap=False, max_steps=10 ** 6, seed_topk=32, max_undo=100):
    """Greedy subset search over kept units, scored by 10-way CE on the calibration images.

    `dv[u]` / `dt[u]` are already restricted to the calibration images / subset class rows, so a
    step costs one [n_cal x d] add and one small matmul."""
    base_v = scaf_v.expand(cal.numel(), -1).clone()
    base_t = scaf_t.expand(len(tsub), -1).clone()
    kept = {u: init == "full" for u in units}
    cur_v, cur_t = base_v.clone(), base_t.clone()
    for u in units:
        if kept[u]:
            if u[0] == "vision":
                cur_v = cur_v + dv[u]
            else:
                cur_t = cur_t + dt[u]

    if init == "empty":
        # all text units ablated => the 10 class embeddings coincide and the loss is log(10)
        # whatever single unit is added, so the search must be seeded with a (vision, text) PAIR
        uv = [u for u in units if u[0] == "vision"]
        ut = [u for u in units if u[0] == "text"]
        fv = base_v + sum(dv[u] for u in uv)
        ft = base_t + sum(dt[u] for u in ut)
        solo = {u: (_ce(base_v + dv[u], ft, tau_inv, cal_y) if u[0] == "vision"
                    else _ce(fv, base_t + dt[u], tau_inv, cal_y)) for u in units}
        uv = sorted(uv, key=lambda u: solo[u])[:seed_topk]
        ut = sorted(ut, key=lambda u: solo[u])[:seed_topk]
        best, cur = None, None
        for a in uv:
            ev = base_v + dv[a]
            for b in ut:
                l = _ce(ev, base_t + dt[b], tau_inv, cal_y)
                if cur is None or l < cur:
                    best, cur = (a, b), l
        for u in best:
            kept[u] = True
            if u[0] == "vision":
                cur_v = cur_v + dv[u]
            else:
                cur_t = cur_t + dt[u]
        trace = [dict(step=0, loss=cur, n_kept=sum(kept.values()),
                      applied=[(u, 1.0) for u in best])]
    else:
        cur = _ce(cur_v, cur_t, tau_inv, cal_y)
        trace = [dict(step=0, loss=cur, n_kept=sum(kept.values()), applied=[])]

    undo_left = max_undo if allow_swap else 0
    for step in range(1, max_steps + 1):
        cand, cand_l, cand_add = None, cur, None
        for u in units:
            add = not kept[u]
            if ((add and init == "full") or (not add and init == "empty")) and undo_left <= 0:
                continue
            s = 1.0 if add else -1.0
            l = (_ce(cur_v + s * dv[u], cur_t, tau_inv, cal_y) if u[0] == "vision"
                 else _ce(cur_v, cur_t + s * dt[u], tau_inv, cal_y))
            if l < cand_l:
                cand, cand_l, cand_add = u, l, add
        if cand is None:
            break
        if (cand_add and init == "full") or (not cand_add and init == "empty"):
            undo_left -= 1
        kept[cand] = cand_add
        s = 1.0 if cand_add else -1.0
        if cand[0] == "vision":
            cur_v = cur_v + s * dv[cand]
        else:
            cur_t = cur_t + s * dt[cand]
        cur = cand_l
        trace.append(dict(step=step, loss=cur, n_kept=sum(kept.values()),
                          applied=[(cand, s)]))
    return [u for u in units if kept[u]], pd.DataFrame(trace)


# --------------------------------------------------------------------- PC level
@torch.no_grad()
def build_pc_units(av, mv, at, mt, k):
    """(component, PC) units with bases and coefficients from the FULL ImageNet activations."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    units, alpha, P = [], {}, {}
    for t in ("vision", "text"):
        for c in component_index(*acts[t]):
            X = comp_vec(*acts[t], c)
            _, Pk = pc_basis(X, k)
            if Pk.shape[0] == 0:
                continue
            alpha[(t, c)] = (X - X.mean(0)) @ Pk.T
            P[(t, c)] = Pk
            units += [(t, c, j) for j in range(Pk.shape[0])]
    scaf = {}
    for t in ("vision", "text"):
        E = embed(*acts[t]).clone()
        for c in component_index(*acts[t]):
            if (t, c) in alpha:
                E -= alpha[(t, c)] @ P[(t, c)]
        scaf[t] = E
    return units, alpha, P, scaf


@torch.no_grad()
def pc_search(units, alpha, P, scaf, cal, tsub, cal_y, tau_inv, **kw):
    dv = {u: alpha[(u[0], u[1])][cal, u[2]].unsqueeze(1) * P[(u[0], u[1])][u[2]]
          for u in units if u[0] == "vision"}
    dt = {u: alpha[(u[0], u[1])][tsub, u[2]].unsqueeze(1) * P[(u[0], u[1])][u[2]]
          for u in units if u[0] == "text"}
    # PC scaffolds are per-sample (they carry the residual), so they are passed as slices
    return _search_with_scaf(units, dv, dt, scaf["vision"][cal], scaf["text"][tsub],
                             cal_y, tau_inv, **kw)


@torch.no_grad()
def _search_with_scaf(units, dv, dt, base_v, base_t, cal_y, tau_inv, init="empty",
                      allow_swap=False, max_steps=10 ** 6, seed_topk=64, max_undo=100):
    """`search`, but the scaffold is already a per-sample matrix rather than one vector."""
    kept = {u: init == "full" for u in units}
    cur_v, cur_t = base_v.clone(), base_t.clone()
    for u in units:
        if kept[u]:
            if u[0] == "vision":
                cur_v = cur_v + dv[u]
            else:
                cur_t = cur_t + dt[u]
    if init == "empty":
        uv = [u for u in units if u[0] == "vision"]
        ut = [u for u in units if u[0] == "text"]
        fv = base_v + sum(dv[u] for u in uv)
        ft = base_t + sum(dt[u] for u in ut)
        solo = {u: (_ce(base_v + dv[u], ft, tau_inv, cal_y) if u[0] == "vision"
                    else _ce(fv, base_t + dt[u], tau_inv, cal_y)) for u in units}
        uv = sorted(uv, key=lambda u: solo[u])[:seed_topk]
        ut = sorted(ut, key=lambda u: solo[u])[:seed_topk]
        best, cur = None, None
        for a in uv:
            ev = base_v + dv[a]
            for b in ut:
                l = _ce(ev, base_t + dt[b], tau_inv, cal_y)
                if cur is None or l < cur:
                    best, cur = (a, b), l
        for u in best:
            kept[u] = True
            if u[0] == "vision":
                cur_v = cur_v + dv[u]
            else:
                cur_t = cur_t + dt[u]
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
            l = (_ce(cur_v + s * dv[u], cur_t, tau_inv, cal_y) if u[0] == "vision"
                 else _ce(cur_v, cur_t + s * dt[u], tau_inv, cal_y))
            if l < cand_l:
                cand, cand_l, cand_add = u, l, add
        if cand is None:
            break
        if (cand_add and init == "full") or (not cand_add and init == "empty"):
            undo_left -= 1
        kept[cand] = cand_add
        s = 1.0 if cand_add else -1.0
        if cand[0] == "vision":
            cur_v = cur_v + s * dv[cand]
        else:
            cur_t = cur_t + s * dt[cand]
        cur = cand_l
        trace.append(dict(step=step, loss=cur, n_kept=sum(kept.values()),
                          applied=[(cand, s)]))
    return [u for u in units if kept[u]], pd.DataFrame(trace)


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_comp(pool, d, scaf_v, scaf_t, n_img, n_txt, y, cls, held_sub, held_all):
    Ev = scaf_v.expand(n_img, -1).clone()
    Et = scaf_t.expand(n_txt, -1).clone()
    for u in pool:
        if u[0] == "vision":
            Ev = Ev + d[u]
        else:
            Et = Et + d[u]
    return _both(Ev, Et, y, cls, held_sub, held_all)


@torch.no_grad()
def eval_pc(pool, alpha, P, scaf, y, cls, held_sub, held_all):
    Ev, Et = scaf["vision"].clone(), scaf["text"].clone()
    for t, c, j in pool:
        dd = alpha[(t, c)][:, j].unsqueeze(1) * P[(t, c)][j]
        if t == "vision":
            Ev = Ev + dd
        else:
            Et = Et + dd
    return _both(Ev, Et, y, cls, held_sub, held_all)


@torch.no_grad()
def replay(trace, delta_of, Ev0, Et0, y, cls, held_sub, held_all, every=10):
    """Walk the search and score it every `every` steps - the plot that matters.

    The calibration loss falls monotonically by construction, so on its own it says nothing about
    whether the search is doing something useful. Putting the 10-class and 1,000-class accuracies
    on the same axis shows where specialisation starts costing generality. Scoring every step is
    wasteful (each 1,000-way pass is a 5,000 x 1,000 matmul), so it is subsampled and the final
    step is always included."""
    Ev, Et = Ev0.clone(), Et0.clone()
    n = len(trace) - 1
    out = []
    for r in trace.itertuples():
        for u, s in r.applied:
            dv_, dt_ = delta_of(u)
            if u[0] == "vision":
                Ev = Ev + s * dv_
            else:
                Et = Et + s * dt_
        if r.step % every == 0 or r.step == n:
            out.append(dict(step=r.step, loss=r.loss, n_kept=r.n_kept,
                            **_both(Ev, Et, y, cls, held_sub, held_all)))
    return pd.DataFrame(out)


@torch.no_grad()
def _both(Ev, Et, y, cls, held_sub, held_all):
    """The 10-class score and the 1,000-class score of the SAME kept set."""
    local = torch.full((int(y.max()) + 1,), -1, dtype=torch.long, device=y.device)
    local[cls] = torch.arange(len(cls), device=y.device)
    return dict(acc_subset=_acc(Ev[held_sub], Et[cls], local[y[held_sub]]),
                acc_full=_acc(Ev[held_all], Et, y[held_all]))


def run_model(model, device="cpu", seed=69, n_cal=2, k=8, subsets=5, reps=3, n_cls=10,
              max_steps=10 ** 6, n_random=3, do_pc=True, eval_every=10):
    av, mv, at, mt, y = load_imagenet(model, device, seed)
    tau_inv = logit_scale(model)
    d, scaf_v, scaf_t = _deltas(av, mv, at, mt)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    Ef_v, Ef_t = embed(av, mv), embed(at, mt)
    full_all = _acc(Ef_v, Ef_t, y)
    print(f"[{model}] {av.shape[0]} images / {at.shape[0]} classes, {len(comps)} components; "
          f"full-model 1000-way top-1 = {full_all:.2f}%", flush=True)

    units = alpha = P = scaf = None
    if do_pc:
        units, alpha, P, scaf = build_pc_units(av, mv, at, mt, k)
        print(f"[{model}] {len(units)} PC units (K<={k})", flush=True)

    rows, traces = [], []
    for sub in range(subsets):
        g = torch.Generator().manual_seed(seed + 1000 * sub)
        cls = torch.randperm(at.shape[0], generator=g)[:n_cls].sort().values.to(device)
        in_sub = torch.isin(y, cls)
        idx_sub = in_sub.nonzero(as_tuple=True)[0]
        local = torch.full((at.shape[0],), -1, dtype=torch.long, device=device)
        local[cls] = torch.arange(n_cls, device=device)
        base_sub = _acc(Ef_v[idx_sub], Ef_t[cls], local[y[idx_sub]])
        print(f"  [subset {sub}] classes={cls.tolist()}  n_img={idx_sub.numel()}  "
              f"full-model 10-way = {base_sub:.2f}%", flush=True)

        for rep in range(reps):
            gg = torch.Generator().manual_seed(seed + rep)
            cal = []
            for c in cls.tolist():
                w = (y == c).nonzero(as_tuple=True)[0]
                cal += w[torch.randperm(w.numel(), generator=gg)[:n_cal].to(w.device)].tolist()
            cal = torch.tensor(sorted(cal), device=device)
            held_all = torch.ones(av.shape[0], dtype=torch.bool, device=device)
            held_all[cal] = False
            held_sub = in_sub & held_all
            cal_y = local[y[cal]]
            meta = dict(model=model, subset=sub, rep=rep, classes=" ".join(map(str, cls.tolist())))
            if rep == 0:
                rows.append(dict(**meta, level="-", variant="full model", method="full model",
                                 size=len(comps),
                                 acc_subset=_acc(Ef_v[held_sub], Ef_t[cls], local[y[held_sub]]),
                                 acc_full=_acc(Ef_v[held_all], Ef_t, y[held_all])))

            dv = {u: d[u][cal] for u in comps if u[0] == "vision"}
            dt = {u: d[u][cls] for u in comps if u[0] == "text"}
            for variant, cfg in VARIANTS.items():
                pool, tr = search(comps, dv, dt, scaf_v, scaf_t, cal, cls, cal_y, tau_inv,
                                  max_steps=max_steps, **cfg)
                r = eval_comp(pool, d, scaf_v, scaf_t, av.shape[0], at.shape[0], y, cls,
                              held_sub, held_all)
                rows.append(dict(**meta, level="component", variant=variant, method="selected",
                                 size=len(pool), **r))
                cj = replay(tr, lambda u: (d[u], d[u]),
                            scaf_v.expand(av.shape[0], -1).clone(),
                            scaf_t.expand(at.shape[0], -1).clone(),
                            y, cls, held_sub, held_all, every=eval_every)
                traces.append(cj.assign(**meta, level="component", variant=variant))
                print(f"    [rep{rep} comp {variant:14s}] kept={len(pool):4d}/{len(comps)} "
                      f"10-way={r['acc_subset']:.2f}%  1000-way={r['acc_full']:.2f}%", flush=True)
                rng = np.random.default_rng(seed + rep)
                for q in range(n_random):
                    sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                    rows.append(dict(**meta, level="component", variant=variant,
                                     method="random pool", size=len(sel),
                                     **eval_comp(sel, d, scaf_v, scaf_t, av.shape[0], at.shape[0],
                                                 y, cls, held_sub, held_all)))
            if do_pc:
                for variant, cfg in VARIANTS.items():
                    pool, tr = pc_search(units, alpha, P, scaf, cal, cls, cal_y, tau_inv,
                                         max_steps=max_steps, **cfg)
                    r = eval_pc(pool, alpha, P, scaf, y, cls, held_sub, held_all)
                    rows.append(dict(**meta, level="pc", variant=variant, method="selected",
                                     size=len(pool), **r))

                    def _pcd(u, alpha=alpha, P=P):
                        dd = alpha[(u[0], u[1])][:, u[2]].unsqueeze(1) * P[(u[0], u[1])][u[2]]
                        return dd, dd
                    cj = replay(tr, _pcd, scaf["vision"], scaf["text"], y, cls,
                                held_sub, held_all, every=eval_every)
                    traces.append(cj.assign(**meta, level="pc", variant=variant))
                    print(f"    [rep{rep} pc   {variant:14s}] kept={len(pool):5d}/{len(units)} "
                          f"10-way={r['acc_subset']:.2f}%  1000-way={r['acc_full']:.2f}%",
                          flush=True)
                    rng = np.random.default_rng(seed + rep)
                    for q in range(n_random):
                        sel = [units[i] for i in
                               rng.choice(len(units), size=len(pool), replace=False)]
                        rows.append(dict(**meta, level="pc", variant=variant,
                                         method="random pool", size=len(sel),
                                         **eval_pc(sel, alpha, P, scaf, y, cls, held_sub,
                                                   held_all)))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"insub_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"insub_trace_{model}.csv"), index=False)
    print(f"[{model}] wrote insub_{{eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("ImageNet 10-class specialisation", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cls", default=10, type=int, help="classes per subset")
    p.add_argument("--subsets", default=5, type=int, help="independent random class draws")
    p.add_argument("--n_cal", default=2, type=int, help="calibration images per class")
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--k", default=8, type=int, help="PCs per component")
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--eval_every", default=10, type=int,
                   help="score the trajectory every N search steps")
    p.add_argument("--no_pc", action="store_true")
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.k, a.subsets, a.reps, a.n_cls,
                  a.max_steps, do_pc=not a.no_pc, eval_every=a.eval_every)
