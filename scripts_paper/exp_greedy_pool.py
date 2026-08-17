"""Training-free greedy component selection by direct loss minimisation.

The heuristic: take a tiny calibration set of CORRECT pairs (default 1 per class), and grow a pool
of components that greedily minimises the contrastive loss computed from the pool alone - every
component outside the pool is mean-ablated. Nothing is trained; the only thing consumed is the
calibration set's pairing.

    pool <- argmin_{(a,b)} loss({a,b})          # one component per tower: a single tower alone
                                                # leaves the other side constant => no signal
    repeat:
        c* <- argmin_{c not in pool} loss(pool + {c})     # c ranges over BOTH towers
        if loss(pool + {c*}) >= loss(pool): stop
        pool <- pool + {c*}

Because the residual stream is a linear sum, "keep only the pool" is an incremental update:
E = (everything mean-ablated) + sum_{c in pool} (c(i) - mean_c), so a candidate costs one rank-1
style update plus the loss on the calibration set.

The selected pool is then evaluated on the FULL task, with everything outside it mean-ablated:
  * zero-shot classification accuracy (top-1) vs the full model, random pools of the same size,
    and the top-|pool| components of the static loss-restricted ranking;
  * COCO retrieval R@1 at increasing gallery sizes, to see whether a pool chosen on a tiny
    calibration set survives a harder task.

  python -m scripts_paper.exp_greedy_pool --models ViT-B-32 --tasks fairface CIFAR100
Writes output_dir/results_paper/6_2/greedy_pool_{trace,pool,eval,retrieval}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_delta, comp_vec,
                                       component_index, embed, load_tower, logit_scale,
                                       retrieval_metrics)

OUT = os.path.join(RES_DIR, "6_2")
TASKS = ["fairface", "CIFAR100", "caltech", "imagenet"]


def _load_task(model, task, device, seed=69):
    """Image tower over the task images + text tower over its class names, plus labels."""
    d = ACT_DIR
    av = torch.from_numpy(np.load(os.path.join(d, f"{task}_attn_{model}_seed_{seed}.npy"))).to(device)
    p = os.path.join(d, f"{task}_mlp_{model}_seed_{seed}.npy")
    mv = torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None
    at = torch.from_numpy(np.load(os.path.join(
        d, f"{task}_classnames_attn_text_{model}_seed_{seed}.npy"))).to(device)
    p = os.path.join(d, f"{task}_classnames_mlp_text_{model}_seed_{seed}.npy")
    mt = torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None
    y = torch.from_numpy(np.load(os.path.join(d, f"{task}_labels_{model}_seed_{seed}.npy"))).to(device)
    return av, mv, at, mt, y.long()


@torch.no_grad()
def _mean_state(av, mv, at, mt, n_img, n_txt):
    """Embeddings with EVERY component mean-ablated, i.e. the constant scaffold."""
    zv = av.mean(0).sum((0, 1)).expand(n_img, -1).clone()
    if mv is not None:
        zv += mv.mean(0).sum(0)
    zt = at.mean(0).sum((0, 1)).expand(n_txt, -1).clone()
    if mt is not None:
        zt += mt.mean(0).sum(0)
    return zv, zt


@torch.no_grad()
def _loss(Ev, Et, tau_inv, tgt):
    L = (torch.nn.functional.normalize(Ev, dim=1) @
         torch.nn.functional.normalize(Et, dim=1).T) * tau_inv
    ce = torch.nn.functional.cross_entropy
    return float((ce(L, tgt) + ce(L.T, tgt)) / 2) if L.shape[0] == L.shape[1] else float(ce(L, tgt))


@torch.no_grad()
def greedy_pool(av, mv, at, mt, tau_inv, img_idx, txt_idx, max_size=64, verbose=True,
                seed_topk=0):
    """Grow the pool while the calibration loss strictly improves. Returns (pool, trace)."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    tgt = torch.arange(img_idx.numel(), device=av.device)
    # per-component calibration-set deltas, precomputed once
    dl = {}
    for t in acts:
        sel = img_idx if t == "vision" else txt_idx
        for c in component_index(*acts[t]):
            dl[(t, c)] = comp_delta(*acts[t], c, *mean[t])[sel]
    base_v, base_t = _mean_state(av, mv, at, mt, img_idx.numel(), txt_idx.numel())

    def loss_of(pool):
        Ev, Et = base_v.clone(), base_t.clone()
        for t, c in pool:
            if t == "vision":
                Ev += dl[(t, c)]
            else:
                Et += dl[(t, c)]
        return _loss(Ev, Et, tau_inv, tgt)

    # --- seed: the best (vision, text) pair; one tower alone leaves the other side constant
    cv = [("vision", c) for c in component_index(av, mv)]
    ct = [("text", c) for c in component_index(at, mt)]
    if seed_topk:
        # The exhaustive seed is |cv|*|ct| loss evaluations, which is the single most expensive
        # step on the big towers. Prefilter each tower by its SOLO loss (that component kept, the
        # other tower left intact) - |cv|+|ct| evaluations - and search pairs only among the top-k.
        full_v, full_t = embed(av, mv)[img_idx], embed(at, mt)[txt_idx]
        solo = {}
        for k in cv:
            solo[k] = _loss(base_v + dl[k], full_t, tau_inv, tgt)
        for k in ct:
            solo[k] = _loss(full_v, base_t + dl[k], tau_inv, tgt)
        seed_v = sorted(cv, key=lambda k: solo[k])[:seed_topk]
        seed_t = sorted(ct, key=lambda k: solo[k])[:seed_topk]
    else:
        seed_v, seed_t = cv, ct
    best, best_l = None, float("inf")
    for a in seed_v:
        Ev = base_v + dl[a]
        for b in seed_t:
            l = _loss(Ev, base_t + dl[b], tau_inv, tgt)
            if l < best_l:
                best, best_l = [a, b], l
    pool, cur = list(best), best_l
    trace = [dict(size=2, loss=cur, added=f"{best[0]}+{best[1]}")]
    if verbose:
        print(f"  seed pair {best[0]} + {best[1]}  loss={cur:.4f}", flush=True)

    # --- forward selection
    remaining = [k for k in cv + ct if k not in pool]
    while len(pool) < max_size and remaining:
        Ev, Et = base_v.clone(), base_t.clone()
        for t, c in pool:
            (Ev if t == "vision" else Et).add_(dl[(t, c)])
        cand, cand_l = None, cur
        for k in remaining:
            l = (_loss(Ev + dl[k], Et, tau_inv, tgt) if k[0] == "vision"
                 else _loss(Ev, Et + dl[k], tau_inv, tgt))
            if l < cand_l:
                cand, cand_l = k, l
        if cand is None:
            break
        pool.append(cand); remaining.remove(cand); cur = cand_l
        trace.append(dict(size=len(pool), loss=cur, added=str(cand)))
        if verbose:
            print(f"  +{cand} -> loss={cur:.4f} (|pool|={len(pool)})", flush=True)
    return pool, pd.DataFrame(trace)


@torch.no_grad()
def keep_only(av, mv, at, mt, pool):
    """Full-dataset embeddings with everything outside `pool` mean-ablated."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    Ev, Et = _mean_state(av, mv, at, mt, av.shape[0], at.shape[0])
    for t, c in pool:
        d = comp_delta(*acts[t], c, *mean[t])
        if t == "vision":
            Ev = Ev + d
        else:
            Et = Et + d
    return Ev, Et


@torch.no_grad()
def zero_shot(Ev, Et, y):
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    return float((L.argmax(1) == y).float().mean() * 100)


def run_task(model, task, device, seed=69, per_class=1, max_size=64, n_random=5,
             cal_seed=None, seed_topk=0):
    av, mv, at, mt, y = _load_task(model, task, device, seed)
    tau_inv = logit_scale(model)
    n_cls = at.shape[0]
    # the calibration DRAW is reseeded separately, so the same run can repeat the selection on
    # different tiny calibration sets and report the spread
    g = torch.Generator().manual_seed(seed if cal_seed is None else cal_seed)
    # calibration: `per_class` correct images per class, paired with their own class name
    img_idx, txt_idx = [], []
    for c in range(n_cls):
        w = (y == c).nonzero(as_tuple=True)[0]
        if w.numel() == 0:
            continue
        pick = w[torch.randperm(w.numel(), generator=g)[:per_class].to(w.device)]
        img_idx += pick.tolist(); txt_idx += [c] * pick.numel()
    img_idx = torch.tensor(img_idx, device=device); txt_idx = torch.tensor(txt_idx, device=device)
    print(f"[{model}/{task}] {n_cls} classes, calibration {img_idx.numel()} pairs, "
          f"{av.shape[0]} images total", flush=True)

    pool, trace = greedy_pool(av, mv, at, mt, tau_inv, img_idx, txt_idx, max_size,
                              seed_topk=seed_topk)
    n_comp = len(component_index(av, mv)) + len(component_index(at, mt))

    full = zero_shot(*(embed(av, mv), embed(at, mt)), y)
    # Test accuracy at EVERY prefix of the pool. The greedy stop is decided on the tiny
    # calibration set, so the pool can keep growing long after test accuracy has peaked -
    # the trajectory is what shows where that happens.
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    Ev, Et = _mean_state(av, mv, at, mt, av.shape[0], at.shape[0])
    traj = []
    for i, (t, c) in enumerate(pool):
        d = comp_delta(*acts[t], c, *mean[t])
        if t == "vision":
            Ev = Ev + d
        else:
            Et = Et + d
        traj.append(dict(size=i + 1, acc=zero_shot(Ev, Et, y)))
    trace = trace.merge(pd.DataFrame(traj), on="size", how="left")
    best_i = int(np.argmax([r["acc"] for r in traj]))
    sel = traj[-1]["acc"]
    cs = dict(model=model, task=task, cal_seed=cal_seed if cal_seed is not None else seed)
    rows = [dict(method="full model", size=n_comp, acc=full, **cs),
            dict(method="greedy pool", size=len(pool), acc=sel, **cs),
            dict(method="greedy pool (best prefix)", size=best_i + 1,
                 acc=traj[best_i]["acc"], **cs)]
    # controls at the SAME budget
    allc = ([("vision", c) for c in component_index(av, mv)] +
            [("text", c) for c in component_index(at, mt)])
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        idx = rng.choice(len(allc), size=len(pool), replace=False)
        rows.append(dict(method="random pool", size=len(pool), rep=r,
                         acc=zero_shot(*keep_only(av, mv, at, mt, [allc[i] for i in idx]), y), **cs))
    p_sc = os.path.join(OUT, f"loss_scores_{model}.csv")
    if os.path.exists(p_sc):
        d = pd.read_csv(p_sc).nsmallest(len(pool), "score")
        top = [(r.tower, (r.kind, int(r.layer), int(r.head))) for r in d.itertuples()]
        rows.append(dict(method="top-k loss score", size=len(pool),
                         acc=zero_shot(*keep_only(av, mv, at, mt, top), y), **cs))
    print(f"[{model}/{task}] full={full:.2f}  pool({len(pool)})={sel:.2f}  "
          f"best prefix({best_i + 1})={traj[best_i]['acc']:.2f}", flush=True)
    return pool, trace, pd.DataFrame(rows)


@torch.no_grad()
def retrieval_scaling(model, pool_src, device, seed=69, sizes=(500, 1000, 2500, 5000),
                      n_cal=64, max_size=64, seed_topk=0):
    """Select on a small COCO calibration set, then test at increasing gallery size."""
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    tau_inv = logit_scale(model)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(av.shape[0], generator=g)
    cal = perm[:n_cal].to(device)
    pool, trace = greedy_pool(av, mv, at, mt, tau_inv, cal, cal, max_size,
                              seed_topk=seed_topk)
    Ev, Et = keep_only(av, mv, at, mt, pool)
    full_v, full_t = embed(av, mv), embed(at, mt)
    rows = []
    for n in sizes:
        s = perm[:n].to(device)
        rows.append(dict(model=model, n=n, method="greedy pool", size=len(pool),
                         **retrieval_metrics(Ev[s], Et[s])))
        rows.append(dict(model=model, n=n, method="full model", size="all",
                         **retrieval_metrics(full_v[s], full_t[s])))
    return pool, trace, pd.DataFrame(rows)


def get_args_parser():
    p = argparse.ArgumentParser("greedy pool selection", add_help=False)
    p.add_argument("--models", nargs="+", default=["ViT-B-32"])
    p.add_argument("--tasks", nargs="+", default=["fairface", "CIFAR100"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per_class", default=1, type=int)
    p.add_argument("--max_size", default=64, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--tag", default="",
                   help="suffix for the output files, so a focused re-run cannot overwrite a "
                        "previous multi-task run")
    p.add_argument("--seed_topk", default=0, type=int,
                   help="prefilter each tower to its top-k solo components before the pair "
                        "search (0 = exhaustive)")
    p.add_argument("--cal_reps", default=1, type=int,
                   help="repeat the selection on this many independent calibration draws")
    p.add_argument("--retrieval", action="store_true", help="also run the COCO scaling test")
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    os.makedirs(OUT, exist_ok=True)
    for m in a.models:
        ev, tr, po = [], [], []
        for t in a.tasks:
            for rep in range(a.cal_reps):
                cs = a.seed + rep
                pool, trace, rows = run_task(m, t, a.device, a.seed, a.per_class, a.max_size,
                                             cal_seed=cs, seed_topk=a.seed_topk)
                ev.append(rows); tr.append(trace.assign(model=m, task=t, cal_seed=cs))
                po += [dict(model=m, task=t, cal_seed=cs, rank=i, tower=x[0], kind=x[1][0],
                            layer=x[1][1], head=x[1][2]) for i, x in enumerate(pool)]
        if a.retrieval:
            pool, trace, rows = retrieval_scaling(m, None, a.device, a.seed, max_size=a.max_size,
                                                 seed_topk=a.seed_topk)
            rows.to_csv(os.path.join(OUT, f"greedy_pool_retrieval_{m}{a.tag}.csv"), index=False)
            tr.append(trace.assign(model=m, task="coco"))
            po += [dict(model=m, task="coco", rank=i, tower=x[0], kind=x[1][0], layer=x[1][1],
                        head=x[1][2]) for i, x in enumerate(pool)]
            print(rows.to_string(index=False))
        pd.concat(ev, ignore_index=True).to_csv(
            os.path.join(OUT, f"greedy_pool_eval_{m}{a.tag}.csv"), index=False)
        pd.concat(tr, ignore_index=True).to_csv(
            os.path.join(OUT, f"greedy_pool_trace_{m}{a.tag}.csv"), index=False)
        pd.DataFrame(po).to_csv(os.path.join(OUT, f"greedy_pool_pool_{m}{a.tag}.csv"), index=False)
        print(f"[{m}] wrote greedy_pool_{{eval,trace,pool}}", flush=True)
