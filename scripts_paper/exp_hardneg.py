"""Component selection on matched caption PAIRS - three documented CLIP failure modes.

Each item is (image, correct caption, minimally-edited wrong caption), so chance is exactly 50% and
a score near chance means the model is blind to the one thing the edit changed:

  negation  the object word is present in BOTH captions, only the polarity differs
  count     the number word is swapped
  order     the two noun phrases around a relation are swapped

The calibration objective is the pairwise margin the task actually scores,

    L = mean_n softplus( -(1/tau) * [ cos(v_n, t_n^+) - cos(v_n, t_n^-) ] )

which is the binary cross-entropy of the pair and is well defined for a single calibration item.
The same four local-search selectors as Waterbirds run over it; everything outside the kept set is
mean-ablated, and the selector never sees the evaluation items.

  python -m scripts_paper.exp_hardneg --tasks negation count order --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/hardneg_{eval,trace,pool}_{task}_{model}.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts_paper.exp_greedy_pool import _mean_state
from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_delta, component_index,
                                       embed, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
TASKS = ["negation", "count", "order"]


def _npy(name, device):
    p = os.path.join(ACT_DIR, name)
    return torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None


def load_pairs(model, task, device, seed=69, max_items=2000):
    """Image activations for the referenced COCO images, plus the 2 captions per item."""
    meta = json.load(open(os.path.join(ACT_DIR, f"coco_hardneg_{task}_meta.json")))[:max_items]
    img = torch.tensor([m["img"] for m in meta], device=device)
    lab = torch.tensor([m["label"] for m in meta], device=device)
    av = _npy(f"coco_attn_{model}_seed_{seed}.npy", device)[img]
    mv = _npy(f"coco_mlp_{model}_seed_{seed}.npy", device)
    if mv is not None:
        mv = mv[img]
    n_txt = 2 * len(meta)
    at = _npy(f"coco_hardneg_{task}_attn_text_{model}_seed_{seed}.npy", device)[:n_txt]
    mt = _npy(f"coco_hardneg_{task}_mlp_text_{model}_seed_{seed}.npy", device)
    if mt is not None:
        mt = mt[:n_txt]
    groups = np.array([m["group"] for m in meta])
    return av, mv, at, mt, lab, groups


@torch.no_grad()
def _margin(Ev, Et):
    """(1/tau)-free margin between the correct and the edited caption, one value per item."""
    v = F.normalize(Ev, dim=1)
    t = F.normalize(Et, dim=1)
    return (v * t[0::2]).sum(1) - (v * t[1::2]).sum(1)


@torch.no_grad()
def _loss(Ev, Et, tau_inv):
    return float(F.softplus(-tau_inv * _margin(Ev, Et)).mean())


@torch.no_grad()
def scores(Ev, Et, lab, groups=None):
    m = _margin(Ev, Et)
    ok = (m > 0).float()
    out = dict(acc=float(ok.mean() * 100), margin=float(m.mean()))
    for name, sel in (("acc_present", lab == 1), ("acc_absent", lab == 0)):
        out[name] = float(ok[sel].mean() * 100) if sel.any() else float("nan")
    if groups is not None:
        per = [float(ok[torch.from_numpy(groups == g).to(ok.device)].mean() * 100)
               for g in np.unique(groups)]
        out["worst_group"] = min(per) if per else float("nan")
    return out


@torch.no_grad()
def local_search(av, mv, at, mt, tau_inv, cal, init="empty", allow_swap=False,
                 max_steps=10 ** 6, seed_topk=32, max_undo=100):
    """Same four selectors as Waterbirds, over the pairwise-margin loss.

    Text rows are per ITEM here (two captions each), so the calibration slice has to take rows
    2n and 2n+1 for every calibration item n. `max_undo` budgets moves against the primary
    direction so a swap-enabled run cannot oscillate; every run still goes to convergence."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    tsel = torch.stack([2 * cal, 2 * cal + 1], dim=1).reshape(-1)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    dl = {}
    for t in acts:
        sel = cal if t == "vision" else tsel
        for c in component_index(*acts[t]):
            dl[(t, c)] = comp_delta(*acts[t], c, *mean[t])[sel]
    base_v, base_t = _mean_state(av, mv, at, mt, cal.numel(), tsel.numel())

    kept = {k: init == "full" for k in comps}
    cur_v = base_v + sum((dl[k] for k in comps if k[0] == "vision" and kept[k]),
                         torch.zeros_like(base_v))
    cur_t = base_t + sum((dl[k] for k in comps if k[0] == "text" and kept[k]),
                         torch.zeros_like(base_t))

    if init == "empty":
        # with every text component ablated the two captions of a pair are identical, so the margin
        # is 0 and the loss is log(2) whatever single component is added: seed with a pair
        cv = [k for k in comps if k[0] == "vision"]
        ct = [k for k in comps if k[0] == "text"]
        if seed_topk:
            fv, ft = embed(av, mv)[cal], embed(at, mt)[tsel]
            solo = {k: (_loss(base_v + dl[k], ft, tau_inv) if k[0] == "vision"
                        else _loss(fv, base_t + dl[k], tau_inv)) for k in comps}
            cv = sorted(cv, key=lambda k: solo[k])[:seed_topk]
            ct = sorted(ct, key=lambda k: solo[k])[:seed_topk]
        best, best_l = None, None
        for a in cv:
            ev = base_v + dl[a]
            for b in ct:
                l = _loss(ev, base_t + dl[b], tau_inv)
                if best_l is None or l < best_l:
                    best, best_l = (a, b), l
        for k in best:
            kept[k] = True
            if k[0] == "vision":
                cur_v = cur_v + dl[k]
            else:
                cur_t = cur_t + dl[k]
        cur, seeded = best_l, [(k, 1.0) for k in best]
    else:
        cur, seeded = _loss(cur_v, cur_t, tau_inv), []

    trace = [dict(step=0, loss=cur, n_kept=sum(kept.values()), applied=seeded)]
    undo_left = max_undo if allow_swap else 0
    for step in range(1, max_steps + 1):
        cand, cand_l, cand_add = None, cur, None
        for k in comps:
            add = not kept[k]
            if ((add and init == "full") or (not add and init == "empty")) and undo_left <= 0:
                continue
            sign = 1.0 if add else -1.0
            l = (_loss(cur_v + sign * dl[k], cur_t, tau_inv) if k[0] == "vision"
                 else _loss(cur_v, cur_t + sign * dl[k], tau_inv))
            if l < cand_l:
                cand, cand_l, cand_add = k, l, add
        if cand is None:
            break
        if (cand_add and init == "full") or (not cand_add and init == "empty"):
            undo_left -= 1
        kept[cand] = cand_add
        sign = 1.0 if cand_add else -1.0
        if cand[0] == "vision":
            cur_v = cur_v + sign * dl[cand]
        else:
            cur_t = cur_t + sign * dl[cand]
        cur = cand_l
        trace.append(dict(step=step, loss=cur, n_kept=sum(kept.values()),
                          applied=[(cand, sign)]))
    return [k for k in comps if kept[k]], pd.DataFrame(trace)


@torch.no_grad()
def _delta_fn(av, mv, at, mt):
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    cache = {}

    def d(k):
        if k not in cache:
            cache[k] = comp_delta(*acts[k[0]], k[1], *mean[k[0]])
        return cache[k]
    return d


@torch.no_grad()
def keep_only(av, mv, at, mt, pool):
    d = _delta_fn(av, mv, at, mt)
    Ev, Et = _mean_state(av, mv, at, mt, av.shape[0], at.shape[0])
    for k in pool:
        if k[0] == "vision":
            Ev = Ev + d(k)
        else:
            Et = Et + d(k)
    return Ev, Et


@torch.no_grad()
def replay(av, mv, at, mt, trace, kept0, lab, groups, held):
    d = _delta_fn(av, mv, at, mt)
    Ev, Et = _mean_state(av, mv, at, mt, av.shape[0], at.shape[0])
    for k in kept0:
        if k[0] == "vision":
            Ev = Ev + d(k)
        else:
            Et = Et + d(k)
    hi = held.nonzero(as_tuple=True)[0]
    ti = torch.stack([2 * hi, 2 * hi + 1], dim=1).reshape(-1)
    out = []
    for r in trace.itertuples():
        for k, sign in r.applied:
            if k[0] == "vision":
                Ev = Ev + sign * d(k)
            else:
                Et = Et + sign * d(k)
        out.append(dict(step=r.step,
                        **scores(Ev[hi], Et[ti], lab[hi], groups[hi.cpu().numpy()])))
    return pd.DataFrame(out)


def run_model(model, task, device="cpu", seed=69, n_cal=32, max_steps=10 ** 6, reps=3,
              n_random=3,
              seed_topk=32, max_items=2000):
    av, mv, at, mt, lab, groups = load_pairs(model, task, device, seed, max_items)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    full = scores(embed(av, mv), embed(at, mt), lab, groups)
    print(f"[{model}/{task}] {av.shape[0]} items, {len(comps)} components  |  FULL MODEL "
          f"acc={full['acc']:.2f}%  present={full['acc_present']:.2f}%  "
          f"absent={full['acc_absent']:.2f}%  worst-subgroup={full['worst_group']:.2f}%", flush=True)

    rows = [dict(model=model, task=task, variant="full model", rep=-1, size=len(comps), **full)]
    traces, pools = [], []
    for rep in range(reps):
        g = torch.Generator().manual_seed(seed + rep)
        # stratify the calibration items by label so a one-sided set cannot be picked
        cal = []
        for c in lab.unique():
            w = (lab == c).nonzero(as_tuple=True)[0]
            cal += w[torch.randperm(w.numel(), generator=g)[:n_cal].to(w.device)].tolist()
        cal = torch.tensor(sorted(cal), device=device)
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        from scripts_paper.exp_waterbirds import VARIANTS
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search(av, mv, at, mt, tau_inv, cal, max_steps=max_steps,
                                    seed_topk=seed_topk, **cfg)
            kept0 = [] if cfg["init"] == "empty" else list(comps)
            tj = replay(av, mv, at, mt, tr, kept0, lab, groups, held).merge(
                tr[["step", "loss", "n_kept"]], on="step")
            base = dict(model=model, task=task, variant=variant, rep=rep)
            rows.append(dict(method="selected", size=len(pool), **base,
                             **tj.iloc[-1].drop("step").to_dict()))
            traces.append(tj.assign(**base))
            pools += [dict(**base, rank=i, tower=t, kind=c[0], layer=c[1], head=c[2])
                      for i, (t, c) in enumerate(pool)]
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):4d}/{len(comps)} "
                  f"steps={len(tr) - 1:4d} acc={tj.iloc[-1].acc:.2f}% "
                  f"(present {tj.iloc[-1].acc_present:.2f} / absent {tj.iloc[-1].acc_absent:.2f})",
                  flush=True)
            rng = np.random.default_rng(seed + rep)
            hi = held.nonzero(as_tuple=True)[0]
            ti = torch.stack([2 * hi, 2 * hi + 1], dim=1).reshape(-1)
            for r in range(n_random):
                sel = [comps[i] for i in rng.choice(len(comps), size=len(pool), replace=False)]
                Ev, Et = keep_only(av, mv, at, mt, sel)
                rows.append(dict(method="random pool", size=len(sel), model=model, task=task,
                                 variant=variant, rep=r,
                                 **scores(Ev[hi], Et[ti], lab[hi], groups[hi.cpu().numpy()])))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"hardneg_eval_{task}_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"hardneg_trace_{task}_{model}.csv"), index=False)
    pd.DataFrame(pools).to_csv(os.path.join(OUT, f"hardneg_pool_{task}_{model}.csv"), index=False)
    print(f"[{model}/{task}] wrote hardneg_{{eval,trace,pool}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("hard-negative caption tasks", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=32, type=int, help="calibration items per label")
    p.add_argument("--max_steps", default=10 ** 6, type=int,
                   help="cap; default = run to convergence")
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--max_items", default=2000, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for t in a.tasks:
        for m in a.models:
            run_model(m, t, a.device, a.seed, a.n_cal, a.max_steps, a.reps,
                      seed_topk=a.seed_topk, max_items=a.max_items)
