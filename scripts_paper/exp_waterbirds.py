"""Greedy component selection on Waterbirds - the spurious-correlation task.

Waterbirds pairs a bird class (landbird / waterbird) with a background (land / water); the two are
correlated in training, so CLIP leans on the background. The metric that matters is therefore
WORST-GROUP accuracy over the four (class, background) cells, not the total.

The selector is the same training-free greedy pool as `exp_greedy_pool`, with one change forced by
the task: with two classes the symmetric InfoNCE degenerates (a 2x2 matrix, and with several images
per class the "negative" texts are duplicates of the positive). The calibration objective here is
therefore the zero-shot classification loss itself,

    L = CE( E_img @ E_txt^T * (1/tau),  labels )

with E_txt the C unique class-name embeddings, which is well defined for any number of calibration
images per class and is exactly what zero-shot accuracy measures.

Two calibration regimes, because they assume different amounts of supervision:
  class   `n_cal` images per CLASS       - uses only class labels, the same supervision the rest
                                           of the pipeline assumes;
  group   `n_cal` images per GROUP       - uses the background label too. Not free supervision, so
                                           it is reported as an upper bound rather than a method.

Components outside the pool are mean-ablated, and everything is evaluated on all 2,000 decomposed
test images.

  python -m scripts_paper.exp_waterbirds --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/waterbirds_{eval,trace,pool}_{model}.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task, _mean_state, keep_only
from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_delta, component_index,
                                       embed, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
META = "datasets/waterbird_complete95_forest2water2/metadata.csv"
TASK = "binary_waterbirds"


def groups(seed=69):
    """(y, place) per decomposed image, recovered through the saved subset index map."""
    csv = pd.read_csv(META)
    t = csv[csv.split == 2].reset_index(drop=True)
    j = json.load(open(os.path.join(ACT_DIR, f"{TASK}_idx_to_class_seed_{seed}.json")))
    idx = np.array([r["index"] for r in j])
    y, place = t.y.values[idx], t.place.values[idx]
    assert (y == np.array([r["label"] for r in j])).all(), "index map does not line up with metadata"
    return torch.from_numpy(y).long(), torch.from_numpy(place).long()


@torch.no_grad()
def _ce(Ev, Et, tau_inv, y):
    L = (torch.nn.functional.normalize(Ev, dim=1) @
         torch.nn.functional.normalize(Et, dim=1).T) * tau_inv
    return float(torch.nn.functional.cross_entropy(L, y))


@torch.no_grad()
def accuracies(Ev, Et, y, place):
    """Total, worst-class and worst-group top-1."""
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    ok = (L.argmax(1) == y).float()
    per_class = [float(ok[y == c].mean() * 100) for c in y.unique()]
    per_group = [float(ok[(y == c) & (place == p)].mean() * 100)
                 for c in y.unique() for p in place.unique() if ((y == c) & (place == p)).any()]
    return dict(total=float(ok.mean() * 100), worst_class=min(per_class),
                worst_group=min(per_group), groups=[round(x, 2) for x in per_group])


@torch.no_grad()
def local_search(av, mv, at, mt, tau_inv, cal_idx, cal_y, init="empty", allow_swap=False,
                 max_steps=10 ** 6, seed_topk=32, verbose=True, max_undo=100):
    """Greedy subset search over which components stay un-ablated.

    The state is a subset of KEPT components (everything else mean-ablated); forward and backward
    are the same space entered from opposite ends:

      init="empty"  start with nothing kept and ADD     (the pool selector)
      init="full"   start with everything kept and REMOVE, building a removal pool

    With `allow_swap` each step considers BOTH directions - add one of the remaining or drop one of
    those already chosen - and takes whichever lowers the loss most, so the search can undo an
    earlier decision instead of only extending it. Stops when no single move improves the loss.

    `max_undo` budgets the reversals only: a swap-enabled search may move against its primary
    direction at most that many times, after which it continues as the plain variant. Every run
    still goes to convergence - the budget stops the search burning its steps oscillating between
    two states without stopping it from finishing.

    Returns (kept components, trace)."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    n_cls = at.shape[0]
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    dl = {}
    for t in acts:
        sel = cal_idx if t == "vision" else torch.arange(n_cls, device=av.device)
        for c in component_index(*acts[t]):
            dl[(t, c)] = comp_delta(*acts[t], c, *mean[t])[sel]
    base_v, base_t = _mean_state(av, mv, at, mt, cal_idx.numel(), n_cls)

    kept = {k: init == "full" for k in comps}
    cur_v = base_v + sum((dl[k] for k in comps if k[0] == "vision" and kept[k]),
                         torch.zeros_like(base_v))
    cur_t = base_t + sum((dl[k] for k in comps if k[0] == "text" and kept[k]),
                         torch.zeros_like(base_t))

    if init == "empty":
        # every text component mean-ablated makes all C class embeddings identical, so the loss is
        # log(C) whatever single component is added: the search has to be seeded with a pair.
        cv = [k for k in comps if k[0] == "vision"]
        ct = [k for k in comps if k[0] == "text"]
        if seed_topk:
            fv, ft = embed(av, mv)[cal_idx], embed(at, mt)
            solo = {k: (_ce(base_v + dl[k], ft, tau_inv, cal_y) if k[0] == "vision"
                        else _ce(fv, base_t + dl[k], tau_inv, cal_y)) for k in comps}
            cv = sorted(cv, key=lambda k: solo[k])[:seed_topk]
            ct = sorted(ct, key=lambda k: solo[k])[:seed_topk]
        best, best_l = None, None
        for a in cv:
            ev = base_v + dl[a]
            for b in ct:
                l = _ce(ev, base_t + dl[b], tau_inv, cal_y)
                if best_l is None or l < best_l:
                    best, best_l = (a, b), l
        for k in best:
            kept[k] = True
            if k[0] == "vision":
                cur_v = cur_v + dl[k]
            else:
                cur_t = cur_t + dl[k]
        cur = best_l
        seeded = [(k, 1.0) for k in best]
    else:
        cur = _ce(cur_v, cur_t, tau_inv, cal_y)
        seeded = []

    # `applied` carries the actual moves so the search can be replayed on the full dataset
    trace = [dict(step=0, loss=cur, n_kept=sum(kept.values()), move="init", applied=seeded)]
    undo_left = max_undo if allow_swap else 0
    for step in range(1, max_steps + 1):
        cand, cand_l, cand_add = None, cur, None
        for k in comps:
            add = not kept[k]
            # a move against the primary direction is an "undo" and costs from the budget
            undo = (add and init == "full") or (not add and init == "empty")
            if undo and undo_left <= 0:
                continue
            sign = 1.0 if add else -1.0
            if k[0] == "vision":
                l = _ce(cur_v + sign * dl[k], cur_t, tau_inv, cal_y)
            else:
                l = _ce(cur_v, cur_t + sign * dl[k], tau_inv, cal_y)
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
                          move=("add " if cand_add else "drop ") + str(cand),
                          applied=[(cand, sign)]))
        if verbose and step % 25 == 0:
            print(f"    step {step} loss={cur:.4f} kept={sum(kept.values())}", flush=True)
    return [k for k in comps if kept[k]], pd.DataFrame(trace)


# the four selectors: where the search starts, and whether it may undo an earlier decision
VARIANTS = {
    "forward":       dict(init="empty", allow_swap=False),   # add only
    "forward+swap":  dict(init="empty", allow_swap=True),    # add, or drop one already chosen
    "backward":      dict(init="full",  allow_swap=False),   # remove only (a removal pool)
    "backward+swap": dict(init="full",  allow_swap=True),    # remove, or restore one removed
}


@torch.no_grad()
def replay(av, mv, at, mt, trace, kept0, y, place, held):
    """Held-out accuracy after each move, by replaying the search on the full dataset."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    delta = {}

    def d(k):
        if k not in delta:
            delta[k] = comp_delta(*acts[k[0]], k[1], *mean[k[0]])
        return delta[k]

    Ev, Et = _mean_state(av, mv, at, mt, av.shape[0], at.shape[0])
    for k in kept0:
        if k[0] == "vision":
            Ev = Ev + d(k)
        else:
            Et = Et + d(k)
    out = []
    for r in trace.itertuples():
        for k, sign in r.applied:
            if k[0] == "vision":
                Ev = Ev + sign * d(k)
            else:
                Et = Et + sign * d(k)
        out.append(dict(step=r.step, **{a: b for a, b in
                                        accuracies(Ev[held], Et, y[held], place[held]).items()
                                        if a != "groups"}))
    return pd.DataFrame(out)


def run_model(model, device="cpu", seed=69, n_cal=8, max_size=10 ** 6, n_random=5, reps=3,
              seed_topk=32):
    av, mv, at, mt, y_lab = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    assert (y == y_lab).all(), "label mismatch"
    tau_inv = logit_scale(model)
    n_comp = len(component_index(av, mv)) + len(component_index(at, mt))
    full = accuracies(embed(av, mv), embed(at, mt), y, place)
    print(f"[{model}] full model: total={full['total']:.2f} worst-class={full['worst_class']:.2f} "
          f"worst-group={full['worst_group']:.2f} groups={full['groups']}", flush=True)

    rows = [dict(model=model, regime="-", variant="-", method="full model", size=n_comp, rep=-1,
                 **{k: v for k, v in full.items() if k != "groups"})]
    traces, pools = [], []
    allc = ([("vision", c) for c in component_index(av, mv)] +
            [("text", c) for c in component_index(at, mt)])
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
            held[cal] = False          # evaluate only on images the selector never saw

            for variant, cfg in VARIANTS.items():
                pool, tr = local_search(av, mv, at, mt, tau_inv, cal, y[cal],
                                        max_steps=max_size, seed_topk=seed_topk,
                                        verbose=False, **cfg)
                kept0 = [] if cfg["init"] == "empty" else list(allc)
                tj = replay(av, mv, at, mt, tr, kept0, y, place, held).merge(
                    tr[["step", "loss", "n_kept"]], on="step")
                bi = int(tj.worst_group.idxmax())
                base = dict(model=model, regime=regime, variant=variant, rep=rep)
                rows.append(dict(method="selected", size=len(pool),
                                 **base, **tj.iloc[-1].drop("step").to_dict()))
                # chosen by reading the TEST metric: an upper bound on headroom, not a method
                rows.append(dict(method="ORACLE best worst-group",
                                 size=int(tj.loc[bi, "n_kept"]),
                                 **base, **tj.loc[bi].drop("step").to_dict()))
                traces.append(tj.assign(**base))
                pools += [dict(**base, rank=i, tower=t, kind=c[0], layer=c[1], head=c[2])
                          for i, (t, c) in enumerate(pool)]
                print(f"  [{regime} rep{rep} {variant:14s}] kept={len(pool):4d}/{n_comp} "
                      f"steps={len(tr) - 1:4d} total={tj.iloc[-1].total:.2f} "
                      f"worst-group={tj.iloc[-1].worst_group:.2f}", flush=True)

                rng = np.random.default_rng(seed + rep)
                for r in range(n_random):
                    sel = [allc[i] for i in rng.choice(len(allc), size=len(pool), replace=False)]
                    Ev, Et = keep_only(av, mv, at, mt, sel)
                    rows.append(dict(method="random pool", size=len(pool), model=model,
                                     regime=regime, variant=variant, rep=r,
                                     **{k: v for k, v in
                                        accuracies(Ev[held], Et, y[held], place[held]).items()
                                        if k != "groups"}))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"waterbirds_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"waterbirds_trace_{model}.csv"), index=False)
    pd.DataFrame(pools).to_csv(os.path.join(OUT, f"waterbirds_pool_{model}.csv"), index=False)
    print(f"[{model}] wrote waterbirds_{{eval,trace,pool}}", flush=True)

def get_args_parser():
    p = argparse.ArgumentParser("waterbirds greedy pool", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=8, type=int, help="calibration images per class/group")
    # a step CAP is not convergence: leave this unbounded unless deliberately truncating
    p.add_argument("--max_size", default=10 ** 6, type=int)
    p.add_argument("--reps", default=3, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.max_size, reps=a.reps, seed_topk=a.seed_topk)
