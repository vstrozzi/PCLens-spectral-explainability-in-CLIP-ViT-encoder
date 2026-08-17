"""Concept-directed selection: name what to keep and what to remove, use no labels at all.

This extends PCLens QuerySystem + PCSelection in two ways, and both come from the same
observation - that ranking a direction by |cos(p, e_c)| alone is not enough.

WHY |cos| IS NOT ENOUGH
    QuerySystem ranks a PC by how well its DIRECTION aligns with a concept. But a direction the
    model barely uses moves nothing: a PC perfectly aligned with "water background" whose
    coefficient alpha is near-constant contributes no discrimination at all. Alignment says where a
    unit points; it does not say how far it pushes.

    So the score here multiplies alignment by ACTUAL USE. For any unit u with per-sample delta
    d_u(m) - the exact quantity mean-ablation removes - and a unit-norm concept vector e_c:

        e_u(m,c) = <d_u(m), e_c>                     effect on the cosine similarity to c
        strength_u(c) = std_m e_u(m,c)               how much it MOVES that similarity

    At PC level this factorises, because d_u(m) = alpha_u(m) * p_u:

        strength_u(c) = std_m[alpha_u] * <p_u, e_c>
                        \_____________/  \_________/
                         how much it is   where it
                         used             points

    which is O(1) per unit once the coefficients exist, and - crucially - is expressed in units of
    cosine-similarity change, so it is COMPARABLE across PCs, across components, and across the two
    granularities. |cos| is not comparable in that way.

WHAT YOU SPECIFY
    Two sets of prompts, no labels:
        keep    "a landbird" / "a waterbird"                 - the distinction to preserve
        remove  "a land background" / "a water background"   - the distinction to destroy
    Each pair becomes a contrast AXIS, e_c1 - e_c2, because what matters is the direction that
    separates them, not either endpoint.

THE OBJECTIVE
    Selection maximises separation along the keep axis and minimises it along the remove axis:

        J(S) = Var_m <E_S(m), keep_axis>  -  lambda * Var_m <E_S(m), remove_axis>

    E_S is the reconstruction with only S kept, everything else mean-ablated. Variance, not
    accuracy, so NO LABELS enter the selection - they are used for evaluation only.

THE THIRD METHOD: QUERY NARROWS, GREEDY REFINES, BOTH TOWERS
    The static top-k needs k tuned. The greedy discovers it, but scanning every unit at PC level is
    expensive. So the third variant does both: the concept strength above draws a CANDIDATE POOL of
    the top-M units per axis, and the greedy then adapts inside that pool.

    It also acts on BOTH ENCODERS, which forces one design consequence worth stating: ablating a
    text unit does not move any image embedding, so an image-only objective would score every text
    move at exactly zero and the greedy would never touch one. The objective therefore carries a
    text term - the two class prompts should sit FAR APART along the keep axis and CLOSE TOGETHER
    along the remove axis:

        J = [Var_m <Ev(m), keep> - lam Var_m <Ev(m), rem>]     (images)
          + [Var_c <Et(c), keep> - lam Var_c <Et(c), rem>]     (class prompts)

    Text PC bases come from a broad corpus, because two class names cannot define one.

WHY GREEDY WITH RECOMPUTATION
    Units interact: once a strongly background-carrying unit is removed, the ranking of the rest
    changes. A static top-k (and the grid search over k that PCLens needs) cannot see that. Each
    step here re-scores every candidate against the CURRENT state and takes the single best move,
    forward or backward, until nothing improves - so k is discovered, not searched.

  python -m scripts_paper.exp_concept --models ViT-B-32 --level pc --device cpu
Writes output_dir/results_paper/6_2/concept_{rank,eval,trace}_{level}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task
from scripts_paper.exp_waterbirds import TASK, accuracies, groups
from scripts_paper.exp_waterbirds_pc import TEXT_SET, pc_basis
from scripts_paper.pclens_core import (ACT_DIR, MODEL_PRETRAINED, MODELS, RES_DIR, comp_vec,
                                       component_index, embed)

OUT = os.path.join(RES_DIR, "6_2")
KEEP = ["a photo of a landbird", "a photo of a waterbird"]
REMOVE = ["a photo of a land background", "a photo of a water background"]


def concept_axes(model, keep, remove, device):
    """Unit-norm contrast axes from the frozen text tower."""
    from utils.models.factory import create_model_and_transforms, get_tokenizer
    m, _, _ = create_model_and_transforms(model, pretrained=MODEL_PRETRAINED[model],
                                          precision="fp32")
    m.eval().to(device)
    tok = get_tokenizer(model)
    with torch.no_grad():
        E = torch.nn.functional.normalize(m.encode_text(tok(keep + remove).to(device)).float(),
                                          dim=1)
    del m
    n = len(keep)
    ax_keep = torch.nn.functional.normalize(E[1] - E[0], dim=0)
    ax_rem = torch.nn.functional.normalize(E[n + 1] - E[n], dim=0)
    return ax_keep, ax_rem, E


@torch.no_grad()
def build_units(attn, mlp, level, k=8, tower="vision", basis_src=None):
    """Per-unit deltas: whole components, or (component, PC) pairs.

    D[u] is the delta the unit contributes - exactly what mean-ablating it removes - and the
    scaffold is everything the units cannot express, so scaffold + sum(all D) reproduces the
    embedding exactly at any k.

    `basis_src` supplies a different (attn, mlp) for estimating the PC directions. The text tower
    needs it: two Waterbirds class names cannot define a basis, so the directions come from a broad
    corpus and the class-name activations are projected onto them."""
    units, D = [], {}
    scaf = embed(attn, mlp).clone()
    src = basis_src if basis_src is not None else (attn, mlp)
    for c in component_index(attn, mlp):
        X = comp_vec(attn, mlp, c)
        mu = X.mean(0)
        if level == "comp":
            d = X - mu
            u = (tower, c, -1)
            units.append(u); D[u] = d; scaf = scaf - d
        else:
            _, P = pc_basis(comp_vec(*src, c), k)
            if P.shape[0] == 0:
                continue
            A = (X - mu) @ P.T
            for j in range(P.shape[0]):
                u = (tower, c, j)
                units.append(u)
                D[u] = A[:, j].unsqueeze(1) * P[j]
                scaf = scaf - D[u]
    return units, D, scaf


def _text_basis_src(model, device, seed=69):
    """Broad-corpus text activations, used only to estimate the text PC directions."""
    ta = torch.from_numpy(np.load(os.path.join(
        ACT_DIR, f"{TEXT_SET}_attn_text_{model}_seed_{seed}.npy"))).to(device)
    p = os.path.join(ACT_DIR, f"{TEXT_SET}_mlp_text_{model}_seed_{seed}.npy")
    tm = torch.from_numpy(np.load(p)).to(device) if os.path.exists(p) else None
    return ta, tm


@torch.no_grad()
def rank_units(units, D, ax_keep, ax_rem, Ev_full=None, Et_full=None):
    """Two quantities per unit per axis, and they answer different questions.

    strength  = std_m <D_u(m), e_c> = |<p_u, e_c>| * std_m[alpha_u]
        HOW MUCH the unit moves similarity to the concept. Sign-invariant for free: flipping the
        SVD sign flips p and alpha together, so the delta alpha*p - and therefore this - does not
        change. A PC is an axis, and this treats it as one without needing an explicit abs().

    share    = Cov_m(<D_u, e_c>, <E, e_c>) / Var_m(<E, e_c>)
        WHICH WAY it pushes, as a signed fraction of the total spread along that axis. This is the
        directional quantity: positive means the unit moves samples the same way the full embedding
        does along that axis, negative means it opposes. Shares sum to 1 across units plus the
        scaffold, so they read as a decomposition of the axis rather than as free-floating numbers.

    A per-unit MEAN over the whole pool would be structurally zero here - alpha is mean-centred by
    construction - so it carries no information and is not reported.

    Strengths are NOT comparable across towers, and the ranking must not pool them. The two towers
    are summarised over different sample sets - thousands of images versus a handful of class
    prompts - and the prompts are the concept endpoints themselves, so text units separate the keep
    axis almost by definition and swamp any joint ranking. `strength_rel` is therefore the
    within-tower share of that tower's largest strength, and every selection below ranks per tower.
    """
    rows = []
    proj_v = None if Ev_full is None else Ev_full @ ax_keep
    projr_v = None if Ev_full is None else Ev_full @ ax_rem
    proj_t = None if Et_full is None else Et_full @ ax_keep
    projr_t = None if Et_full is None else Et_full @ ax_rem
    for u in units:
        ek, er = D[u] @ ax_keep, D[u] @ ax_rem
        pk, pr = (proj_v, projr_v) if u[0] == "vision" else (proj_t, projr_t)
        def share(e, p):
            if p is None or p.numel() < 2 or float(p.var()) < 1e-12:
                return float("nan")
            return float(((e - e.mean()) * (p - p.mean())).mean() / p.var())
        rows.append(dict(tower=u[0], kind=u[1][0], layer=u[1][1], head=u[1][2], pc=u[2],
                         strength_keep=float(ek.std()), strength_remove=float(er.std()),
                         share_keep=share(ek, pk), share_remove=share(er, pr)))
    d = pd.DataFrame(rows)
    # what selection wants: carries the class distinction, not the background one
    d["selectivity"] = d.strength_keep - d.strength_remove
    for col in ("strength_keep", "strength_remove"):
        d[col + "_rel"] = d.groupby("tower")[col].transform(lambda v: v / (v.max() + 1e-12))
    return d


@torch.no_grad()
def objective(Ev, Et, ax_keep, ax_rem, lam=1.0, w_txt=1.0, obj="margin"):
    """Separation along the keep axis, minus separation along the remove axis, on BOTH sides.

    Embeddings are L2-NORMALISED before projecting, and that is not cosmetic. CLIP classifies by
    cosine, so only the direction matters - but mean-ablating a large fraction of the units changes
    the norm a great deal. On raw sums the search can raise the variance simply by inflating the
    norm, which does nothing for a cosine classifier and was exactly what made an earlier version of
    this objective underperform a plain top-k.

    `obj` selects the functional, and the choice matters more than any weighting:

      "margin"  mean_m |<E(m), axis>| - the mean zero-shot MARGIN between the two class prompts,
                since <E,t1> - <E,t2> = <E, t1-t2>. It rewards samples sitting FAR FROM THE
                DECISION BOUNDARY, which is what a cosine classifier actually needs.
      "var"     Var_m <E(m), axis> - rewards SPREAD along the axis. This is the wrong functional:
                variance centres by the mean, so a configuration where every sample is far along
                the axis in the SAME direction scores near zero despite being perfectly decisive.
                Kept only so the comparison can be reported; it lost to a plain static top-k.

    The text term is what makes text units selectable at all - without it every text move scores
    exactly zero, because ablating a text unit leaves the image embeddings untouched."""
    def f(E, ax):
        z = torch.nn.functional.normalize(E, dim=1) @ ax
        return z.abs().mean() if obj == "margin" else z.var()
    J = float(f(Ev, ax_keep) - lam * f(Ev, ax_rem))
    if Et is not None and w_txt:
        J += w_txt * float(f(Et, ax_keep) - lam * f(Et, ax_rem))
    return J


@torch.no_grad()
def greedy_concept(units, D, scaf_v, scaf_t, ax_keep, ax_rem, lam=1.0, w_txt=1.0, init="empty",
                   allow_undo=False, max_steps=10 ** 6, max_undo=100, cand_pool=None,
                   obj="margin", verbose=False):
    """Bidirectional greedy on J, over BOTH towers. No labels are consulted in this function.

    `cand_pool` restricts which units may move - the query step narrows, the greedy refines - while
    every unit outside it stays in whatever state `init` gave it."""
    movable = list(cand_pool) if cand_pool is not None else list(units)
    kept = {u: init == "full" for u in units}
    Ev, Et = scaf_v.clone(), scaf_t.clone()
    for u in units:
        if kept[u]:
            if u[0] == "vision":
                Ev = Ev + D[u]
            else:
                Et = Et + D[u]
    best = objective(Ev, Et, ax_keep, ax_rem, lam, w_txt, obj)
    trace = [dict(step=0, J=best, n_kept=sum(kept.values()))]
    # ungated undo makes forward and backward the same search - they converged to an identical
    # subset before this was gated
    undo_left = max_undo if allow_undo else 0
    for step in range(1, max_steps + 1):
        cand, cand_J, cand_add = None, best, None
        for u in movable:
            add = not kept[u]
            undo = (add and init == "full") or (not add and init == "empty")
            if undo and undo_left <= 0:
                continue
            s_ = 1.0 if add else -1.0
            if u[0] == "vision":
                J = objective(Ev + s_ * D[u], Et, ax_keep, ax_rem, lam, w_txt, obj)
            else:
                J = objective(Ev, Et + s_ * D[u], ax_keep, ax_rem, lam, w_txt, obj)
            if J > cand_J:
                cand, cand_J, cand_add = u, J, add
        if cand is None:
            break
        if (cand_add and init == "full") or (not cand_add and init == "empty"):
            undo_left -= 1
        kept[cand] = cand_add
        s_ = 1.0 if cand_add else -1.0
        if cand[0] == "vision":
            Ev = Ev + s_ * D[cand]
        else:
            Et = Et + s_ * D[cand]
        best = cand_J
        trace.append(dict(step=step, J=best, n_kept=sum(kept.values())))
        if verbose and step % 25 == 0:
            print(f"    step {step} J={best:.6f} kept={sum(kept.values())}", flush=True)
    return [u for u in units if kept[u]], pd.DataFrame(trace)


@torch.no_grad()
def reconstruct(D, scaf_v, scaf_t, sel):
    Ev, Et = scaf_v.clone(), scaf_t.clone()
    for u in sel:
        if u[0] == "vision":
            Ev = Ev + D[u]
        else:
            Et = Et + D[u]
    return Ev, Et


def _acc(Ev, Et, y, place):
    return {a: b for a, b in accuracies(Ev, Et, y, place).items() if a != "groups"}


def run_model(model, level="pc", device="cpu", seed=69, k=8, lam=1.0, w_txt=1.0, obj="margin",
              topk=(10, 25, 50, 100), pool_M=200, towers=("vision", "text"),
              max_steps=10 ** 6, n_random=3):
    av, mv, at, mt, _ = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    ax_keep, ax_rem, _ = concept_axes(model, KEEP, REMOVE, device)

    units, D, scaf_v = build_units(av, mv, level, k, tower="vision")
    scaf_t = embed(at, mt).clone()
    if "text" in towers:
        tu, tD, scaf_t = build_units(at, mt, level, k, tower="text",
                                     basis_src=_text_basis_src(model, device, seed)
                                     if level == "pc" else None)
        units = units + tu
        D.update(tD)

    full = _acc(embed(av, mv), embed(at, mt), y, place)
    n_v = sum(1 for u in units if u[0] == "vision")
    print(f"[{model}/{level}] {len(units)} units ({n_v} vision, {len(units) - n_v} text); "
          f"full model total={full['total']:.2f} worst-group={full['worst_group']:.2f}", flush=True)

    rk = rank_units(units, D, ax_keep, ax_rem, embed(av, mv), embed(at, mt))\
        .assign(model=model, level=level)
    rk["idx"] = range(len(rk))
    for tw in sorted(set(rk.tower)):
        g = rk[rk.tower == tw]
        print(f"  [{tw}] top 3 by KEEP strength (the class axis):", flush=True)
        for r in g.nlargest(3, "strength_keep").itertuples():
            nm = f"MLP {r.layer}" if r.kind == "mlp" else f"L{r.layer}H{r.head}"
            print(f"    {nm:9s} PC{r.pc:<3d} keep={r.strength_keep:.4f} "
                  f"remove={r.strength_remove:.4f}  share_keep={r.share_keep:+.3f}", flush=True)
        print(f"  [{tw}] top 3 by REMOVE strength (the spurious axis):", flush=True)
        for r in g.nlargest(3, "strength_remove").itertuples():
            nm = f"MLP {r.layer}" if r.kind == "mlp" else f"L{r.layer}H{r.head}"
            print(f"    {nm:9s} PC{r.pc:<3d} remove={r.strength_remove:.4f} "
                  f"keep={r.strength_keep:.4f}  share_remove={r.share_remove:+.3f}", flush=True)

    rows = [dict(model=model, level=level, method="full model", variant="-", size=len(units),
                 **full)]

    # ---- static baselines: the PCLens-style top-k, which needs k tuned
    def per_tower_top(col, kk):
        """kk units FROM EACH TOWER - a joint nlargest would return text units only."""
        return set(rk.groupby("tower", group_keys=False)
                   .apply(lambda g: g.nlargest(min(kk, len(g)), col)).idx)

    for kk in topk:
        if kk >= len(units):
            continue
        keepi = per_tower_top("strength_keep", kk)
        Ev, Et = reconstruct(D, scaf_v, scaf_t, [units[i] for i in keepi])
        rows.append(dict(model=model, level=level, method=f"keep top-{kk}", variant="static",
                         size=len(keepi), **_acc(Ev, Et, y, place)))
        # vision only: the text tower has few components and they all carry the 2-class
        # classifier, so dropping any of them collapses it (total fell to chance before this)
        dropi = set(rk[rk.tower == "vision"].nlargest(kk, "strength_remove").idx)
        sel = [u for i, u in enumerate(units) if i not in dropi]
        Ev, Et = reconstruct(D, scaf_v, scaf_t, sel)
        rows.append(dict(model=model, level=level, method=f"drop top-{kk} (vision)",
                         variant="static", size=len(sel), **_acc(Ev, Et, y, place)))

    # ---- greedy on the concept objective: over everything, then over the narrowed pool
    traces = []
    pool_idx = per_tower_top("strength_keep", pool_M) | per_tower_top("strength_remove", pool_M)
    pool = [units[i] for i in sorted(pool_idx)]
    variants = (("forward", "empty", False), ("forward+swap", "empty", True),
                ("backward", "full", False), ("backward+swap", "full", True))
    for tag, cand in (("all units", None), (f"query pool M={pool_M}", pool)):
        for variant, init, undo in variants:
            sel, tr = greedy_concept(units, D, scaf_v, scaf_t, ax_keep, ax_rem, lam, w_txt,
                                     init=init, allow_undo=undo, max_steps=max_steps,
                                     cand_pool=cand, obj=obj)
            Ev, Et = reconstruct(D, scaf_v, scaf_t, sel)
            a = _acc(Ev, Et, y, place)
            rows.append(dict(model=model, level=level, method=f"greedy-{obj} ({tag})", variant=variant,
                             size=len(sel), **a))
            traces.append(tr.assign(model=model, level=level, variant=variant, pool=tag))
            print(f"  [greedy {tag:18s} {variant:8s}] kept={len(sel):5d}/{len(units)} "
                  f"total={a['total']:.2f} worst-group={a['worst_group']:.2f}", flush=True)
            rng = np.random.default_rng(seed)
            for q in range(n_random):
                r = [units[i] for i in rng.choice(len(units), size=len(sel), replace=False)]
                Ev, Et = reconstruct(D, scaf_v, scaf_t, r)
                rows.append(dict(model=model, level=level, method="random", variant=variant,
                                 size=len(r), **_acc(Ev, Et, y, place)))
    os.makedirs(OUT, exist_ok=True)
    rk.to_csv(os.path.join(OUT, f"concept_rank_{level}_{model}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"concept_eval_{level}_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"concept_trace_{level}_{model}.csv"), index=False)
    print(f"[{model}/{level}] wrote concept_{{rank,eval,trace}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("concept-directed selection", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--level", default="pc", choices=["comp", "pc"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", default=8, type=int, help="PCs per component")
    p.add_argument("--lam", default=1.0, type=float, help="weight on the remove axis")
    p.add_argument("--w_txt", default=1.0, type=float, help="weight on the text term")
    p.add_argument("--obj", default="margin", choices=["margin", "var"])
    p.add_argument("--pool_M", default=200, type=int, help="candidate pool size per axis")
    p.add_argument("--towers", nargs="+", default=["vision", "text"])
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.level, a.device, a.seed, a.k, a.lam, a.w_txt, a.obj, pool_M=a.pool_M,
                  towers=tuple(a.towers), max_steps=a.max_steps)
