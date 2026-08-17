"""Loss-restricted component ranking: the contrastive objective evaluated on one component.

Taking the unrolled objective (paper Eqs. 26-27) and keeping a single component `a` gives

    score_a = prod_i [ ( sum_j prod_b r_ab(i,j) ) * ( sum_k prod_b r~_ab(k,i) ) ]

i.e. the value the loss would take if `a` were the only component. Two structural facts:

1. `prod_b` TELESCOPES. Since the stored components are already divided by the output norm,
   S_ab(i,j) = <c_a(i), d_b(j)> and sum_b S_ab(i,j) = <c_a(i), y_j>, so the product over all
   partners is just the component against the other tower's FULL output. No P*Q*M^2 cache of
   r_ab / r~_ab is needed - only the running (mean-ablated) partner embedding, O(M*d).
2. The product over positives becomes a sum of logs. With G = c_a Y^T / tau,

       log score_a = sum_i [ LSE_j(G_ij - G_ii) + LSE_k(G_ki - G_ii) ]

   which is the ONLY difference from a mean-of-exponentials aggregation: the loss weights every
   positive equally, rather than being dominated by the single worst one.

LOWER = the component makes positives more distinguishable = more important.

The same formula with G = c_a d_b^T / tau (one partner instead of all) gives the PAIR score,
used to find the most aligned (vision, text) pairs.

Because score_a depends only on `a` and the OTHER tower, ablating within one tower cannot change
that tower's own scores: greedy re-scoring is a no-op for vision-only / text-only and is applied
only in the joint protocol (asserted at runtime).

  python -m scripts_paper.exp_loss_ranking --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/loss_{scores,pairs,order,ablation}_{model}.csv
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.pclens_core import (MODELS, RES_DIR, ablated_embedding, comp_delta, comp_vec,
                                       component_index, embed, load_tower, logit_scale,
                                       retrieval_metrics)

OUT = os.path.join(RES_DIR, "6_2")


def restricted_loss(G):
    """log of the loss restricted to whatever G describes, per positive pair.

    G[i, j] = <image-side at i, text-side at j> / tau. Row i holds the text negatives j of
    positive i; column i holds the image negatives k of positive i."""
    d = torch.diagonal(G)
    both = torch.logsumexp(G - d[:, None], 1) + torch.logsumexp(G - d[None, :], 0)
    return float(both.sum() / G.shape[0])


@torch.no_grad()
def _G(Ca, other, tau_inv, tower):
    """G always oriented [image index, text index]."""
    return (Ca @ other.T if tower == "vision" else other @ Ca.T) * tau_inv


@torch.no_grad()
def static_scores(av, mv, at, mt, tau_inv, idx):
    """score_a for every component of both towers against the FULL other tower."""
    X, Y = embed(av, mv)[idx], embed(at, mt)[idx]
    out = {}
    for tower, (a, m), other in (("vision", (av, mv), Y), ("text", (at, mt), X)):
        for c in component_index(a, m):
            out[(tower, c)] = restricted_loss(_G(comp_vec(a, m, c)[idx], other, tau_inv, tower))
    return out


@torch.no_grad()
def pair_scores(av, mv, at, mt, tau_inv, idx, q_chunk=32, verbose=True):
    """[P, Q] loss restricted to a single (vision, text) component PAIR."""
    cv, ct = component_index(av, mv), component_index(at, mt)
    V = torch.stack([comp_vec(av, mv, c)[idx] for c in cv])
    T = torch.stack([comp_vec(at, mt, c)[idx] for c in ct])
    M = idx.numel()
    ar = torch.arange(M, device=av.device)
    out = torch.empty(len(cv), len(ct))
    for p in range(len(cv)):
        for q0 in range(0, len(ct), q_chunk):
            S = torch.einsum("id,qjd->qij", V[p], T[q0:q0 + q_chunk]) * tau_inv
            d = S[:, ar, ar]
            both = (torch.logsumexp(S - d[:, :, None], 2) +
                    torch.logsumexp(S - d[:, None, :], 1))
            out[p, q0:q0 + q_chunk] = both.sum(1).cpu() / M
        if verbose and (p + 1) % 50 == 0:
            print(f"  pair scores {p + 1}/{len(cv)}", flush=True)
    return out, cv, ct


@torch.no_grad()
def greedy_order(av, mv, at, mt, tau_inv, idx, mean, verbose=True):
    """Joint ordering with re-scoring after every ablation.

    After a component is mean-ablated its tower's embedding moves, so every component of the
    OTHER tower must be re-scored against the updated partner - that is the whole point of the
    greedy variant. Returns [(tower, comp, score_at_selection), ...]."""
    acts = {"vision": (av, mv), "text": (at, mt)}
    cur = {"vision": embed(av, mv)[idx], "text": embed(at, mt)[idx]}
    remaining = ([("vision", c) for c in component_index(av, mv)] +
                 [("text", c) for c in component_index(at, mt)])
    vecs = {(t, c): comp_vec(*acts[t], c)[idx] for t, c in remaining}
    other = {"vision": "text", "text": "vision"}

    order, scores, dirty = [], {}, {"vision": True, "text": True}
    while remaining:
        for t, c in remaining:
            if dirty[t]:      # only re-score the tower whose PARTNER moved
                scores[(t, c)] = restricted_loss(_G(vecs[(t, c)], cur[other[t]], tau_inv, t))
        dirty = {"vision": False, "text": False}
        best = min(remaining, key=lambda k: scores[k])
        order.append((best[0], best[1], scores[best]))
        remaining.remove(best)
        cur[best[0]] = cur[best[0]] - comp_delta(*acts[best[0]], best[1], *mean[best[0]])[idx]
        dirty[other[best[0]]] = True      # the partner of the OTHER tower just changed
        if verbose and len(order) % 50 == 0:
            print(f"  greedy {len(order)}/{len(order) + len(remaining)}", flush=True)
    return order


@torch.no_grad()
def infonce(X, Y, tau_inv):
    """The symmetric InfoNCE of Eq. (1) on the current (ablated) embeddings - the quantity the
    ranking is derived from, tracked side by side with retrieval so the two can be compared."""
    L = (torch.nn.functional.normalize(X, dim=1) @
         torch.nn.functional.normalize(Y, dim=1).T) * tau_inv
    t = torch.arange(L.shape[0], device=L.device)
    ce = torch.nn.functional.cross_entropy
    return float((ce(L, t) + ce(L.T, t)) / 2)


@torch.no_grad()
def curve(av, mv, at, mt, seq, mean, steps, mode="ablate", tau_inv=100.0):
    """Retrieval + InfoNCE while cumulatively ablating (or keeping only) seq."""
    towers = {"vision": (av, mv), "text": (at, mt)}
    keep = mode == "keep"
    ems = {t: (ablated_embedding(*towers[t], [], *mean[t], keep_only=True) if keep
               else embed(*towers[t])) for t in towers}
    rows = []
    for step in range(len(seq) + 1):
        if step in steps:
            rows.append(dict(step=step, loss=infonce(ems["vision"], ems["text"], tau_inv),
                             **retrieval_metrics(ems["vision"], ems["text"])))
        if step < len(seq):
            t, c = seq[step]
            d = comp_delta(*towers[t], c, *mean[t])
            ems[t] = ems[t] + d if keep else ems[t] - d
    return rows


def _steps(n):
    s = set(np.round(np.linspace(0, n, 20)).astype(int).tolist())
    return s | {x for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64) if x <= n}


def run_model(model, device="cpu", n_samples=256, seed=69, n_random=3, do_pairs=True):
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    tau_inv = logit_scale(model)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(av.shape[0], generator=g)[:min(n_samples, av.shape[0])].to(device)
    mean = {"vision": (av.mean(0), None if mv is None else mv.mean(0)),
            "text": (at.mean(0), None if mt is None else mt.mean(0))}
    print(f"[{model}] loss-restricted ranking, {idx.numel()} samples, 1/tau={tau_inv:.1f}", flush=True)

    # ---- static per-component scores ------------------------------------------------
    st = static_scores(av, mv, at, mt, tau_inv, idx)
    pd.DataFrame([dict(model=model, tower=t, kind=c[0], layer=c[1], head=c[2], score=v)
                  for (t, c), v in st.items()]).to_csv(
        os.path.join(OUT, f"loss_scores_{model}.csv"), index=False)

    # ---- greedy joint ordering (re-scored after every ablation) ---------------------
    gr = greedy_order(av, mv, at, mt, tau_inv, idx, mean)
    pd.DataFrame([dict(model=model, rank=r, tower=t, kind=c[0], layer=c[1], head=c[2], score=s)
                  for r, (t, c, s) in enumerate(gr)]).to_csv(
        os.path.join(OUT, f"loss_order_{model}.csv"), index=False)
    greedy_seq = [(t, c) for t, c, _ in gr]

    # How much does re-scoring actually reshuffle things? A component's score never depends on
    # its OWN tower, so in a single-tower ablation greedy is provably identical to static. In the
    # joint protocol the towers interleave, so each text ablation moves Y and re-scores every
    # vision component (and vice versa) - this measures how much that matters.
    static_joint = sorted(st, key=lambda k: st[k])
    from scipy.stats import spearmanr
    pos_s = {k: i for i, k in enumerate(static_joint)}
    rho = spearmanr([pos_s[k] for k in greedy_seq], range(len(greedy_seq))).statistic
    moved = sum(a != b for a, b in zip(greedy_seq, static_joint))
    print(f"[{model}] greedy vs static joint order: spearman={rho:.4f}, "
          f"{moved}/{len(greedy_seq)} positions differ", flush=True)

    # ---- pair scores + greedy disjoint matching -------------------------------------
    if do_pairs:
        ps, cv, ct = pair_scores(av, mv, at, mt, tau_inv, idx)
        pd.DataFrame([dict(model=model, v_kind=a[0], v_layer=a[1], v_head=a[2], t_kind=b[0],
                           t_layer=b[1], t_head=b[2], score=float(ps[p, q]))
                      for p, a in enumerate(cv) for q, b in enumerate(ct)]).to_csv(
            os.path.join(OUT, f"loss_pairs_{model}.csv"), index=False)
        used_v, used_t, pairs = set(), set(), []
        for flat in torch.argsort(ps.flatten()).numpy():
            p, q = int(flat // len(ct)), int(flat % len(ct))
            if p in used_v or q in used_t:
                continue
            used_v.add(p); used_t.add(q); pairs.append((cv[p], ct[q], float(ps[p, q])))
        pd.DataFrame([dict(model=model, rank=r, v_kind=a[0], v_layer=a[1], v_head=a[2],
                           t_kind=b[0], t_layer=b[1], t_head=b[2], score=s)
                      for r, (a, b, s) in enumerate(pairs)]).to_csv(
            os.path.join(OUT, f"loss_pair_matched_{model}.csv"), index=False)
        print(f"[{model}] top pairs: " +
              ", ".join(f"{a}<->{b} ({s:.1f})" for a, b, s in pairs[:3]), flush=True)

    # ---- evaluations ---------------------------------------------------------------
    rng = np.random.default_rng(seed)
    rows = []
    for tower in ("vision", "text", "joint"):
        pool = [(t, c) for (t, c) in st if tower in ("joint", t)]
        asc = sorted(pool, key=lambda k: st[k])
        orders = {"lossB": [asc], "lossB_rev": [asc[::-1]],
                  "random": [list(rng.permutation(np.array(pool, dtype=object)).tolist())
                             for _ in range(n_random)]}
        if tower == "joint":
            orders["lossB_greedy"] = [greedy_seq]
            orders["lossB_greedy_rev"] = [greedy_seq[::-1]]
            if do_pairs:
                seq = []
                for a, b, _ in pairs:
                    seq += [("vision", a), ("text", b)]
                orders["pair_matched"] = [seq]
        steps = _steps(len(pool))
        for name, seqs in orders.items():
            for rep, s in enumerate(seqs):
                s = [(t, tuple(c)) for t, c in s]
                for mode in ("ablate", "keep"):
                    for r in curve(av, mv, at, mt, s, mean, steps, mode, tau_inv):
                        rows.append(dict(model=model, tower=tower, order=name, mode=mode,
                                         rep=rep, frac=r["step"] / max(len(s), 1), **r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"loss_ablation_{model}.csv"), index=False)
    print(f"[{model}] wrote loss_ablation ({len(df)} rows)", flush=True)
    return df


def get_args_parser():
    p = argparse.ArgumentParser("loss-restricted ranking", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", default=256, type=int)
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--no_pairs", action="store_true", help="skip the O(P*Q*M^2*d) pair matrix")
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.n_samples, a.seed, do_pairs=not a.no_pairs)
