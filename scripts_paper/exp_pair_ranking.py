"""Pair-level component ranking from the loss factors r_ab and r_tilde_ab (paper Eq. 19-20).

Score of a cross-encoder component pair (a, b), summed over every positive index i and both
negative indices j (text) and k (image):

    score(a,b) = E_{i,j,k} [ r_ab(i,j) * r~_ab(k,i) ]

Because the stored components are already divided by ||encoder output||, the pair term of View 1
is just an inner product, S_ab(i,j) = <c_a(m_i), d_b(m'_j)>, and

    tau * log[ r_ab(i,j) * r~_ab(k,i) ] = S_ab(i,j) + S_ab(k,i) - 2 S_ab(i,i),

so with j and k independent given i the triple expectation factorises:

    log score(a,b) = LSE_i[ -2 S_ii/tau + (LSE_j S_ij/tau - log N) + (LSE_k S_ki/tau - log N) ] - log N

which is evaluated entirely in log space (the loud scaffold pairs would otherwise overflow fp32).
SMALL score = the pair scores matched pairs above mismatched ones in BOTH directions = "aligned".

Two selections are derived from the [P, Q] score matrix:

  matched  greedy one-to-one matching: walk the pairs in ascending score, keep a pair only if
           neither of its components was already taken. Each ablation step then removes one
           vision AND one text component (this is the new thing).
  min      per-component score = best (smallest) score over partners; ranks components inside a
           tower, directly comparable to the Metric B ordering of section 6.2.

  python -m scripts_paper.exp_pair_ranking --models ViT-B-32
Writes output_dir/results_paper/6_2/{pair_scores,pair_ablation}_{model}.csv
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


@torch.no_grad()
def pair_score_matrix(av, mv, at, mt, tau_inv, n_samples=512, seed=69, q_chunk=64, verbose=True):
    """[P, Q] matrix of log score(a,b), P vision components x Q text components.

    `n_samples` subsamples the positive pairs (cost is P*Q*n^2*d); indices i, j, k all range
    over the same subsample, which keeps the positive diagonal inside the negative pools."""
    dev = av.device
    N = av.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:min(n_samples, N)].to(dev)
    M = idx.numel()
    logM = math.log(M)

    cv = component_index(av, mv)
    ct = component_index(at, mt)
    # [P, M, d] and [Q, M, d] stacks of the subsampled component outputs
    V = torch.stack([comp_vec(av, mv, c)[idx] for c in cv])
    T = torch.stack([comp_vec(at, mt, c)[idx] for c in ct])
    P, Q = V.shape[0], T.shape[0]
    ar = torch.arange(M, device=dev)

    out = torch.empty(P, Q, device=dev)
    for p in range(P):
        for q0 in range(0, Q, q_chunk):
            Tq = T[q0:q0 + q_chunk]                              # [Qc, M, d]
            S = torch.einsum("id,qjd->qij", V[p], Tq) * tau_inv   # [Qc, M, M] = S_ab(i,j)/tau
            diag = S[:, ar, ar]                                   # [Qc, M]  S_ab(i,i)/tau
            lse_j = torch.logsumexp(S, dim=2) - logM              # over text negatives j, per i
            lse_k = torch.logsumexp(S, dim=1) - logM              # over image negatives k, per i
            out[p, q0:q0 + q_chunk] = torch.logsumexp(-2 * diag + lse_j + lse_k, dim=1) - logM
        if verbose and (p + 1) % 50 == 0:
            print(f"  pair scores {p + 1}/{P}")
    return out, cv, ct


@torch.no_grad()
def component_scores_1d_sym(av, mv, at, mt, tau_inv, n_samples=512, seed=69):
    """Per-component Metric B on the SAME subsample, one-directional and symmetric.

    Metric B's product over partners telescopes, since sum_b S_ab(i,j) = <c_a(i), y_j>, so both
    variants keep the loss's own aggregation (sum inside the exponent) and differ from each other
    only in DIRECTION. That isolates the directionality change from the pair-level aggregation
    change, which `pair_min` / `pair_mean` make at the same time:

        B_1d(a)  = E_ij   [ exp( (1/tau) [ <c_a(i), y_j> - <c_a(i), y_i> ] ) ]
        B_sym(a) = E_ijk  [ exp( (1/tau) [ <c_a(i), y_j> + <c_a(k), y_i> - 2 <c_a(i), y_i> ] ) ]

    For a text component the two sides swap, so G is always [image index, text index]."""
    dev = av.device
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(av.shape[0], generator=g)[:min(n_samples, av.shape[0])].to(dev)
    M = idx.numel()
    logM = math.log(M)
    ar = torch.arange(M, device=dev)
    X, Y = embed(av, mv)[idx], embed(at, mt)[idx]      # full normalised outputs

    out = {}
    for tower, (a, m), full_other in (("vision", (av, mv), Y), ("text", (at, mt), X)):
        for c in component_index(a, m):
            Ca = comp_vec(a, m, c)[idx]
            # G[i, j] = <image side at i, text side at j> / tau
            G = (Ca @ full_other.T if tower == "vision" else full_other @ Ca.T) * tau_inv
            diag = G[ar, ar]
            lse_j = torch.logsumexp(G, dim=1) - logM    # text negatives j, per positive i
            lse_k = torch.logsumexp(G, dim=0) - logM    # image negatives k, per positive i
            out[(tower, c)] = (
                float(torch.logsumexp(lse_j - diag, 0) - logM),                 # B_1d
                float(torch.logsumexp(lse_j + lse_k - 2 * diag, 0) - logM))     # B_sym
    return out


def greedy_matching(score, cv, ct):
    """Ascending-score walk keeping only pairs whose two components are both still free."""
    P, Q = score.shape
    order = torch.argsort(score.flatten()).cpu().numpy()
    used_v, used_t, pairs = set(), set(), []
    for flat in order:
        p, q = int(flat // Q), int(flat % Q)
        if p in used_v or q in used_t:
            continue
        used_v.add(p); used_t.add(q)
        pairs.append((cv[p], ct[q], float(score[p, q])))
        if len(pairs) == min(P, Q):
            break
    return pairs


@torch.no_grad()
def curve(av, mv, at, mt, seq, mean, steps, mode="ablate"):
    """Retrieval while cumulatively ablating (or keeping only) `seq` = [(tower, comp), ...]."""
    towers = {"vision": (av, mv), "text": (at, mt)}
    keep = mode == "keep"
    ems = {t: (ablated_embedding(*towers[t], [], *mean[t], keep_only=True) if keep
               else embed(*towers[t])) for t in towers}
    rows = []
    for step in range(len(seq) + 1):
        if step in steps:
            rows.append(dict(step=step, **retrieval_metrics(ems["vision"], ems["text"])))
        if step < len(seq):
            t, c = seq[step]
            d = comp_delta(*towers[t], c, *mean[t])
            ems[t] = ems[t] + d if keep else ems[t] - d
    return rows


def run_model(model, device="cuda:0", n_samples=512, seed=69, n_random=3):
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    tau_inv = logit_scale(model)
    print(f"[{model}] pair scores over {min(n_samples, av.shape[0])} samples, 1/tau={tau_inv:.1f}")

    score, cv, ct = pair_score_matrix(av, mv, at, mt, tau_inv, n_samples, seed)

    # ---- save the pair scores (long form) + the per-component best-partner score
    sc = score.cpu().numpy()
    rows = [dict(model=model, v_kind=a[0], v_layer=a[1], v_head=a[2],
                 t_kind=b[0], t_layer=b[1], t_head=b[2], log_score=float(sc[p, q]))
            for p, a in enumerate(cv) for q, b in enumerate(ct)]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"pair_scores_{model}.csv"), index=False)

    pairs = greedy_matching(score, cv, ct)
    pd.DataFrame([dict(model=model, rank=r, v_kind=a[0], v_layer=a[1], v_head=a[2],
                       t_kind=b[0], t_layer=b[1], t_head=b[2], log_score=s)
                  for r, (a, b, s) in enumerate(pairs)]).to_csv(
        os.path.join(OUT, f"pair_matched_{model}.csv"), index=False)
    print(f"[{model}] top matched pairs: " +
          ", ".join(f"{a}<->{b} ({s:.2f})" for a, b, s in pairs[:3]))

    # ---- orderings ------------------------------------------------------------------
    orders = {}
    # (1) matched pairs: one vision + one text component per step (the new, paired selection)
    seq = []
    for a, b, _ in pairs:
        seq += [("vision", a), ("text", b)]
    orders["pair_matched"] = [seq]
    # (2)+(3) reduce the pair score to ONE number per component and pool both towers into a
    # single ranked list, ablating one component per step -- i.e. exactly the "joint" protocol
    # the section-6.2 Metric B curves use, so the two orderings are directly comparable.
    #   min  : score with its best partner (the user's "highest score with whatever component")
    #   mean : aggregated over all partners, closer in spirit to Metric B's product over partners
    lse = torch.logsumexp(score, dim=1) - math.log(score.shape[1])   # over text partners
    lse_t = torch.logsumexp(score, dim=0) - math.log(score.shape[0])  # over vision partners
    for name, red_v, red_t in (("pair_min", sc.min(1), sc.min(0)),
                               ("pair_mean", lse.cpu().numpy(), lse_t.cpu().numpy())):
        pooled = ([("vision", cv[p], float(red_v[p])) for p in range(len(cv))] +
                  [("text", ct[q], float(red_t[q])) for q in range(len(ct))])
        orders[name] = [[(t, c) for t, c, _ in sorted(pooled, key=lambda x: x[2])]]
    # (3a) Metric B recomputed on THIS subsample, one-directional and symmetric: same aggregation
    # as metricB, so the gap between these two is purely the directionality change.
    bs = component_scores_1d_sym(av, mv, at, mt, tau_inv, n_samples, seed)
    for k, name in ((0, "metricB_1d"), (1, "metricB_sym")):
        orders[name] = [[tc for tc, _ in sorted(bs.items(), key=lambda kv: kv[1][k])]]
    pd.DataFrame([dict(model=model, tower=t, kind=c[0], layer=c[1], head=c[2],
                       metricB_1d=v[0], metricB_sym=v[1]) for (t, c), v in bs.items()]).to_csv(
        os.path.join(OUT, f"pair_metricB_variants_{model}.csv"), index=False)

    # (3b) the section-6.2 Metric B ordering as originally computed, for continuity
    p_sc = os.path.join(OUT, f"component_scores_{model}.csv")
    if os.path.exists(p_sc):
        d = pd.read_csv(p_sc).sort_values("metricB")
        orders["metricB"] = [[(r.tower, (r.kind, int(r.layer), int(r.head))) for r in d.itertuples()]]
    # (4) random control
    rng = np.random.default_rng(seed)
    orders["random"] = []
    for _ in range(n_random):
        o = list(orders["pair_min"][0]); rng.shuffle(o); orders["random"].append(o)

    # per-component reductions, saved so the ranking can be compared to Metric B directly
    pd.DataFrame([dict(model=model, tower="vision", kind=c[0], layer=c[1], head=c[2],
                       pair_min=float(sc.min(1)[p]), pair_mean=float(lse[p]))
                  for p, c in enumerate(cv)] +
                 [dict(model=model, tower="text", kind=c[0], layer=c[1], head=c[2],
                       pair_min=float(sc.min(0)[q]), pair_mean=float(lse_t[q]))
                  for q, c in enumerate(ct)]).to_csv(
        os.path.join(OUT, f"pair_component_scores_{model}.csv"), index=False)

    mean = {"vision": (av.mean(0), None if mv is None else mv.mean(0)),
            "text": (at.mean(0), None if mt is None else mt.mean(0))}
    n_all = len(cv) + len(ct)
    steps = set(np.round(np.linspace(0, n_all, 20)).astype(int).tolist())
    steps |= {s for s in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64) if s <= n_all}

    rows = []
    for name, seqs in orders.items():
        for rep, s in enumerate(seqs):
            for mode in ("ablate", "keep"):
                for r in curve(av, mv, at, mt, s, mean, steps, mode):
                    rows.append(dict(model=model, order=name, mode=mode, rep=rep,
                                     frac=r["step"] / len(s), **r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"pair_ablation_{model}.csv"), index=False)
    print(f"[{model}] wrote pair_scores / pair_matched / pair_ablation ({len(df)} rows)")
    return df


def get_args_parser():
    p = argparse.ArgumentParser("pair-level ranking", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", default=512, type=int,
                   help="positive pairs subsampled for the triple expectation (cost ~ P*Q*n^2*d)")
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.n_samples, a.seed)
