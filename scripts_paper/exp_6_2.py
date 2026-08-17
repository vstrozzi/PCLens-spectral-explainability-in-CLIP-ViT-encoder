"""Paper section 6.2 - Where are the top components?

Measures COCO-Karpathy-test 1-to-1 retrieval (image<->caption, 5000 pairs) while
mean-ablating components of either tower, in several orderings:

  layer-level   forward / backward / random(5)   x  {attn heads, mlp}  x {vision, text}
  component     metricB / metricA / norm / random(5), both as cumulative ABLATION
                (most-important first, and the reverse) and cumulative KEEP-ONLY
                (reconstruction: everything mean-ablated, add the top-K back)
  joint         the same component ranking pooled over BOTH towers

Rankings:
  metricB  literal Eq.(19): B(a) = E_{i, j~neg}[ r_a(i,j) ],
           r_a = exp( (1/tau) * w_a(i) * [cos(c_a(i), y_j) - cos(c_a(i), y_i)] ),
           tau = 1/logit_scale of the checkpoint. B < 1 => the component pushes positives
           above negatives (discriminative); ascending B = most important first.
  metricA  user's PVE-weighted Hungarian PC-cosine, max over the other tower's components.
  norm     E[w_a] = E[||c_a||] (loudness) - the control that separates loud from informative.

Mean source for the ablation: "self" (COCO's own mean, headline) and "ref" (the wide
D_I=ImageNet / D_T=top_1500_nouns means, robustness check).

  python -m scripts_paper.exp_6_2 --models ViT-B-32 [--device cuda:0]
Writes output_dir/results_paper/6_2/{retrieval_ablation,component_scores}_{model}.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, A_metric, ablated_embedding,
                                       align_geo, comp_delta, comp_vec, component_index, embed,
                                       load_tower, logit_scale, metric_B_component, pcs_of,
                                       retrieval_metrics, sample_negatives)

OUT = os.path.join(RES_DIR, "6_2")
REF_TEXT_SET = "top_1500_nouns_5_sentences_imagenet_bias_clean"


# --------------------------------------------------------------------------- rankings
@torch.no_grad()
def component_scores(av, mv, at, mt, model, device, n_neg=64, kmax=8, seed=69):
    """Per-component metricB / metricA / norm for BOTH towers -> tidy DataFrame."""
    tau_inv = logit_scale(model)
    X, Y = embed(av, mv), embed(at, mt)
    n = X.shape[0]
    neg = sample_negatives(n, n_neg, device, seed)

    comps = {"vision": component_index(av, mv), "text": component_index(at, mt)}
    acts = {"vision": (av, mv), "text": (at, mt)}
    other = {"vision": Y, "text": X}

    # PC bundles once per component (kmax leading dirs, 99% PVE cap)
    pcs = {}
    for tower, cl in comps.items():
        a, m = acts[tower]
        for c in cl:
            U, rho, sv2, _ = pcs_of(comp_vec(a, m, c), var=0.99, kmax=kmax)
            pcs[(tower, c)] = (U, rho, sv2)

    # metric A / Lambda_geo: cross-tower, keep max over partners as the per-component score
    bestA = {k: 0.0 for k in pcs}
    bestG = {k: 0.0 for k in pcs}
    for cv in comps["vision"]:
        Ua, ra, sa = pcs[("vision", cv)]
        for ct in comps["text"]:
            Ub, rb, sb = pcs[("text", ct)]
            a_ = A_metric(Ua, ra, Ub, rb)
            g_ = align_geo(Ua, sa, Ub, sb)
            if a_ > bestA[("vision", cv)]:
                bestA[("vision", cv)] = a_
            if a_ > bestA[("text", ct)]:
                bestA[("text", ct)] = a_
            if g_ > bestG[("vision", cv)]:
                bestG[("vision", cv)] = g_
            if g_ > bestG[("text", ct)]:
                bestG[("text", ct)] = g_

    rows = []
    for tower, cl in comps.items():
        a, m = acts[tower]
        for c in cl:
            cv_ = comp_vec(a, m, c)
            B, logB = metric_B_component(cv_, other[tower], tau_inv, neg)
            rows.append(dict(model=model, tower=tower, kind=c[0], layer=c[1], head=c[2],
                             metricB=B, metricB_log=logB,
                             metricA=bestA[(tower, c)], lambda_geo=bestG[(tower, c)],
                             norm=float(cv_.norm(dim=-1).mean())))
    return pd.DataFrame(rows)


def ranked_components(scores, tower, by):
    """(tower, component) pairs ordered most-important-first according to `by`.

    For "joint" the two towers' components are pooled and compete on one ranking, so the
    tower tag has to travel with the component (both towers own e.g. ('attn', 10, 4))."""
    df = scores if tower == "joint" else scores[scores.tower == tower]
    asc = (by in ("metricB", "metricB_log"))          # r_a < 1 == discriminative
    df = df.sort_values(by, ascending=asc)
    return [(r.tower, (r.kind, int(r.layer), int(r.head))) for r in df.itertuples()]


# --------------------------------------------------------------------------- curves
@torch.no_grad()
def layer_curves(av, mv, at, mt, means, model, seed=69, n_random=5):
    """Cumulative per-layer mean ablation, forward/backward/random, per tower and kind."""
    rows = []
    towers = {"vision": (av, mv), "text": (at, mt)}
    for mean_source, (mv_a, mv_m, mt_a, mt_m) in means.items():
        mean = {"vision": (mv_a, mv_m), "text": (mt_a, mt_m)}
        for tower, (a, m) in towers.items():
            other = "text" if tower == "vision" else "vision"
            oa, om = towers[other]
            for kind in ("attn", "mlp"):
                if kind == "mlp" and m is None:
                    continue
                n_layers = a.shape[1] if kind == "attn" else m.shape[1]
                heads = range(a.shape[2]) if kind == "attn" else [-1]
                layer_comps = [[(kind, l, h) for h in heads] for l in range(n_layers)]

                orders = {"forward": [list(range(n_layers))],
                          "backward": [list(range(n_layers))[::-1]],
                          "random": []}
                rng = np.random.default_rng(seed)
                for _ in range(n_random):
                    o = list(range(n_layers)); rng.shuffle(o); orders["random"].append(o)

                Yo = embed(oa, om)
                for order_name, seqs in orders.items():
                    for rep, seq in enumerate(seqs):
                        X = embed(a, m)               # running embedding, ablated incrementally
                        for step in range(n_layers + 1):
                            met = (retrieval_metrics(X, Yo) if tower == "vision"
                                   else retrieval_metrics(Yo, X))
                            rows.append(dict(model=model, mean_source=mean_source, tower=tower,
                                             kind=kind, order=order_name, rep=rep, step=step,
                                             ablated=str(seq[:step]), **met))
                            if step < n_layers:
                                for c in layer_comps[seq[step]]:
                                    X = X - comp_delta(a, m, c, *mean[tower])
    return pd.DataFrame(rows)


@torch.no_grad()
def component_curves(av, mv, at, mt, scores, means, model, n_steps=20, seed=69, n_random=3):
    """Cumulative component ablation / keep-only along each ranking, for each tower and jointly."""
    rows = []
    towers = {"vision": (av, mv), "text": (at, mt)}
    rankings = ["metricB", "metricA", "norm"]

    for mean_source, (mv_a, mv_m, mt_a, mt_m) in means.items():
        mean = {"vision": (mv_a, mv_m), "text": (mt_a, mt_m)}
        for tower in ("vision", "text", "joint"):
            targets = ["vision", "text"] if tower == "joint" else [tower]
            n_comp = sum(len(component_index(*towers[t])) for t in targets)
            # linear grid + a fine head: almost all of the drop happens in the first few percent,
            # so the concentration curve needs resolution there, and each extra probe is one matmul
            steps = set(np.round(np.linspace(0, n_comp, n_steps)).astype(int).tolist())
            steps |= {s for s in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
                      if s <= n_comp}

            orders = {r: [ranked_components(scores, tower, r)] for r in rankings}
            orders["metricB_rev"] = [orders["metricB"][0][::-1]]
            rng = np.random.default_rng(seed)
            orders["random"] = []
            for _ in range(n_random):
                o = list(orders["metricB"][0]); rng.shuffle(o); orders["random"].append(o)

            for order_name, seqs in orders.items():
                for rep, seq in enumerate(seqs):
                    for mode in ("ablate", "keep"):
                        keep = (mode == "keep")
                        # running embeddings, updated one component at a time
                        ems = {t: (ablated_embedding(*towers[t], [], *mean[t], keep_only=True)
                                   if (keep and t in targets) else embed(*towers[t]))
                               for t in ("vision", "text")}
                        for step in range(n_comp + 1):
                            if step in steps:
                                met = retrieval_metrics(ems["vision"], ems["text"])
                                rows.append(dict(model=model, mean_source=mean_source,
                                                 tower=tower, order=order_name, mode=mode,
                                                 rep=rep, step=step,
                                                 frac=float(step) / n_comp, **met))
                            if step < n_comp:
                                t, c = seq[step]
                                d = comp_delta(*towers[t], c, *mean[t])
                                ems[t] = ems[t] + d if keep else ems[t] - d
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- driver
def ref_means(model, av, mv, at, mt, device, seed=69):
    """Means of the wide reference sets D_I (imagenet) / D_T (top_1500 nouns), shape-matched."""
    def _m(path, like):
        if not os.path.exists(path):
            return like.mean(0)
        arr = np.load(path, mmap_mode="r")
        return torch.from_numpy(np.asarray(arr.mean(0))).to(device=like.device, dtype=like.dtype)

    return (_m(os.path.join(ACT_DIR, f"imagenet_attn_{model}_seed_{seed}.npy"), av),
            _m(os.path.join(ACT_DIR, f"imagenet_mlp_{model}_seed_{seed}.npy"), mv) if mv is not None else None,
            _m(os.path.join(ACT_DIR, f"{REF_TEXT_SET}_attn_text_{model}_seed_{seed}.npy"), at),
            _m(os.path.join(ACT_DIR, f"{REF_TEXT_SET}_mlp_text_{model}_seed_{seed}.npy"), mt) if mt is not None else None)


def run_model(model, device="cuda:0", seed=69, n_steps=20, skip_ref=False):
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    print(f"[{model}] vision {tuple(av.shape)} + mlp {None if mv is None else tuple(mv.shape)} | "
          f"text {tuple(at.shape)} + mlp {None if mt is None else tuple(mt.shape)}")

    base = retrieval_metrics(embed(av, mv), embed(at, mt))
    print(f"[{model}] baseline retrieval: " +
          " ".join(f"{k}={v:.2f}" for k, v in base.items() if k.startswith("R@")))
    pd.DataFrame([dict(model=model, **base)]).to_csv(
        os.path.join(OUT, f"baseline_{model}.csv"), index=False)

    sc = component_scores(av, mv, at, mt, model, device, seed=seed)
    sc.to_csv(os.path.join(OUT, f"component_scores_{model}.csv"), index=False)

    means = {"self": (av.mean(0), None if mv is None else mv.mean(0),
                      at.mean(0), None if mt is None else mt.mean(0))}
    if not skip_ref:
        means["ref"] = ref_means(model, av, mv, at, mt, device, seed)

    lc = layer_curves(av, mv, at, mt, means, model, seed=seed)
    lc.to_csv(os.path.join(OUT, f"layer_ablation_{model}.csv"), index=False)

    cc = component_curves(av, mv, at, mt, sc, means, model, n_steps=n_steps, seed=seed)
    cc.to_csv(os.path.join(OUT, f"component_ablation_{model}.csv"), index=False)
    print(f"[{model}] wrote layer_ablation ({len(lc)} rows), component_ablation ({len(cc)} rows)")
    return base


def get_args_parser():
    p = argparse.ArgumentParser("6.2 retrieval ablation", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", default=69, type=int)
    p.add_argument("--n_steps", default=20, type=int)
    p.add_argument("--skip_ref", action="store_true", help="only the COCO-mean ablation")
    return p


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    for m in args.models:
        run_model(m, args.device, args.seed, args.n_steps, args.skip_ref)
