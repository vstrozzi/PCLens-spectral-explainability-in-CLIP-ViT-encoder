"""Paper section 6.3 - Intrinsic dimensionality of the components.

For every unit of analysis, on both towers of every model, over the COCO activations:
  linear ID    #PCs reaching 80 / 95 / 99 % of the variance          (pca80/95/99)
  nonlinear ID TwoNN maximum-likelihood estimator (Facco et al.)     (twonn)
  ratio        pca95 / twonn - the "linear inflation" of the manifold
  evr1         explained-variance ratio of the leading PC
  geometry     ||mean - mean(image embeddings)||, ||mean - mean(text embeddings)||,
               |cos(mean, PC1)|, ||mean||, E[||c||]

Units ("level"):
  component   a single MSA head or MLP slot
  layer       the cumulative residual stream up to and including layer l (all components so far)
  final       the encoder output itself

  python -m scripts_paper.exp_6_3 --models ViT-B-32
Writes output_dir/results_paper/6_3/intrinsic_dim_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.pclens_core import (MODELS, RES_DIR, comp_vec, component_index, embed,
                                       load_tower, pcs_of)

OUT = os.path.join(RES_DIR, "6_3")


@torch.no_grad()
def twonn(X, frac=0.9, max_n=5000, seed=69):
    """TwoNN intrinsic dimension (Facco et al. 2017), MLE on the r2/r1 ratios.

    Discards the top (1-frac) of ratios, which is the standard robustness step, and fits
    the linear relation log(1 - F(mu)) = -d log(mu) through the origin."""
    if X.shape[0] > max_n:
        g = torch.Generator().manual_seed(seed)
        X = X[torch.randperm(X.shape[0], generator=g)[:max_n]]
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    r, _ = torch.sort(D, dim=1)
    r1, r2 = r[:, 0], r[:, 1]
    ok = (r1 > 0) & torch.isfinite(r2)
    mu = (r2[ok] / r1[ok]).double()
    mu, _ = torch.sort(mu)
    n = mu.numel()
    keep = int(frac * n)
    F = torch.arange(1, n + 1, dtype=torch.float64, device=mu.device) / n
    x = torch.log(mu[:keep])
    y = -torch.log1p(-F[:keep].clamp(max=1 - 1e-12))
    return float((x @ y) / (x @ x).clamp_min(1e-30))       # least squares through origin


def _dist(mu, ref):
    """||mu - ref||, or NaN when the two do not share a space.

    Pre-projection the two towers live in their own widths (e.g. 768 vs 512), so the
    cross-tower centroid distance is undefined there - only the same-tower one is meaningful."""
    return float((mu - ref).norm()) if ref is not None and ref.shape == mu.shape else float("nan")


@torch.no_grad()
def stats_for(X, ref_img_mean, ref_txt_mean, level, **tags):
    """All the section-6.3 quantities for one activation matrix X [N, d]."""
    mu = X.mean(0)
    Xc = X - mu
    S = torch.linalg.svdvals(Xc)
    pve = (S ** 2) / (S ** 2).sum().clamp_min(1e-30)
    cum = torch.cumsum(pve, 0)
    npc = {f"pca{int(v*100)}": int(torch.searchsorted(cum, torch.tensor(v, device=cum.device)).item()) + 1
           for v in (0.80, 0.95, 0.99)}
    U, _, _, _ = pcs_of(X, var=0.99, kmax=1)
    tn = twonn(X)
    return dict(level=level, **tags, **npc, twonn=tn,
                ratio=npc["pca95"] / max(tn, 1e-9), evr1=float(pve[0]),
                d=int(X.shape[1]), n=int(X.shape[0]),
                mean_norm=float(mu.norm()), act_norm=float(X.norm(dim=-1).mean()),
                dist_img_mean=_dist(mu, ref_img_mean),
                dist_txt_mean=_dist(mu, ref_txt_mean),
                cos_mean_pc1=float((mu / mu.norm().clamp_min(1e-12) @ U[0]).abs()))


@torch.no_grad()
def run_model(model, device="cuda:0", towers=("vision", "text"), act_dir=None, space="post"):
    """space="post": shared-space components (default). space="pre": the same statistics on the
    pre-projection residual stream (encoder width), for the 'before vs after projection' question."""
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device, act_dir=act_dir)
    at, mt = load_tower(model, "text", device=device, act_dir=act_dir)
    X, Y = embed(av, mv), embed(at, mt)
    ref_i, ref_t = X.mean(0), Y.mean(0)          # the two modality centroids (modality gap)

    acts = {"vision": (av, mv), "text": (at, mt)}
    rows = []
    for tower in towers:
        a, m = acts[tower]
        for kind, l, h in component_index(a, m):
            rows.append(stats_for(comp_vec(a, m, (kind, l, h)), ref_i, ref_t, "component",
                                  model=model, tower=tower, kind=kind, layer=l, head=h))
        # what each layer contributes ON ITS OWN (not cumulative): the sum of that layer's heads,
        # its MLP, and the two together - three values per layer, mirroring the cumulative view
        for l in range(a.shape[1]):
            heads = a[:, l].sum(dim=1)
            rows.append(stats_for(heads, ref_i, ref_t, "layer_own",
                                  model=model, tower=tower, kind="attn", layer=l, head=-1))
            if m is not None and l < m.shape[1]:
                mlp = m[:, l]
                rows.append(stats_for(mlp, ref_i, ref_t, "layer_own",
                                      model=model, tower=tower, kind="mlp", layer=l, head=-1))
                rows.append(stats_for(heads + mlp, ref_i, ref_t, "layer_own",
                                      model=model, tower=tower, kind="total", layer=l, head=-1))
        # cumulative residual stream after each layer (heads of layers <=l  +  mlps of layers <=l)
        for l in range(a.shape[1]):
            cum = a[:, :l + 1].sum(dim=(1, 2))
            if m is not None:
                cum = cum + m[:, :min(l + 2, m.shape[1])].sum(dim=1)
            rows.append(stats_for(cum, ref_i, ref_t, "layer",
                                  model=model, tower=tower, kind="cumulative", layer=l, head=-1))
        rows.append(stats_for(embed(a, m), ref_i, ref_t, "final",
                              model=model, tower=tower, kind="embedding", layer=-1, head=-1))
        print(f"[{model}/{tower}] done ({len(rows)} rows so far)")

    df = pd.DataFrame(rows).assign(space=space)
    out = os.path.join(OUT, f"intrinsic_dim_{model}{'' if space == 'post' else '_pre'}.csv")
    df.to_csv(out, index=False)
    print(f"[{model}] wrote {out}")
    return df


def get_args_parser():
    p = argparse.ArgumentParser("6.3 intrinsic dimensionality", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--act_dir", default=None, help="activation dir override (pre-projection run)")
    p.add_argument("--space", default="post", choices=["post", "pre"])
    return p


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    for m in args.models:
        run_model(m, args.device, act_dir=args.act_dir, space=args.space)
