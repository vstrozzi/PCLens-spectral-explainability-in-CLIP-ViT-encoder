"""
intrinsic_dim.py — per-component intrinsic-dimension diagnostics for the PRS
residual-stream decomposition.

For every atomic component of a CLIP encoder (each attention-head output and each
MLP output, per layer) we build its [N, d] point cloud over a dataset and estimate

    L      linear ID  = # principal components for 99% of the variance  (PCA, s**2)
    N      TwoNN nonlinear intrinsic dimension (Facco et al., 2017)
    Ratio  = L / N          (curvature / nonlinearity of the component manifold)
    EVR1   explained-variance ratio of PC1        (= sklearn explained_variance_ratio_[0])
    PC80   # principal components for 80% of the variance
    PC95   # principal components for 95% of the variance

Three "views" of the residual stream are reported per component (parallel-head
aware, i.e. heads inside a layer share the same layer input):

    indiv       the component's OWN output
    cum_before  the running residual stream just BEFORE this component is added
    cum_after   cum_before + this component

Files follow the run_explanations / manifold_viz convention:
    {ds}_{kind}{tag}_{model}_seed_{seed}.npy      (tag = "_text" for the text tower)

Run as a module to (re)build all cached CSVs under
    output_dir/activations_and_datasets_idxs_{seed}/intrinsic_dim/{ds}_{model}_{tower}.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def pca_metrics(X):
    """Return (EVR1, PC80, PC95, PC99) from the variance spectrum (s**2)."""
    Xc = X - X.mean(0)
    if not np.isfinite(Xc).all() or np.allclose(Xc, 0):
        return (np.nan, np.nan, np.nan, np.nan)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2
    tot = var.sum()
    if tot <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    cum = np.cumsum(var / tot)
    nfor = lambda t: int(np.searchsorted(cum, t) + 1)
    return float(var[0] / tot), nfor(0.80), nfor(0.95), nfor(0.99)


def twonn(X, cap=1500, seed=0):
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    Uses the 1st/2nd nearest-neighbour distance ratio and a line fit through the
    origin of the empirical CDF (dropping the top 10% of ratios). Row-subsampled
    to `cap` points for tractability.
    """
    n0 = len(X)
    if n0 < 10:
        return np.nan
    if n0 > cap:
        X = X[np.random.default_rng(seed).choice(n0, cap, replace=False)]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    m = r1 > 0
    if m.sum() < 20:
        return np.nan
    mu = np.sort(r2[m] / r1[m])
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = max(2, int(0.9 * n))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    denom = float(np.sum(x * x))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _row(view, X, twonn_cap, seed):
    e1, p80, p95, p99 = pca_metrics(X)
    n2 = twonn(X, cap=twonn_cap, seed=seed)
    ratio = p99 / n2 if (n2 == n2 and p99 == p99 and n2 > 0) else np.nan
    return dict(view=view, L=p99, N=n2, Ratio=ratio,
                EVR1=e1, PC80=p80, PC95=p95, n_points=len(X))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _path(input_dir, ds, kind, model, seed, tag):
    return os.path.join(input_dir, f"{ds}_{kind}{tag}_{model}_seed_{seed}.npy")


def load_components(input_dir, dataset, model, seed=69, tower="image"):
    """Return (attn [N,L,H,d] mmap, mlp [N,Lm,d] mmap). Text tower uses the
    `{dataset}_classnames` files with the `_text` tag."""
    tag = "_text" if tower == "text" else ""
    ds = f"{dataset}_classnames" if tower == "text" else dataset
    attn = np.load(_path(input_dir, ds, "attn", model, seed, tag), mmap_mode="r")
    mlp = np.load(_path(input_dir, ds, "mlp", model, seed, tag), mmap_mode="r")
    return attn, mlp


# --------------------------------------------------------------------------- #
# Per-(model, tower, dataset) table
# --------------------------------------------------------------------------- #
def component_table(input_dir, dataset, model, seed=69, tower="image",
                    views=("indiv", "cum_before", "cum_after"),
                    twonn_cap=1500, verbose=True):
    """Build the tidy per-component intrinsic-dimension table.

    One row per (component, view). `component` is an attention head (kind=attn,
    comp=head index) or an MLP (kind=mlp, comp=-1) of a given layer.
    """
    attn, mlp = load_components(input_dir, dataset, model, seed, tower)
    N, L, H, d = attn.shape
    Lm = mlp.shape[1]
    has_layer_mlp = (Lm == L + 1)            # ViT: mlp[:,0]=emb, mlp[:,l+1]=MLP_l ; RN: Lm==1

    # running residual stream that enters layer l (= emb + all earlier blocks)
    cum_before_layer = np.asarray(mlp[:, 0]).astype(np.float32)   # ln_pre embedding
    rows = []

    for l in range(L):
        attn_l = np.asarray(attn[:, l]).astype(np.float32)        # [N, H, d]
        attn_sum = attn_l.sum(1)                                  # [N, d]
        before_heads = cum_before_layer                          # shared by all heads (parallel)

        for h in range(H):
            comp = attn_l[:, h]                                   # [N, d]
            for v in views:
                if v == "indiv":
                    X = comp
                elif v == "cum_before":
                    X = before_heads
                else:  # cum_after
                    X = before_heads + comp
                r = _row(v, X, twonn_cap, seed)
                r.update(dict(layer=l, comp=h, kind="attn"))
                rows.append(r)

        # MLP of this layer (ViT only); sits after attention in the block
        if has_layer_mlp:
            mlp_l = np.asarray(mlp[:, l + 1]).astype(np.float32)  # [N, d]
            before_mlp = before_heads + attn_sum
            for v in views:
                if v == "indiv":
                    X = mlp_l
                elif v == "cum_before":
                    X = before_mlp
                else:
                    X = before_mlp + mlp_l
                r = _row(v, X, twonn_cap, seed)
                r.update(dict(layer=l, comp=-1, kind="mlp"))
                rows.append(r)
            cum_before_layer = before_mlp + mlp_l                 # = input to layer l+1
        else:
            cum_before_layer = before_heads + attn_sum

        if verbose:
            print(f"    {model:9s} {tower:5s} {dataset:17s} layer {l+1:>2d}/{L}", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(1, "tower", tower)
    df.insert(2, "dataset", dataset)
    cols = ["model", "tower", "dataset", "layer", "comp", "kind", "view",
            "L", "N", "Ratio", "EVR1", "PC80", "PC95", "n_points"]
    return df[cols]


# --------------------------------------------------------------------------- #
# Discovery + batch driver
# --------------------------------------------------------------------------- #
def discover(input_dir, seed=69):
    """Return sorted lists (models, image_datasets) present as *_attn_* files."""
    models, datasets = set(), set()
    for f in os.listdir(input_dir):
        if f.endswith(f"_seed_{seed}.npy") and "_attn_" in f and "_text_" not in f \
           and "_cls_attn_" not in f:
            # {ds}_attn_{model}_seed_{seed}.npy
            base = f[:-len(f"_seed_{seed}.npy")]
            ds, model = base.split("_attn_")
            models.add(model)
            datasets.add(ds)
    return sorted(models), sorted(datasets)


def out_csv(input_dir, dataset, model, tower):
    return os.path.join(input_dir, "intrinsic_dim", f"{dataset}_{model}_{tower}.csv")


def export_comparison(input_dir, seed=69):
    """Aggregate every cached per-combo CSV into two comparison files under
    `intrinsic_dim/`:

    intrinsic_dim_all_components.csv  every component row (all models/towers/datasets)
    intrinsic_dim_summary.csv         one row per (model,tower,dataset,view,layer,kind):
                                      attn rows are the MEAN over that layer's heads,
                                      mlp rows are the single MLP component. This is the
                                      layer-aligned unit you can compare across models
                                      (like the reference tables).
    Returns (all_df, summary_df).
    """
    id_dir = os.path.join(input_dir, "intrinsic_dim")
    paths = sorted(f for f in os.listdir(id_dir) if f.endswith(".csv")
                   and not f.startswith("intrinsic_dim_"))
    if not paths:
        raise FileNotFoundError(f"no per-combo CSVs in {id_dir}")
    all_df = pd.concat([pd.read_csv(os.path.join(id_dir, p)) for p in paths],
                       ignore_index=True)

    metrics = ["L", "N", "Ratio", "EVR1", "PC80", "PC95"]
    keys = ["model", "tower", "dataset", "view", "layer", "kind"]
    g = all_df.groupby(keys, as_index=False).agg(
        {**{m: "mean" for m in metrics}, "comp": "count"})
    g = g.rename(columns={"comp": "n_components"})
    g = g.sort_values(keys).reset_index(drop=True)

    all_path = os.path.join(id_dir, "intrinsic_dim_all_components.csv")
    sum_path = os.path.join(id_dir, "intrinsic_dim_summary.csv")
    all_df.to_csv(all_path, index=False)
    g.to_csv(sum_path, index=False)
    print("wrote", all_path, f"({len(all_df):,} rows)")
    print("wrote", sum_path, f"({len(g):,} rows)")
    return all_df, g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",
                    default="output_dir/activations_and_datasets_idxs_69")
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--towers", nargs="+", default=["image", "text"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--twonn_cap", type=int, default=1500)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    models, datasets = discover(args.input_dir, args.seed)
    if args.models:
        models = [m for m in models if m in args.models]
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]
    os.makedirs(os.path.join(args.input_dir, "intrinsic_dim"), exist_ok=True)

    print("models  :", models)
    print("datasets:", datasets)
    print("towers  :", args.towers)

    for model in models:
        for ds in datasets:
            for tower in args.towers:
                dst = out_csv(args.input_dir, ds, model, tower)
                if os.path.exists(dst) and not args.overwrite:
                    print("skip (cached):", dst, flush=True)
                    continue
                try:
                    if not os.path.exists(_path(
                            args.input_dir,
                            f"{ds}_classnames" if tower == "text" else ds,
                            "attn", model, args.seed,
                            "_text" if tower == "text" else "")):
                        print("skip (no data):", ds, model, tower, flush=True)
                        continue
                    print("==>", ds, model, tower, flush=True)
                    df = component_table(args.input_dir, ds, model, args.seed,
                                         tower, twonn_cap=args.twonn_cap)
                    df.to_csv(dst, index=False)
                    print("saved:", dst, "rows", len(df), flush=True)
                except Exception as e:  # keep the sweep going
                    print("FAILED:", ds, model, tower, "->", repr(e), flush=True)

    try:
        export_comparison(args.input_dir, args.seed)
    except Exception as e:
        print("export_comparison skipped:", repr(e), flush=True)


if __name__ == "__main__":
    main()
