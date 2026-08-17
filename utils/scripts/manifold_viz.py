"""
manifold_viz.py — PCA vs MLLE geometry / linearity diagnostics for the PRS
residual-stream decomposition.

Everything (final CLIP embedding + every per-head / per-MLP "atom") lives in the
SAME shared CLIP space, so we can co-embed atoms and outputs with a single
linear (PCA) or nonlinear (MLLE) map and ask: does MLLE recover the same shape
as PCA? If yes, the space is effectively a flat linear subspace.

Loaders follow the same file convention as run_explanations / the `load()` helper
in algorithms_text_explanations_funcs.py:
    {ds}_{kind}{tag}_{model}_seed_{seed}.npy   (tag = "_text" for the text tower)

Image tower : ds = e.g. "imagenet",             final = *_embeddings_*.npy
Text  tower : ds = e.g. "imagenet_classnames",  final = sum of the atoms
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from scipy.spatial import procrustes

import matplotlib.pyplot as plt
from matplotlib import animation


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load(input_dir, ds, kind, model, seed, tag, mmap=False):
    arr = np.load(
        os.path.join(input_dir, f"{ds}_{kind}{tag}_{model}_seed_{seed}.npy"),
        mmap_mode="r" if mmap else None,
    )
    return arr if mmap else arr.astype(np.float32)


def load_atoms(input_dir, dataset, model, seed=69, tower="image",
               components="all", last_n_layers=None):
    """
    Load per-head attention atoms and per-MLP atoms and flatten them into a
    single [M, d] matrix, plus a parallel meta dict describing each row.

    Returns
    -------
    X_atoms : [M, d] float32           every atom (contribution vector)
    meta    : dict of [M] arrays with keys:
                layer     (int)   residual-stream layer index
                unit      (int)   head index (attn) or -1 for an mlp atom
                is_mlp    (bool)
                img       (int)   source image / class index
    final   : [N, d] float32           final embeddings (one per image / class)
    labels  : [N] int  or None         class labels (image tower only)
    """
    tag = "_text" if tower == "text" else ""
    ds = f"{dataset}_classnames" if tower == "text" else dataset

    # memory-map: ViT-L-14 attn is ~5.6 GB; only the last-N-layer slice is copied
    attn = _load(input_dir, ds, "attn", model, seed, tag, mmap=True)   # [N, L, H, d]
    mlp = _load(input_dir, ds, "mlp", model, seed, tag, mmap=True)     # [N, Lm, d]
    N, L, H, d = attn.shape
    Lm = mlp.shape[1]

    la = 0 if last_n_layers is None else max(0, L - last_n_layers)
    lm = 0 if last_n_layers is None else max(0, Lm - last_n_layers)

    rows, m_layer, m_unit, m_ismlp, m_img = [], [], [], [], []

    if components in ("attn", "all"):
        sub = np.ascontiguousarray(attn[:, la:, :, :]).astype(np.float32)  # copy slice only
        Lp = sub.shape[1]
        rows.append(sub.reshape(N * Lp * H, d))
        img = np.repeat(np.arange(N), Lp * H)
        lyr = np.tile(np.repeat(np.arange(la, L), H), N)
        hd = np.tile(np.tile(np.arange(H), Lp), N)
        m_layer.append(lyr); m_unit.append(hd)
        m_ismlp.append(np.zeros(N * Lp * H, bool)); m_img.append(img)

    if components in ("mlp", "all"):
        sub = np.ascontiguousarray(mlp[:, lm:, :]).astype(np.float32)  # [N, Lm', d]
        Lp = sub.shape[1]
        rows.append(sub.reshape(N * Lp, d))
        img = np.repeat(np.arange(N), Lp)
        lyr = np.tile(np.arange(lm, Lm), N)
        m_layer.append(lyr); m_unit.append(np.full(N * Lp, -1))
        m_ismlp.append(np.ones(N * Lp, bool)); m_img.append(img)

    X_atoms = np.concatenate(rows, 0)
    meta = dict(
        layer=np.concatenate(m_layer),
        unit=np.concatenate(m_unit),
        is_mlp=np.concatenate(m_ismlp),
        img=np.concatenate(m_img),
    )

    # final embeddings
    if tower == "image":
        final = _load(input_dir, dataset, "embeddings", model, seed, "")
        try:
            labels = np.load(
                os.path.join(input_dir, f"{dataset}_labels_{model}_seed_{seed}.npy")
            )
        except FileNotFoundError:
            labels = None
    else:
        # text final embedding = sum over all atoms of an item (L2-normalized)
        final = attn.sum((1, 2)) + mlp.sum(1)
        final = final / np.linalg.norm(final, axis=1, keepdims=True)
        labels = None

    return X_atoms, meta, final, labels


def subsample_atoms(X, meta, budget, seed=0):
    """Randomly keep ~budget atoms (uniform over rows). Returns X', meta'."""
    if X.shape[0] <= budget:
        return X, meta
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], budget, replace=False)
    return X[idx], {k: v[idx] for k, v in meta.items()}


def subsample_rows(X, budget, seed=0):
    if X.shape[0] <= budget:
        return X, np.arange(X.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], budget, replace=False)
    return X[idx], idx


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_pca(X_fit, n_components=3):
    p = PCA(n_components=n_components).fit(X_fit)
    return p


def fit_mlle(X_fit, k, n_components=3, seed=0):
    m = LocallyLinearEmbedding(
        n_neighbors=k, n_components=n_components, method="modified",
        eigen_solver="dense", random_state=seed,
    ).fit(X_fit)
    return m


def procrustes_score(A, B):
    """Symmetric Procrustes disparity in [0,1]; ~0 => same shape up to
    translation/scale/rotation/reflection => the manifold is ~linear."""
    n = min(len(A), len(B))
    _, _, disparity = procrustes(A[:n], B[:n])
    return disparity


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def shared_limits(arrays, pad=0.05):
    """Global (min,max) per axis across a list of [n,k] point sets."""
    allp = np.concatenate([np.asarray(a) for a in arrays], 0)
    lo, hi = allp.min(0), allp.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return list(zip(lo - pad * span, hi + pad * span))


def scatter_2d(ax, pts, color, limits=None, title="", cmap="viridis",
               s=4, alpha=0.5, cbar_label=None):
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=color, cmap=cmap, s=s,
                    alpha=alpha, linewidths=0)
    if limits is not None:
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dim 1", fontsize=8); ax.set_ylabel("dim 2", fontsize=8)
    return sc


def overlay_final_2d(ax, final_pts, color="crimson", s=6):
    ax.scatter(final_pts[:, 0], final_pts[:, 1], c=color, s=s,
               marker="x", alpha=0.8, label="final emb", linewidths=0.6)


def rotate_gif(pts3d, color, path, cmap="viridis", final_pts=None,
               n_frames=60, title="", s=4, alpha=0.5, dpi=90):
    """Save an azimuth-sweep rotating 3D scatter as a GIF."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=color, cmap=cmap,
               s=s, alpha=alpha, linewidths=0)
    if final_pts is not None:
        ax.scatter(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2],
                   c="crimson", s=s * 2, marker="x", linewidths=0.6)
    ax.set_title(title, fontsize=10)

    def update(i):
        ax.view_init(elev=20, azim=i * 360 / n_frames)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=15), dpi=dpi)
    plt.close(fig)
    return path
