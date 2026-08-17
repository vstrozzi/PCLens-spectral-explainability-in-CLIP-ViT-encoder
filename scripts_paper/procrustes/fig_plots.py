"""Figures for the Procrustes-alignment / circuit study."""
import os, json
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

OUT = os.path.dirname(__file__)
mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9.5,
    "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.8,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
# diverging: blue (destroyed/-) -> neutral -> red (enforced/+)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, RED = "#2a78d6", "#e34948"

def sym_norm(A):
    """Signed-log diverging norm: reveals mid-range structure under the dominant
    MLP-corner entries (interaction matrices span ~2 orders of magnitude)."""
    v = float(np.abs(A).max()) + 1e-9
    return dict(cmap=DIV, norm=SymLogNorm(linthresh=v / 80, vmin=-v, vmax=v, base=10))

# ---------------------------------------------------------------- accuracy figure
def fig_accuracy():
    res = json.load(open(f"{OUT}/procrustes_results.json"))
    models = list(res.keys())
    mats = ["Wpos", "Wneg", "Wpos@Wneg", "Wneg@Wpos", "Wpos@Wneg^T"]
    cols = [("imagenet", "imagenet", "IN\nfit·IN"),
            ("CIFAR100", "CIFAR100", "C100\nfit·C100"),
            ("imagenet", "CIFAR100", "C100\nIN→C100"),
            ("CIFAR100", "imagenet", "IN\nC100→IN")]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25], wspace=0.42)
    for mi, model in enumerate(models):
        ax = fig.add_subplot(gs[0, mi])
        Z = np.zeros((len(mats), len(cols)))
        for j, (src, evd, _) in enumerate(cols):
            base = res[model]["baseline"][evd]
            for i, m in enumerate(mats):
                Z[i, j] = res[model]["fit"][src][evd][m] - base
        im = ax.imshow(Z, aspect="auto", **sym_norm(Z))
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2] for c in cols], fontsize=7.5)
        ax.set_yticks(range(len(mats))); ax.set_yticklabels(mats, fontsize=8.5)
        for i in range(len(mats)):
            for j in range(len(cols)):
                ax.text(j, i, f"{Z[i,j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if abs(Z[i,j]) < 0.35 else "white")
        ax.set_title(f"{model}\nΔ zero-shot acc  (matrix − baseline)", fontsize=9.5)
    # held-out generalisation gap
    ax = fig.add_subplot(gs[0, 2])
    labels, base, tr, te = [], [], [], []
    for model in models:
        for ds in ["imagenet", "CIFAR100"]:
            s = res[model]["split"][ds]
            labels.append(f"{model.split('-',1)[1]}\n{ds[:4]}")
            base.append(s["base_test"]); tr.append(s["Wpos_train"]); te.append(s["Wpos_test"])
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, base, w, color="#b8b7b2", label="baseline")
    ax.bar(x, tr, w, color=RED, label="Wpos — TRAIN (fit set)")
    ax.bar(x + w, te, w, color=BLUE, label="Wpos — TEST (held-out)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("zero-shot accuracy"); ax.set_ylim(0, 1.0)
    ax.axhline(0, color="#b8b7b2", lw=0.6)
    ax.set_title("Wpos over-fits: train ↑, held-out ↓", fontsize=9.5)
    ax.legend(fontsize=7.3, frameon=False, loc="upper left")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    fig.suptitle("Orthogonal-Procrustes alignment W — zero-shot accuracy   "
                 "(Wneg and every product with it ≈ 0; gains are transductive only)",
                 fontsize=10.5, y=1.02, weight="bold")
    fig.savefig(f"{OUT}/fig_accuracy.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ------------------------------------------------------------- circuit reshaping
def block_lines(ax, nI, nT):
    ax.axvline(144 - 0.5, color="#0b0b0b", lw=0.9, alpha=0.55)   # attn|mlp image
    ax.axhline(96 - 0.5, color="#0b0b0b", lw=0.9, alpha=0.55)    # attn|mlp text

def fig_circuit(model="ViT-B-16", ds="imagenet"):
    Z = np.load(f"{OUT}/circuit_{model}_{ds}.npz", allow_pickle=True)
    Mbar, Mpar, dM = Z["Mbar"], Z["Mpar"], Z["dM"]
    dimg, dtxt = Z["dimg"], Z["dtxt"]; il, tl = Z["img_lab"], Z["txt_lab"]
    nI, nT = int(Z["nI"]), int(Z["nT"])
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.28)
    for k, (Mx, tt) in enumerate([(Mbar, "M̄  baseline interaction"),
                                  (Mpar, "M̄′  after W (rotate image)"),
                                  (dM, "ΔM̄ = M̄′ − M̄")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(Mx, aspect="auto", **sym_norm(Mbar if k < 2 else dM))
        block_lines(ax, nI, nT)
        ax.set_title(tt, fontsize=10)
        ax.set_xlabel("image components  (0–143 attn heads · 144–156 MLP)", fontsize=8)
        if k == 0: ax.set_ylabel("text components\n(0–95 attn · 96–108 MLP)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # marginals
    ax = fig.add_subplot(gs[1, :2])
    colr = np.where(dimg >= 0, RED, BLUE)
    ax.bar(np.arange(nI), dimg, color=colr, width=0.9)
    ax.axvline(144 - 0.5, color="#0b0b0b", lw=0.9, alpha=0.5)
    ax.set_title("Image-component marginal  Δ(true-class score)   — right block = MLP", fontsize=9.5)
    ax.set_xlabel("image component index"); ax.axhline(0, color="#52514e", lw=0.6)
    for i in np.argsort(np.abs(dimg))[::-1][:8]:
        ax.annotate(il[i], (i, dimg[i]), fontsize=7, ha="center",
                    va="bottom" if dimg[i] >= 0 else "top", color="#0b0b0b")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax = fig.add_subplot(gs[1, 2])
    colr = np.where(dtxt >= 0, RED, BLUE)
    ax.bar(np.arange(nT), dtxt, color=colr, width=0.9)
    ax.axvline(96 - 0.5, color="#0b0b0b", lw=0.9, alpha=0.5)
    ax.set_title("Text-component marginal", fontsize=9.5)
    ax.set_xlabel("text component index"); ax.axhline(0, color="#52514e", lw=0.6)
    for i in np.argsort(np.abs(dtxt))[::-1][:5]:
        ax.annotate(tl[i], (i, dtxt[i]), fontsize=7, ha="center",
                    va="bottom" if dtxt[i] >= 0 else "top", color="#0b0b0b")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    fig.suptitle(f"{model} · ImageNet — how W reshapes the component interaction matrix "
                 f"M(y,c)=A(c)·W·B(y)ᵀ   (red = enforced, blue = destroyed)",
                 fontsize=11, y=0.97, weight="bold")
    fig.savefig(f"{OUT}/fig_circuit_{model}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ------------------------------------------------- sample circuits + 2D geometry
def fig_samples(model="ViT-B-16", ds="imagenet"):
    Z = np.load(f"{OUT}/circuit_{model}_{ds}.npz", allow_pickle=True)
    ids = Z["sample_ids"]; names = Z["sample_names"]; nI, nT = int(Z["nI"]), int(Z["nT"])
    fig = plt.figure(figsize=(13.5, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.36, wspace=0.32)
    # top: per-sample ΔM heatmaps
    dms = [Z[f"Mp_{n}"] - Z[f"M_{n}"] for n in ids]
    vmax = np.percentile([np.abs(d).max() for d in dms], 100)
    for k, n in enumerate(list(ids)[:4]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(dms[k], aspect="auto",
                       cmap=DIV, norm=SymLogNorm(linthresh=vmax / 80, vmin=-vmax, vmax=vmax, base=10))
        block_lines(ax, nI, nT)
        ax.set_title(f"ΔM · “{names[k]}”", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    # bottom-left: cos before vs after (uniform lift, order preserved)
    ax = fig.add_subplot(gs[1, :2])
    cb, ca, lab = Z["scat_cos_before"], Z["scat_cos_after"], Z["scat_lab"]
    ax.scatter(cb, ca, s=5, c="#9ec5f4", alpha=0.45, edgecolors="none", rasterized=True)
    classes = [1, 340, 933, 555, 207, 817]
    for ci, c in enumerate(classes):
        m = lab == c
        ax.scatter(cb[m], ca[m], s=26, c=CAT[ci], edgecolors="white", linewidths=0.4,
                   label=f"cls {c}", zorder=3)
    lo, hi = min(cb.min(), ca.min()), max(cb.max(), ca.max())
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.0, ls="--", label="y = x")
    ax.set_xlabel("cos(true class, f)  — before W"); ax.set_ylabel("cos(true class, W f)  — after W")
    ax.set_title("Per-sample true-class similarity: uniform +Δ lift, ranking preserved", fontsize=9.5)
    ax.legend(fontsize=6.6, frameon=False, ncol=2, loc="lower right")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    # bottom-right: attention-stream vs MLP-stream score, before -> after
    ax = fig.add_subplot(gs[1, 2:])
    ab, aa = Z["scat_attn_before"], Z["scat_attn_after"]
    mb, ma = Z["scat_mlp_before"], Z["scat_mlp_after"]
    ax.scatter(ab, mb, s=7, c=BLUE, alpha=0.35, edgecolors="none", label="before W", rasterized=True)
    ax.scatter(aa, ma, s=7, c=RED, alpha=0.35, edgecolors="none", label="after W", rasterized=True)
    # mean arrow
    ax.annotate("", xy=(aa.mean(), ma.mean()), xytext=(ab.mean(), mb.mean()),
                arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=1.8))
    ax.set_xlabel("attention-stream score  Σ_heads e·b"); ax.set_ylabel("MLP-stream score  Σ_mlp e·b")
    ax.set_title("W inflates the MLP (scaffold) axis; attention axis ~unchanged", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    fig.suptitle(f"{model} · ImageNet — sample circuits and dataset geometry under W",
                 fontsize=11, y=0.98, weight="bold")
    fig.savefig(f"{OUT}/fig_samples_{model}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    fig_accuracy()
    for m in ["ViT-B-16", "ViT-B-32"]:
        fig_circuit(m); fig_samples(m)
    print("figures written:", [f for f in os.listdir(OUT) if f.endswith(".png")])
