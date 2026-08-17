"""Figures for paper sections 6.2-6.4, drawn from the CSVs the exp_6_* scripts write.

Stats and plotting stay decoupled: every function here reads only output_dir/results_paper/**.csv.
Palette is Okabe-Ito (colorblind-safe categorical, fixed order, never cycled); magnitude uses a
single-hue sequential ramp and signed quantities a two-hue diverging ramp with a neutral midpoint.

  python -m scripts_paper.exp_plots --sections 6.2 6.3 6.4
Writes .pdf + .png into output_dir/results_paper/figures/
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from scripts_paper.pclens_core import MODELS, RES_DIR

FIG = os.path.join(RES_DIR, "figures")
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})
# fixed identity -> colour map, assigned in order and never recycled
ORDER_COLOR = {"forward": OKABE_ITO[0], "backward": OKABE_ITO[1], "random": OKABE_ITO[2],
               "lossB": OKABE_ITO[0], "lossB_greedy": OKABE_ITO[3], "lossB_static": OKABE_ITO[4],
               "greedy_forward": OKABE_ITO[0], "greedy_backward": OKABE_ITO[1],
               "metricB": OKABE_ITO[0], "metricA": OKABE_ITO[3], "norm": OKABE_ITO[4],
               "metricB_rev": OKABE_ITO[1], "pc": OKABE_ITO[7], "text": OKABE_ITO[0],
               "image": OKABE_ITO[1], "both": OKABE_ITO[2],
               "pair_min": OKABE_ITO[5], "pair_mean": OKABE_ITO[6], "pair_matched": OKABE_ITO[3]}
R1 = "R@1"          # headline metric: mean of the two retrieval directions


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {os.path.join(FIG, name)}.pdf/.png")


def _csv(section, name, model):
    p = os.path.join(RES_DIR, section, f"{name}_{model}.csv")
    if not os.path.exists(p):
        print(f"[skip] missing {p}")
        return None
    df = pd.read_csv(p)
    if "R@1_i2t" in df:
        df[R1] = (df["R@1_i2t"] + df["R@1_t2i"]) / 2      # symmetric headline
    return df


def _band(ax, g, x, y, label, color):
    """Mean +- std across repetitions, as a line with a translucent band."""
    m = g.groupby(x)[y].mean()
    s = g.groupby(x)[y].std().fillna(0)
    ax.plot(m.index, m.values, lw=2, color=color, label=label,
            marker="o", ms=3.5, markeredgecolor="white", markeredgewidth=0.6)
    if (s > 0).any():
        # R@1 is a percentage: clip the +-std band so it cannot dip below 0
        ax.fill_between(m.index, (m - s).clip(lower=0), m + s, color=color, alpha=0.18, lw=0)


# =========================================================================== 6.2
def fig_layer_ablation(model, mean_source="self"):
    """Fig: retrieval collapse under cumulative per-layer mean ablation (the section-6.2 headline)."""
    df = _csv("6_2", "layer_ablation", model)
    if df is None:
        return
    df = df[df.mean_source == mean_source]
    panels = [("vision", "attn"), ("vision", "mlp"), ("text", "attn"), ("text", "mlp")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for ax, (tower, kind) in zip(axes.flat, panels):
        g = df[(df.tower == tower) & (df.kind == kind)]
        if g.empty:
            ax.set_visible(False); continue
        for order in ("forward", "backward", "random"):
            gg = g[g.order == order]
            if not gg.empty:
                _band(ax, gg, "step", R1, order, ORDER_COLOR[order])
        ax.set_title(f"{tower} tower - {'MSA heads' if kind == 'attn' else 'MLPs'}")
        ax.set_xlabel("# layers mean-ablated"); ax.set_ylabel("COCO R@1 (%)")
    axes.flat[0].legend(title="ablation order", frameon=False)
    fig.suptitle(f"{model}: where the retrieval signal lives  (mean source: {mean_source})", y=1.0)
    fig.tight_layout()
    _save(fig, f"6_2_layer_ablation_{model}_{mean_source}")


def fig_component_ranking(model, mean_source="self", mode="ablate"):
    """Fig: does Metric B pick the components the model actually relies on?"""
    df = _csv("6_2", "component_ablation", model)
    if df is None:
        return
    df = df[(df.mean_source == mean_source) & (df["mode"] == mode)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), sharey=True)
    for ax, tower in zip(axes, ("vision", "text", "joint")):
        g = df[df.tower == tower]
        for order in ("metricB", "metricA", "norm", "metricB_rev", "random"):
            gg = g[g.order == order]
            if not gg.empty:
                _band(ax, gg, "frac", R1, order, ORDER_COLOR.get(order, OKABE_ITO[2]))
        ax.set_title(tower)
        ax.set_xlabel(f"fraction of components {'kept' if mode == 'keep' else 'ablated'}")
    axes[0].set_ylabel("COCO R@1 (%)")
    axes[-1].legend(frameon=False, title="ranking")
    fig.suptitle(f"{model}: cumulative component {'reconstruction' if mode == 'keep' else 'ablation'}"
                 f" by ranking", y=1.04)
    fig.tight_layout()
    _save(fig, f"6_2_component_{mode}_{model}_{mean_source}")


def fig_component_map(model, score="metricB"):
    """Fig: the literal answer to 'where are the top components' - a layer x head map per tower."""
    df = _csv("6_2", "component_scores", model)
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, tower in zip(axes, ("vision", "text")):
        g = df[(df.tower == tower) & (df.kind == "attn")]
        if g.empty:
            ax.set_visible(False); continue
        L, H = g.layer.max() + 1, g["head"].max() + 1
        M = np.full((int(L), int(H)), np.nan)
        for r in g.itertuples():
            M[int(r.layer), int(r.head)] = getattr(r, score)
        # sequential single hue; log scale because E[r_a] spans orders of magnitude
        norm = LogNorm(vmin=max(np.nanmin(M), 1e-3), vmax=np.nanmax(M))
        im = ax.imshow(M, cmap="cividis_r", norm=norm, aspect="auto", origin="lower")
        ax.set_title(f"{tower} tower"); ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=r"$\mathbb{E}[r_a]$  (lower = discriminative)")
        # direct-label the 3 most important heads only
        for r in g.nsmallest(3, score).itertuples():
            ax.text(r.head, r.layer, "*", ha="center", va="center", color="white", fontsize=11)
    fig.suptitle(f"{model}: Metric B per MSA head (stars = top-3)", y=1.02)
    fig.tight_layout()
    _save(fig, f"6_2_component_map_{model}")


def fig_loud_vs_informative(model):
    """Fig: loudness (norm fraction) does not imply pair-discriminative signal - the 'norm hiding'
    claim of the paper, in one scatter."""
    df = _csv("6_2", "component_scores", model)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    handles = []
    for tower, mark in (("vision", "o"), ("text", "^")):
        g = df[df.tower == tower]
        sc = ax.scatter(g["norm"], g.metricB_log, c=g.layer, cmap="viridis", marker=mark,
                        s=26, alpha=0.85, edgecolor="white", linewidth=0.4)
        # shape carries the tower, colour carries depth: keep the legend key neutral
        handles.append(plt.Line2D([], [], marker=mark, ls="", color="0.45",
                                  label=f"{tower} (heads+MLPs)"))
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel(r"loudness  $\mathbb{E}\,\|c_a\|$  (norm fraction)")
    ax.set_ylabel(r"$\mathbb{E}[\log r_a]$   (<0 = discriminative)")
    ax.set_title(f"{model}: loud $\\neq$ informative")
    ax.legend(handles=handles, frameon=False)
    fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout()
    _save(fig, f"6_2_loud_vs_informative_{model}")


def fig_baselines(models=MODELS):
    """Fig: COCO 1-to-1 retrieval baseline per model (context for every other 6.2 figure)."""
    rows = []
    for m in models:
        d = _csv("6_2", "baseline", m)
        if d is not None:
            rows.append(d.assign(model=m))
    if not rows:
        return
    df = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["R@1_i2t"], w, label="image->text", color=OKABE_ITO[0])
    ax.bar(x + w / 2, df["R@1_t2i"], w, label="text->image", color=OKABE_ITO[1])
    for xi, (a, b) in enumerate(zip(df["R@1_i2t"], df["R@1_t2i"])):
        ax.text(xi - w / 2, a + 0.6, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, b + 0.6, f"{b:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df.model, rotation=20)
    ax.set_ylabel("R@1 (%)"); ax.set_title("COCO Karpathy-test 1-to-1 retrieval (5000 pairs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "6_2_baselines")


def fig_concentration(models=MODELS, tower="joint"):
    """Fig: how concentrated is importance? R@1 vs the fraction of components mean-ablated,
    ranked by Metric B, one line per model - the cross-model version of the section's claim."""
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    drawn = 0
    for i, m in enumerate(models):
        df = _csv("6_2", "component_ablation", m)
        if df is None:
            continue
        g = df[(df.mean_source == "self") & (df["mode"] == "ablate") &
               (df.tower == tower) & (df.order == "metricB")].sort_values("frac")
        if g.empty:
            continue
        ax.plot(g.frac * 100, g[R1] / g[R1].iloc[0] * 100, lw=2, marker="o", ms=3,
                color=OKABE_ITO[i % len(OKABE_ITO)], label=m,
                markeredgecolor="white", markeredgewidth=0.5)
        drawn += 1
    if not drawn:
        plt.close(fig); return
    ax.axhline(50, ls="--", lw=1, color="0.5")
    ax.set_xlim(0, 30)
    ax.set_xlabel("% of components mean-ablated (Metric B order)")
    ax.set_ylabel("% of full-model R@1 retained")
    ax.set_title(f"Importance is concentrated in a few components ({tower})")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, f"6_2_concentration_{tower}")


def fig_ranking_quality(models=MODELS, tower="joint", frac=0.1):
    """Fig: which ranking finds the load-bearing components? Retention after ablating the
    top `frac` of components, per ranking, per model (lower = the ranking found them)."""
    rows = []
    for m in models:
        df = _csv("6_2", "component_ablation", m)
        if df is None:
            continue
        g = df[(df.mean_source == "self") & (df["mode"] == "ablate") & (df.tower == tower)]
        if g.empty:
            continue
        base = g[g.step == 0][R1].mean()
        for order in ("metricB", "metricA", "norm", "random"):
            gg = g[g.order == order]
            if gg.empty:
                continue
            j = (gg.frac - frac).abs().idxmin()
            rows.append(dict(model=m, order=order, retained=gg.loc[j, R1] / base * 100))
    if not rows:
        return
    d = pd.DataFrame(rows)
    orders = [o for o in ("metricB", "metricA", "norm", "random") if o in set(d.order)]
    ms = list(dict.fromkeys(d.model))
    x = np.arange(len(ms)); w = 0.8 / len(orders)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, o in enumerate(orders):
        v = [d[(d.model == m) & (d.order == o)].retained.mean() for m in ms]
        ax.bar(x + (i - (len(orders) - 1) / 2) * w, v, w * 0.92, label=o, color=ORDER_COLOR[o])
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=20)
    ax.set_ylabel(f"% R@1 retained after ablating top {frac:.0%}")
    ax.set_title("Lower is better: which ranking finds the load-bearing components")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    _save(fig, f"6_2_ranking_quality_{tower}")


# =========================================================================== 6.3
def fig_pair_ranking(model):
    """Fig: the pair-derived orderings vs Metric B, ablating one pooled component per step.

    All curves use the same joint protocol (both towers in one ranked list, one component
    removed per step), so lower = the ranking found the load-bearing components sooner.
    `pair_matched` instead removes one vision AND one text component per step."""
    d = _csv("6_2", "pair_ablation", model)
    if d is None:
        return
    d = d[d["mode"] == "ablate"].copy()
    d[R1] = (d["R@1_i2t"] + d["R@1_t2i"]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    names = ("metricB", "pair_min", "pair_mean", "pair_matched", "random")
    for ax, xmax in zip(axes, (24, None)):
        for name in names:
            g = d[d.order == name].groupby("step")[R1].mean()
            if g.empty:
                continue
            if xmax:
                g = g[g.index <= xmax]
            ax.plot(g.index, g.values, "--" if name == "random" else "-",
                    color=ORDER_COLOR.get(name, "0.35"), marker="o", ms=2.5, lw=1.4, label=name)
        ax.set_xlabel("# components mean-ablated"); ax.set_ylabel("R@1")
        ax.grid(alpha=.3)
    axes[0].set_title("first 24 components"); axes[1].set_title("all")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: pair-derived ranking vs Metric B (lower = better ranking)", fontsize=9)
    _save(fig, f"6_2_pair_ranking_{model}")


def fig_pair_map(model, topk=10):
    """Fig: where the top matched disjoint pairs sit in the two towers."""
    d = _csv("6_2", "pair_matched", model)
    if d is None:
        return
    d = d.head(topk)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.log_score.values, color=OKABE_ITO[0], height=.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"v {r.v_kind[0]}{r.v_layer}h{r.v_head} - t {r.t_kind[0]}{r.t_layer}h{r.t_head}"
                        for r in d.itertuples()], fontsize=6.5)
    ax.set_xlabel("log score  (lower = more aligned)")
    ax.set_title(f"{model}: top-{topk} matched pairs", fontsize=9)
    ax.grid(alpha=.3, axis="x")
    _save(fig, f"6_2_pair_map_{model}")


def fig_id_table(models=MODELS, tower="vision", cols=("pca95", "twonn", "ratio", "evr1"),
                 labels=("L", "N", "Ratio", "EVR$_1$")):
    """Fig: layer x {L, N, Ratio, EVR1} heat-table, one column-block per model.

    Rows are layers of the cumulative residual stream, so the table reads as 'how the
    representation's dimensionality evolves with depth'. Each column gets its own single-hue
    sequential ramp normalised across all models, i.e. colour encodes magnitude within a quantity."""
    data = {}
    for m in models:
        d = _csv("6_3", "intrinsic_dim", m)
        if d is None:
            continue
        g = d[(d.tower == tower) & (d.level == "layer")].sort_values("layer")
        if not g.empty:
            data[m] = g
    if not data:
        return
    ramps = ["Reds", "Blues", "Greys", "Greens"]
    nmax = max(len(g) for g in data.values())
    fig, axes = plt.subplots(1, len(data), figsize=(2.05 * len(data), 0.22 * nmax + 1.1),
                             squeeze=False)
    vlim = {c: (min(g[c].min() for g in data.values()), max(g[c].max() for g in data.values()))
            for c in cols}
    for ax, (m, g) in zip(axes[0], data.items()):
        M = g[list(cols)].to_numpy()
        ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(g))
        for j, c in enumerate(cols):
            cmap = plt.get_cmap(ramps[j % len(ramps)])
            nrm = Normalize(*vlim[c])
            for i in range(len(g)):
                v = M[i, j]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(0.12 + 0.75 * nrm(v)), lw=0))
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                        ha="center", va="center", fontsize=5.6,
                        color="white" if nrm(v) > 0.62 else "0.15")
        ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(np.arange(len(g)) + 0.5)
        ax.set_yticklabels(g.layer.astype(int), fontsize=5.6)
        ax.set_title(m, fontsize=8); ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel("layer (cumulative residual stream)", fontsize=8)
    fig.suptitle(f"Intrinsic dimensionality by depth - {tower} tower "
                 f"(L = #PCs at 95% var, N = TwoNN, Ratio = L/N)", y=1.02, fontsize=9)
    fig.tight_layout()
    _save(fig, f"6_3_id_table_{tower}")


def fig_loss_ranking(model):
    """Fig: the loss-restricted ranking (product over positives, both directions) vs its
    greedy re-scored variant and the pair-matched selection, joint protocol."""
    d = _csv("6_2", "loss_ablation", model)
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, cmode in zip(axes, ("ablate", "keep")):
        g = d[(d["mode"] == cmode) & (d.tower == "joint")]
        for o in ("lossB", "lossB_greedy", "pair_matched", "random"):
            gg = g[g.order == o]
            if not gg.empty:
                _band(ax, gg, "step", R1, o, ORDER_COLOR.get(o, OKABE_ITO[2]))
        ax.set_xscale("symlog", linthresh=8)
        ax.set_xlabel(f"# components {'ablated' if cmode == 'ablate' else 'kept'}")
        ax.set_ylabel("COCO R@1 (%)")
        ax.set_title("ablation" if cmode == "ablate" else "reconstruction")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: loss-restricted ranking", y=1.03)
    fig.tight_layout()
    _save(fig, f"6_2_loss_ranking_{model}")


def fig_greedy_order(model):
    """Fig: context-aware greedy orderings (forward / backward) vs the marginal ranking."""
    d = _csv("6_2", "greedy_order_ablation", model)
    if d is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0))
    for i, cmode in enumerate(("ablate", "keep")):
        for j, tower in enumerate(("vision", "text", "joint")):
            ax = axes[i][j]
            g = d[(d["mode"] == cmode) & (d.tower == tower)]
            if g.empty:
                ax.set_visible(False); continue
            for o in ("greedy_forward", "greedy_backward", "lossB_static", "random"):
                gg = g[g.order == o]
                if not gg.empty:
                    _band(ax, gg, "step", R1, o, ORDER_COLOR.get(o, OKABE_ITO[2]))
            ax.set_xscale("symlog", linthresh=8)
            if i == 0:
                ax.set_title(tower)
            ax.set_xlabel(f"# components {'ablated' if cmode == 'ablate' else 'kept'}")
            if j == 0:
                ax.set_ylabel(f"{'ablation' if cmode == 'ablate' else 'reconstruction'}\nR@1 (%)")
    axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"{model}: context-aware orderings (chosen given what is already selected)", y=1.02)
    fig.tight_layout()
    _save(fig, f"6_2_greedy_order_{model}")


def fig_pool_trajectory(model, tag=""):
    """Fig: calibration loss and held-out task accuracy against pool size - where the tiny
    calibration set starts to overfit."""
    p = os.path.join(RES_DIR, "6_2", f"greedy_pool_trace_{model}{tag}.csv")
    if not os.path.exists(p):
        print(f"[skip] missing {p}")
        return
    d = pd.read_csv(p)
    tasks = list(dict.fromkeys(d.task))
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.6 * len(tasks), 2.7), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = d[d.task == t]
        m = g.groupby("size")["loss"].mean()
        ax.plot(m.index, m.values, color=OKABE_ITO[0], lw=1.5, label="calibration loss")
        ax.set_xlabel("|pool|"); ax.set_title(t, fontsize=8)
        ax.set_ylabel("calibration loss", color=OKABE_ITO[0])
        ax.tick_params(axis="y", labelcolor=OKABE_ITO[0])
        if "acc" in g and g.acc.notna().any():
            a = g.groupby("size")["acc"].mean()
            ax2 = ax.twinx(); ax2.grid(False)
            ax2.plot(a.index, a.values, color=OKABE_ITO[1], lw=1.5)
            ax2.set_ylabel("task accuracy (%)", color=OKABE_ITO[1])
            ax2.tick_params(axis="y", labelcolor=OKABE_ITO[1])
            ax2.axvline(a.idxmax(), color=OKABE_ITO[1], ls=":", lw=1)
    fig.suptitle(f"{model}: greedy pool - loss keeps falling, accuracy need not", y=1.04)
    fig.tight_layout()
    _save(fig, f"6_2_pool_trajectory_{model}{tag}")


def fig_pool_bars(models=MODELS):
    """Fig: task accuracy of the selected pool vs the full model and same-budget controls."""
    ds = []
    for m in models:
        p = os.path.join(RES_DIR, "6_2", f"greedy_pool_eval_{m}.csv")
        if os.path.exists(p):
            ds.append(pd.read_csv(p))
    if not ds:
        print("[skip] no greedy_pool_eval yet")
        return
    d = pd.concat(ds, ignore_index=True)
    tasks = list(dict.fromkeys(d.task))
    meths = [x for x in ("full model", "greedy pool (best prefix)", "greedy pool",
                         "top-k loss score", "random pool") if x in set(d.method)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.9 * len(tasks), 3.0), squeeze=False,
                             sharey=False)
    ms = [m for m in models if m in set(d.model)]
    for ax, t in zip(axes[0], tasks):
        g = d[d.task == t]
        x = np.arange(len(ms)); w = 0.8 / max(len(meths), 1)
        for i, me in enumerate(meths):
            v = [g[(g.model == m) & (g.method == me)].acc.mean() for m in ms]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, label=me,
                   color=OKABE_ITO[i % len(OKABE_ITO)])
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_title(t, fontsize=8); ax.set_ylabel("top-1 (%)")
    axes[0][0].legend(frameon=False, fontsize=6)
    fig.suptitle("Greedy pool selected on a tiny calibration set, evaluated on the full task",
                 y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_pool_bars")


def fig_pool_retrieval(models=MODELS):
    """Fig: does a pool chosen on 64 COCO pairs survive a larger gallery?"""
    ds = []
    for m in models:
        p = os.path.join(RES_DIR, "6_2", f"greedy_pool_retrieval_{m}.csv")
        if os.path.exists(p):
            ds.append(pd.read_csv(p))
    if not ds:
        print("[skip] no greedy_pool_retrieval yet")
        return
    d = pd.concat(ds, ignore_index=True)
    d[R1] = (d["R@1_i2t"] + d["R@1_t2i"]) / 2
    ms = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(ms), figsize=(2.5 * len(ms), 2.7), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        g = d[d.model == m]
        for meth, c in (("full model", "0.35"), ("greedy pool", OKABE_ITO[0])):
            gg = g[g.method == meth].sort_values("n")
            if not gg.empty:
                ax.plot(gg.n, gg[R1], marker="o", ms=3, lw=1.5, color=c, label=meth)
        ax.set_xscale("log"); ax.set_xlabel("gallery size"); ax.set_title(m, fontsize=8)
    axes[0][0].set_ylabel("R@1 (%)"); axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Pool selected on 64 COCO pairs, tested at increasing gallery size", y=1.04)
    fig.tight_layout()
    _save(fig, "6_2_pool_retrieval")


def fig_waterbirds(models=MODELS):
    """Fig: worst-group accuracy on Waterbirds - the metric the spurious correlation actually
    moves - for the greedy pool against the full model and same-budget random pools."""
    ds = []
    for m in models:
        p = os.path.join(RES_DIR, "6_2", f"waterbirds_eval_{m}.csv")
        if os.path.exists(p):
            ds.append(pd.read_csv(p))
    if not ds:
        print("[skip] no waterbirds_eval yet")
        return
    d = pd.concat(ds, ignore_index=True)
    d = d[~d.method.str.contains("ORACLE")]
    ms = [m for m in models if m in set(d.model)]
    series = [("full model", "-", "0.35"), ("greedy pool", "class", OKABE_ITO[0]),
              ("greedy pool", "group", OKABE_ITO[2]), ("random pool", "class", OKABE_ITO[1])]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for ax, metric in zip(axes, ("worst_group", "total")):
        x = np.arange(len(ms)); w = 0.8 / len(series)
        for i, (meth, reg, c) in enumerate(series):
            v = [d[(d.model == m) & (d.method == meth) & (d.regime == reg)][metric].mean()
                 for m in ms]
            lab = meth if reg in ("-",) else f"{meth} ({reg}-balanced)"
            ax.bar(x + i * w - 0.4 + w / 2, v, w, label=lab, color=c)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_ylabel(f"{metric.replace('_', '-')} accuracy (%)")
        ax.set_title("worst group" if metric == "worst_group" else "total")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Waterbirds: training-free component selection vs the spurious correlation",
                 y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_waterbirds")


def fig_id_layer_own(models=MODELS, col="twonn"):
    """Fig: what each layer contributes on its own (NOT cumulative) - heads of the layer, its
    MLP, and the two together. Same grid as the cumulative view, three curves per cell."""
    d = pd.concat([x for x in (_csv("6_3", "intrinsic_dim", m) for m in models) if x is not None],
                  ignore_index=True) if models else None
    if d is None or "level" not in d or d[d.level == "layer_own"].empty:
        print("[skip] no layer_own rows yet")
        return
    d = d[d.level == "layer_own"]
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(2, len(ms), figsize=(2.5 * len(ms), 5.0), squeeze=False, sharex=False)
    for j, m in enumerate(ms):
        for i, tower in enumerate(("vision", "text")):
            ax = axes[i][j]
            g = d[(d.model == m) & (d.tower == tower)]
            if g.empty:
                ax.set_visible(False); continue
            for kind, lab in (("attn", "heads of layer"), ("mlp", "MLP of layer"),
                              ("total", "layer total")):
                gg = g[g.kind == kind].sort_values("layer")
                if not gg.empty:
                    ax.plot(gg.layer, gg[col], marker="o", ms=2.5, lw=1.4,
                            color=ORDER_COLOR.get({"attn": "metricB", "mlp": "metricA",
                                                   "total": "norm"}[kind]), label=lab)
            if i == 0:
                ax.set_title(m, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{tower}\n{col}")
            ax.set_xlabel("layer")
            ax.grid(alpha=.3)
    axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Per-layer own contribution ({col}, not cumulative)", y=1.01)
    fig.tight_layout()
    _save(fig, f"6_3_id_layer_own_{col}")


def fig_id_components(model):
    """Fig: per-component linear vs nonlinear ID against depth - are late heads low-rank?"""
    d = _csv("6_3", "intrinsic_dim", model)
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, tower in zip(axes, ("vision", "text")):
        g = d[(d.tower == tower) & (d.level == "component") & (d.kind == "attn")]
        if g.empty:
            ax.set_visible(False); continue
        for col, lab, c in (("pca95", "linear (95% var)", OKABE_ITO[0]),
                            ("twonn", "TwoNN", OKABE_ITO[1])):
            _band(ax, g, "layer", col, lab, c)
        ax2 = g.groupby("layer")["ratio"].mean()
        ax.set_title(f"{tower} tower (heads)"); ax.set_xlabel("layer")
        ax.set_ylabel("intrinsic dimension")
        ax.legend(frameon=False)
    fig.suptitle(f"{model}: per-head dimensionality vs depth", y=1.03)
    fig.tight_layout()
    _save(fig, f"6_3_id_components_{model}")


def fig_modality_geometry(model):
    """Fig: where each component sits between the two modality centroids (the modality gap,
    resolved per component instead of per embedding)."""
    d = _csv("6_3", "intrinsic_dim", model)
    if d is None:
        return
    g = d[d.level == "component"]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for tower, c, mark in (("vision", OKABE_ITO[0], "o"), ("text", OKABE_ITO[1], "^")):
        gg = g[g.tower == tower]
        ax.scatter(gg.dist_img_mean, gg.dist_txt_mean, s=24, marker=mark, color=c,
                   alpha=0.75, edgecolor="white", linewidth=0.4, label=f"{tower} components")
    lim = [0, max(g.dist_img_mean.max(), g.dist_txt_mean.max()) * 1.05]
    ax.plot(lim, lim, ls="--", lw=1, color="0.5")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\|\bar c_a - \bar x\|$  (to image centroid)")
    ax.set_ylabel(r"$\|\bar c_a - \bar y\|$  (to text centroid)")
    ax.set_title(f"{model}: components vs the modality gap")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, f"6_3_modality_geometry_{model}")


def fig_ratio_depth(models=MODELS, tower="vision"):
    """Fig: the linear/nonlinear dimensionality ratio against relative depth, all models -
    the 'linear inflation peaks in the late layers' claim in one panel."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    drawn = 0
    for i, m in enumerate(models):
        d = _csv("6_3", "intrinsic_dim", m)
        if d is None:
            continue
        g = d[(d.tower == tower) & (d.level == "component") & (d.kind == "attn")]
        if g.empty:
            continue
        agg = g.groupby("layer")[["ratio", "twonn"]].mean()
        rel = agg.index / agg.index.max()
        c = OKABE_ITO[i % len(OKABE_ITO)]
        axes[0].plot(rel, agg.ratio, lw=2, marker="o", ms=3, color=c, label=m,
                     markeredgecolor="white", markeredgewidth=0.5)
        axes[1].plot(rel, agg.twonn, lw=2, marker="o", ms=3, color=c, label=m,
                     markeredgecolor="white", markeredgewidth=0.5)
        drawn += 1
    if not drawn:
        plt.close(fig); return
    axes[0].set_ylabel("linear / nonlinear ID  (L/N)")
    axes[1].set_ylabel("TwoNN ID  (N)")
    for ax in axes:
        ax.set_xlabel("relative depth")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle(f"Per-head dimensionality across architectures ({tower} tower)", y=1.03)
    fig.tight_layout()
    _save(fig, f"6_3_ratio_depth_{tower}")


# =========================================================================== 6.4
def fig_linear_approx(model, target="late4", tower="joint"):
    """Fig: how much retrieval survives a rank-k approximation of the top components,
    and how much of that the *nameable* (text/image-embedding) bases recover."""
    d = _csv("6_4", "linear_approx", model)
    if d is None:
        return
    base = d[d.basis == "exact"][R1].iloc[0]
    g = d[(d.target == target) & (d.tower == tower)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for basis in ("pc", "text", "image", "both"):
        gg = g[g.basis == basis].sort_values("k")
        if gg.empty:
            continue
        axes[0].plot(gg.k, gg[R1], marker="o", ms=4, lw=2, color=ORDER_COLOR[basis], label=basis,
                     markeredgecolor="white", markeredgewidth=0.6)
        axes[1].plot(gg.k, gg.cos_recon, marker="o", ms=4, lw=2, color=ORDER_COLOR[basis],
                     label=basis, markeredgecolor="white", markeredgewidth=0.6)
    axes[0].axhline(base, ls="--", lw=1, color="0.4")
    axes[0].text(g.k.min(), base, "full model ", va="bottom", ha="left", fontsize=7, color="0.35")
    for ax, yl in zip(axes, ("COCO R@1 (%)", "cosine to the exact embedding")):
        ax.set_xscale("log", base=2); ax.set_xlabel("rank k of the approximation"); ax.set_ylabel(yl)
    axes[0].legend(frameon=False, title="basis")
    fig.suptitle(f"{model}: rank-k linear approximation of the {target} components ({tower})", y=1.03)
    fig.tight_layout()
    _save(fig, f"6_4_linear_approx_{model}_{target}_{tower}")


def fig_approx_heat(model, tower="joint"):
    """Fig: % of baseline retrieval retained, basis x k, for each target set."""
    d = _csv("6_4", "linear_approx", model)
    if d is None:
        return
    base = d[d.basis == "exact"][R1].iloc[0]
    targets = [t for t in d.target.unique() if t != "none"]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.4 * len(targets), 2.6), squeeze=False)
    for ax, target in zip(axes[0], targets):
        g = d[(d.target == target) & (d.tower == tower)]
        bases = [b for b in ("pc", "text", "image", "both") if b in set(g.basis)]
        ks = sorted(g.k.unique())
        M = np.array([[g[(g.basis == b) & (g.k == k)][R1].mean() / base * 100 for k in ks]
                      for b in bases])
        im = ax.imshow(M, cmap="Purples", vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 62 else "0.15")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=7)
        ax.set_yticks(range(len(bases))); ax.set_yticklabels(bases, fontsize=7)
        ax.set_xlabel("rank k"); ax.set_title(target); ax.grid(False)
    fig.colorbar(im, ax=axes[0][-1], label="% of full-model R@1")
    fig.suptitle(f"{model}: retrieval retained by rank-k approximations ({tower})", y=1.05)
    fig.tight_layout()
    _save(fig, f"6_4_approx_heat_{model}_{tower}")


def fig_k90(models=MODELS, target="late4", tower="joint", thresholds=(0.5, 0.9)):
    """Fig: the rank k at which each basis first reaches a fraction of the full-model R@1 -
    the compact 'how many directions is a component worth' summary."""
    rows, k_tested = [], 0
    for m in models:
        d = _csv("6_4", "linear_approx", m)
        if d is None:
            continue
        base = d[d.basis == "exact"][R1].iloc[0]
        g = d[(d.target == target) & (d.tower == tower)]
        k_tested = max(k_tested, int(g.k.max()) if not g.empty else 0)
        for basis in ("pc", "text", "image", "both"):
            gg = g[g.basis == basis].sort_values("k")
            if gg.empty:
                continue
            for th in thresholds:
                hit = gg[gg[R1] >= th * base]
                rows.append(dict(model=m, basis=basis, th=th,
                                 k=(hit.k.iloc[0] if len(hit) else np.nan)))
    if not rows:
        return
    d = pd.DataFrame(rows)
    ms = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(thresholds), figsize=(3.6 * len(thresholds), 2.9), squeeze=False)
    for ax, th in zip(axes[0], thresholds):
        dd = d[d.th == th]
        bases = [b for b in ("pc", "text", "image", "both") if b in set(dd.basis)]
        x = np.arange(len(ms)); w = 0.8 / max(len(bases), 1)
        for i, b in enumerate(bases):
            v = [dd[(dd.model == m) & (dd.basis == b)].k.mean() for m in ms]
            v = [np.nan if pd.isna(x_) else x_ for x_ in v]
            ax.bar(x + (i - (len(bases) - 1) / 2) * w, v, w * 0.92, label=b, color=ORDER_COLOR[b])
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_yscale("log", base=2)
        ax.set_ylabel(f"k to reach {th:.0%} of full R@1")
        ax.set_title(f"{target}, {tower}")
    axes[0][0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
    fig.suptitle(f"Missing bar = that basis never reaches the threshold within k<={k_tested}",
                 y=1.04, fontsize=8)
    fig.tight_layout()
    _save(fig, f"6_4_k90_{target}_{tower}")


# ============================================================ 6.2 selector variants
VARIANT_COLOR = {"forward": OKABE_ITO[0], "forward+swap": OKABE_ITO[5],
                 "backward": OKABE_ITO[1], "backward+swap": OKABE_ITO[4]}
VARIANT_ORDER = ["forward", "forward+swap", "backward", "backward+swap"]


def _cat(section, name, models, tag=""):
    """Concatenate per-model CSVs. The task scripts name files `<name>_<model><tag>.csv`."""
    ds = []
    for m in models:
        p = os.path.join(RES_DIR, section, f"{name}_{m}{tag}.csv")
        if os.path.exists(p):
            ds.append(pd.read_csv(p))
    if not ds:
        print(f"[skip] no {name}{tag} yet")
        return None
    return pd.concat(ds, ignore_index=True)


def fig_waterbirds_variants(models=MODELS, level="comp"):
    """Fig: all four local-search selectors on Waterbirds, against the full model and a
    same-budget random pool. Worst-group is the number the spurious correlation moves; total
    is shown beside it so an improvement that merely trades one group for another is visible."""
    name = "waterbirds_eval" if level == "comp" else "waterbirds_pc_eval"
    d = _cat("6_2", name, models)
    if d is None:
        return
    d = d[~d.method.astype(str).str.contains("ORACLE")]
    ms = [m for m in models if m in set(d.model)]
    regimes = [r for r in ("class", "group") if r in set(d.regime)]
    sel = "selected"
    rnd = "random pool" if "random pool" in set(d.method) else "random units"
    fig, axes = plt.subplots(len(regimes), 2, figsize=(9.2, 3.1 * len(regimes)), squeeze=False)
    for ri, reg in enumerate(regimes):
        for ci, metric in enumerate(("worst_group", "total")):
            ax = axes[ri][ci]
            x = np.arange(len(ms))
            vs = [v for v in VARIANT_ORDER if v in set(d[d.method == sel].variant)]
            bars = [("full model", "0.4")] + [(v, VARIANT_COLOR[v]) for v in vs] + [(rnd, "0.75")]
            w = 0.84 / len(bars)
            for i, (lab, c) in enumerate(bars):
                if lab == "full model":
                    g = d[d.method == "full model"]
                elif lab == rnd:
                    g = d[(d.method == rnd) & (d.regime == reg)]
                else:
                    g = d[(d.method == sel) & (d.regime == reg) & (d.variant == lab)]
                v = [g[g.model == m][metric].mean() for m in ms]
                ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w * 0.92, label=lab, color=c,
                       edgecolor="white", lw=0.4)
            ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
            ax.set_ylabel(f"{metric.replace('_', '-')} acc (%)")
            ax.set_title(f"{reg}-balanced calibration - {metric.replace('_', ' ')}", fontsize=8.5)
    axes[0][0].legend(frameon=False, fontsize=6.3, ncol=2)
    unit = "components" if level == "comp" else "(component, PC) units"
    fig.suptitle(f"Waterbirds: four training-free selectors over {unit}", y=1.005)
    fig.tight_layout()
    _save(fig, f"6_2_waterbirds_variants_{level}")


def fig_waterbirds_trace(models=MODELS, regime="class", level="comp"):
    """Fig: the search trajectory itself - calibration loss (what the selector optimises) and
    held-out worst-group accuracy (what we care about) against step, per selector."""
    name = "waterbirds_trace" if level == "comp" else "waterbirds_pc_trace"
    d = _cat("6_2", name, models)
    if d is None:
        return
    d = d[d.regime == regime]
    ms = [m for m in models if m in set(d.model)]
    if not ms:
        return
    fig, axes = plt.subplots(2, len(ms), figsize=(2.5 * len(ms), 5.0), squeeze=False, sharex="col")
    for ci, m in enumerate(ms):
        g = d[d.model == m]
        for ri, y in enumerate(("loss", "worst_group")):
            ax = axes[ri][ci]
            for v in VARIANT_ORDER:
                gg = g[g.variant == v]
                if gg.empty or y not in gg:
                    continue
                mm = gg.groupby("step")[y].mean()
                ax.plot(mm.index, mm.values, lw=1.4, color=VARIANT_COLOR[v], label=v)
            ax.set_ylabel("calibration loss" if y == "loss" else "worst-group acc (%)")
            if ri == 0:
                ax.set_title(m, fontsize=8)
            else:
                ax.set_xlabel("search step")
    axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Waterbirds search trajectories ({regime}-balanced calibration): the loss is "
                 f"monotone by construction, the held-out metric is not", y=1.01, fontsize=8.5)
    fig.tight_layout()
    _save(fig, f"6_2_waterbirds_trace_{regime}_{level}")


def fig_waterbirds_pc_vs_comp(models=("ViT-B-32", "ViT-L-14")):
    """Fig: does splitting each component into PC directions buy anything? Same selectors, same
    task, units are (component, PC) pairs instead of whole components."""
    c = _cat("6_2", "waterbirds_eval", models)
    p = _cat("6_2", "waterbirds_pc_eval", models)
    if c is None or p is None:
        return
    ms = [m for m in models if m in set(p.model)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, reg in zip(axes, ("class", "group")):
        x = np.arange(len(ms)); groups = []
        full = [c[(c.model == m) & (c.method == "full model")].worst_group.mean() for m in ms]
        groups.append(("full model", full, "0.4"))
        for lvl, d, hatch in (("component", c, None), ("PC unit", p, "//")):
            for v in ("forward", "backward"):
                vals = [d[(d.model == m) & (d.method == "selected") & (d.regime == reg) &
                          (d.variant == v)].worst_group.mean() for m in ms]
                groups.append((f"{v} ({lvl})", vals, VARIANT_COLOR[v] if hatch is None
                               else VARIANT_COLOR[v], hatch))
        w = 0.84 / len(groups)
        for i, g in enumerate(groups):
            lab, vals, col = g[0], g[1], g[2]
            hatch = g[3] if len(g) > 3 else None
            ax.bar(x + (i - (len(groups) - 1) / 2) * w, vals, w * 0.92, label=lab, color=col,
                   hatch=hatch, edgecolor="white", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8)
        ax.set_ylabel("worst-group accuracy (%)")
        ax.set_title(f"{reg}-balanced calibration", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6.3, ncol=2)
    fig.suptitle("Waterbirds: whole components vs (component, PC) units - hatched = PC level",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "6_2_waterbirds_pc_vs_comp")


def fig_pc_moves(model="ViT-B-32", regime="class"):
    """Fig: where in the network the selected PC units live - tower x depth x PC index. Forward
    ADDS units (these carry the task), backward DROPS them (these carry the spurious cue)."""
    p = os.path.join(RES_DIR, "6_2", f"waterbirds_pc_moves_{model}.csv")
    if not os.path.exists(p):
        print(f"[skip] missing {p}")
        return
    d = pd.read_csv(p)
    d = d[(d.regime == regime) & (d.rep == d.rep.min())]
    if d.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2))
    for ri, (variant, action) in enumerate((("forward", "add"), ("backward", "drop"))):
        g = d[(d.variant == variant) & (d.action == action)]
        if g.empty:
            continue
        ax = axes[ri][0]
        for i, tw in enumerate(("vision", "text")):
            gg = g[g.tower == tw]
            if gg.empty:
                continue
            h = gg.groupby("layer").size()
            ax.bar(h.index + (i - 0.5) * 0.4, h.values, 0.4, label=tw, color=OKABE_ITO[i])
        ax.set_xlabel("layer"); ax.set_ylabel(f"# units {action}ed")
        ax.set_title(f"{variant}: depth of {action}ed units", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[ri][1]
        h = g.groupby("pc").size()
        ax.bar(h.index, h.values, 0.85, color=VARIANT_COLOR[variant])
        ax.set_xlabel("PC index within its component"); ax.set_ylabel(f"# units {action}ed")
        ax.set_title(f"{variant}: which PC of the component", fontsize=8.5)
    fig.suptitle(f"{model} Waterbirds ({regime}-balanced): anatomy of the selected PC units",
                 y=1.01)
    fig.tight_layout()
    _save(fig, f"6_2_pc_moves_{model}_{regime}")


# ============================================================ 6.2 bias audit
def fig_bias_audit(models=MODELS):
    """Fig: the harmful error rate, per FairFace race group, before and after selection.

    Two things must hold for a claimed fix to be real: the harmful rate falls, AND the primate
    images stay correctly classified (otherwise the model has merely collapsed the label space).
    Both are plotted."""
    d = _cat("6_2", "bias_audit_eval", models)
    if d is None:
        return
    groups = [c[len("crime_"):] for c in d.columns if c.startswith("crime_")]
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(2, len(ms), figsize=(2.7 * len(ms), 5.4), squeeze=False)
    for ci, m in enumerate(ms):
        g = d[d.model == m]
        for ri, pref in enumerate(("crime_", "nonhuman_")):
            ax = axes[ri][ci]
            x = np.arange(len(groups))
            series = [("full model", "0.4"), ("forward", VARIANT_COLOR["forward"]),
                      ("backward", VARIANT_COLOR["backward"]), ("random pool", "0.75")]
            series = [s for s in series if s[0] in set(g.variant)]
            w = 0.84 / max(len(series), 1)
            for i, (v, c) in enumerate(series):
                vals = [g[g.variant == v][pref + gr].mean() for gr in groups]
                ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, label=v, color=c)
            ax.set_xticks(x)
            ax.set_xticklabels([gr.replace(" ", "\n") for gr in groups], fontsize=5.5, rotation=90)
            ax.set_ylabel(f"faces -> {pref[:-1]} (%)")
            if ri == 0:
                ax.set_title(m, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=6)
    fig.suptitle("Dehumanisation / crime-category audit: rate at which FairFace faces are given a "
                 "non-human or crime label", y=1.005, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "6_2_bias_audit")

    # the integrity check, as its own panel
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(ms))
    series = [v for v in ("full model", "forward", "forward+swap", "backward", "backward+swap",
                          "random pool") if v in set(d.variant)]
    w = 0.84 / len(series)
    for i, v in enumerate(series):
        vals = [d[(d.model == m) & (d.variant == v)].primate_nonhuman.mean() for m in ms]
        col = VARIANT_COLOR.get(v, "0.4" if v == "full model" else "0.75")
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, label=v, color=col)
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
    ax.set_ylabel("primates -> non-human (%)")
    ax.legend(frameon=False, fontsize=6, ncol=3)
    ax.set_title("Sanity check: the primate images must STAY non-human, else the fix is a collapse")
    fig.tight_layout()
    _save(fig, "6_2_bias_audit_sanity")


# ============================================================ 6.2 extra tasks
def fig_task_pool(tag, title, models=MODELS):
    """Fig: greedy-pool result for one of the auxiliary tasks (MNIST, typographic, ...)."""
    d = _cat("6_2", "greedy_pool_eval", models, tag)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    tasks = list(dict.fromkeys(d.task))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 3.0), squeeze=False)
    meths = [x for x in ("full model", "greedy pool", "top-k loss score", "random pool")
             if x in set(d.method)]
    for ax, t in zip(axes[0], tasks):
        g = d[d.task == t]
        x = np.arange(len(ms)); w = 0.84 / max(len(meths), 1)
        for i, me in enumerate(meths):
            v = [g[(g.model == m) & (g.method == me)].acc.mean() for m in ms]
            ax.bar(x + (i - (len(meths) - 1) / 2) * w, v, w * 0.9, label=me,
                   color=["0.4", OKABE_ITO[0], OKABE_ITO[2], "0.75"][i % 4])
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_ylabel("top-1 (%)"); ax.set_title(t, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    _save(fig, f"6_2_task_pool{tag}")


def fig_mnist_prompt_control(models=MODELS):
    """Fig: the MNIST gain, with and without a prompt template. If the raw-classname run gains a
    lot and the templated run gains little, the 'fix' was compensating for a bad prompt."""
    a = _cat("6_2", "greedy_pool_eval", models, "_mnist")
    b = _cat("6_2", "greedy_pool_eval", models, "_mnistp")
    if a is None and b is None:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ms = [m for m in models if (a is not None and m in set(a.model))
          or (b is not None and m in set(b.model))]
    x = np.arange(len(ms)); series = []
    for lab, d, cols in (("bare classnames", a, ("0.55", OKABE_ITO[0])),
                         ("with prompt template", b, ("0.3", OKABE_ITO[1]))):
        if d is None:
            continue
        series.append((f"{lab}: full", [d[(d.model == m) & (d.method == "full model")].acc.mean()
                                        for m in ms], cols[0]))
        series.append((f"{lab}: greedy pool",
                       [d[(d.model == m) & (d.method == "greedy pool")].acc.mean() for m in ms],
                       cols[1]))
    w = 0.84 / max(len(series), 1)
    for i, (lab, v, c) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, v, w * 0.9, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=20, fontsize=7)
    ax.set_ylabel("MNIST top-1 (%)"); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_title("MNIST: is the selection gain real, or is it fixing the prompt?")
    fig.tight_layout()
    _save(fig, "6_2_mnist_prompt_control")


def fig_hardneg(models=MODELS, tasks=("negation", "count", "order")):
    """Fig: matched caption pairs - chance is exactly 50%, so the dashed line is the whole story.

    Any bar near it means the model is blind to the single edit that separates the two captions."""
    ds = []
    for t in tasks:
        d = _cat("6_2", f"hardneg_eval_{t}", models)
        if d is not None:
            ds.append(d)
    if not ds:
        return
    d = pd.concat(ds, ignore_index=True)
    d["variant"] = d.variant.where(d.method.isna() | (d.method != "random pool"), "random pool")
    tasks = [t for t in tasks if t in set(d.task)]
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.1), squeeze=False,
                             sharey=True)
    series = ["full model"] + VARIANT_ORDER + ["random pool"]
    for ax, t in zip(axes[0], tasks):
        g = d[d.task == t]
        x = np.arange(len(ms))
        use = [s for s in series if s in set(g.variant)]
        w = 0.84 / max(len(use), 1)
        for i, v in enumerate(use):
            gg = g[g.variant == v]
            if v != "full model" and "method" in gg:
                gg = gg[gg.method != "random pool"] if v != "random pool" else gg
            vals = [gg[gg.model == m].acc.mean() for m in ms]
            col = VARIANT_COLOR.get(v, "0.4" if v == "full model" else "0.75")
            ax.bar(x + (i - (len(use) - 1) / 2) * w, vals, w * 0.9, label=v, color=col)
        ax.axhline(50, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_title(t, fontsize=9)
    axes[0][0].set_ylabel("pair accuracy (%)")
    axes[0][0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Matched caption pairs: dashed line is chance (50%)", y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_hardneg")


def fig_hardneg_negation_split(models=MODELS):
    """Fig: negation, split by whether the object IS or IS NOT in the image.

    The failure is one-sided - CLIP is near-perfect when the object is present and near-zero when
    it is absent, because the object word fires either way. A fix has to move the absent bar
    without destroying the present one, so both are plotted."""
    d = _cat("6_2", "hardneg_eval_negation", models)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), sharey=True)
    series = ["full model"] + VARIANT_ORDER
    for ax, col in zip(axes, ("acc_present", "acc_absent")):
        x = np.arange(len(ms))
        use = [s for s in series if s in set(d.variant)]
        w = 0.84 / max(len(use), 1)
        for i, v in enumerate(use):
            g = d[(d.variant == v) & (d.method != "random pool")] if "method" in d else d[d.variant == v]
            vals = [g[g.model == m][col].mean() for m in ms]
            ax.bar(x + (i - (len(use) - 1) / 2) * w, vals, w * 0.9, label=v,
                   color=VARIANT_COLOR.get(v, "0.4"))
        ax.axhline(50, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_title("object present in the image" if col == "acc_present"
                     else "object ABSENT from the image", fontsize=9)
    axes[0].set_ylabel("pair accuracy (%)")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle('Negation: "a photo of a dog" vs "a photo with no dog"', y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_hardneg_negation_split")


def _table_fig(cell_text, row_labels, col_labels, shade, title, note, name, colw=1.35):
    """Render a numeric table as a figure, shaded by `shade` (same shape as cell_text).

    Shading is per COLUMN: models differ by many points, so a global ramp would only re-say which
    model is strongest instead of which method wins within a model."""
    nr, nc = len(row_labels), len(col_labels)
    hdr = 1 + max(str(c).count("\n") for c in col_labels)
    fig, ax = plt.subplots(figsize=(1.9 + colw * nc, 0.34 * (nr + hdr) + 0.75))
    ax.axis("off"); ax.grid(False)
    # bbox fills the axes so the table does not float in a sea of whitespace
    tb = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                  cellLoc="center", rowLoc="left", bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(8)
    cmap = plt.get_cmap("Greens")
    sh = np.asarray(shade, dtype=float)
    for j in range(nc):
        col = sh[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        for i in range(nr):
            if np.isnan(col[i]):
                continue
            f = 0.0 if hi == lo else (col[i] - lo) / (hi - lo)
            c = tb[i + 1, j]
            c.set_facecolor(cmap(0.08 + 0.55 * f))
            c.set_text_props(color="white" if f > 0.82 else "black")
    for j in range(nc):
        tb[0, j].set_text_props(weight="bold")
    for i in range(nr):
        tb[i + 1, -1].set_text_props(weight="bold" if "ours" in row_labels[i] else "normal")
    ax.set_title(title, fontsize=9.5, pad=14)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7, color="0.3")
    _save(fig, name)


WB_ROWS = [("full model", "-", "Baseline (full model)"),
           ("random pool", "-", "Random pool (same size)"),
           ("selected", "forward", "Forward (ours)"),
           ("selected", "forward+swap", "Forward+swap (ours)"),
           ("selected", "backward", "Backward (ours)"),
           ("selected", "backward+swap", "Backward+swap (ours)")]


def fig_waterbirds_table(models=MODELS, regime="class"):
    """Fig: the Waterbirds results table - total and worst-group per model, method by method."""
    d = _cat("6_2", "waterbirds_table", models)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    cells, shade, rows = [], [], []
    for meth, var, lab in WB_ROWS:
        r_txt, r_sh = [], []
        for m in ms:
            g = d[(d.model == m) & (d.method == meth)]
            if meth != "full model":
                g = g[(g.regime == regime) & (g.variant == var)] if var != "-" \
                    else g[g.regime == regime]
            if g.empty:
                r_txt.append("-"); r_sh.append(np.nan); continue
            r_txt.append(f"{g.total.mean():.2f}, {g.worst_group.mean():.2f}")
            r_sh.append(g.worst_group.mean())
        if all(t == "-" for t in r_txt):
            continue
        cells.append(r_txt); shade.append(r_sh); rows.append(lab)
    if not rows:
        return
    _table_fig(cells, rows, ms, shade,
               f"Waterbirds - total, worst-group  ({regime}-balanced calibration)",
               "Each cell is total accuracy, worst-group accuracy. Shading is per column, by "
               "worst-group.", f"6_2_waterbirds_table_{regime}")


def fig_waterbirds_table_groups(models=MODELS, regime="class", model=None):
    """Fig: every cell of the 2x2 - the two minority cells are where the shortcut lives."""
    d = _cat("6_2", "waterbirds_table", models)
    if d is None:
        return
    cols = ["landbird_land", "landbird_water", "waterbird_land", "waterbird_water",
            "class_0", "class_1", "total"]
    nice = ["landbird\non land", "landbird\non water*", "waterbird\non land*", "waterbird\non water",
            "landbird\n(all)", "waterbird\n(all)", "total"]
    for m in ([model] if model else [x for x in models if x in set(d.model)]):
        cells, shade, rows = [], [], []
        for meth, var, lab in WB_ROWS:
            g = d[(d.model == m) & (d.method == meth)]
            if meth != "full model":
                g = g[(g.regime == regime) & (g.variant == var)] if var != "-" \
                    else g[g.regime == regime]
            if g.empty:
                continue
            cells.append([f"{g[c].mean():.1f}" for c in cols])
            shade.append([g[c].mean() for c in cols])
            rows.append(lab)
        if not rows:
            continue
        _table_fig(cells, rows, nice, shade,
                   f"{m} - Waterbirds, every group ({regime}-balanced calibration)",
                   "* = minority cell: the bird on the background it is NOT correlated with. "
                   "Shading is per column.", f"6_2_waterbirds_groups_{m}_{regime}", colw=0.95)


def fig_typographic(models=MODELS):
    """Fig: the typographic attack - the two bars must be read together.

    `acc_true` and `attack_rate` are not complements (998 other classes exist). A selector that
    lowers both is destroying the model; one that lowers only the attack rate is suppressing the
    text-reading route, which is the claim being tested."""
    d = _cat("6_2", "typographic_eval", models)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    series = ["full model"] + VARIANT_ORDER + ["random pool"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for ax, col in zip(axes, ("acc_true", "attack_rate")):
        x = np.arange(len(ms))
        use = [s for s in series if s in set(d.variant) or s == "random pool"]
        w = 0.84 / max(len(use), 1)
        for i, v in enumerate(use):
            if v == "random pool":
                g = d[d.method == "random pool"]
            elif v == "full model":
                g = d[d.method == "full model"]
            else:
                g = d[(d.method == "selected") & (d.variant == v)]
            vals = [g[g.model == m][col].mean() for m in ms]
            col_c = VARIANT_COLOR.get(v, "0.4" if v == "full model" else "0.75")
            ax.bar(x + (i - (len(use) - 1) / 2) * w, vals, w * 0.9, label=v, color=col_c)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_ylabel("% of images")
        ax.set_title("assigned the TRUE class (higher better)" if col == "acc_true"
                     else "assigned the PRINTED word (lower better)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Typographic attack: a wrong class name printed on a real ImageNet photo", y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_typographic")


def fig_insub(models=MODELS, level="component"):
    """Fig: specialise on 10 ImageNet classes, then pay for it on the other 990."""
    d = _cat("6_2", "insub_eval", models)
    if d is None:
        return
    d = d[d.level.isin([level, "-"])]
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    series = ["full model"] + VARIANT_ORDER
    for ax, col in zip(axes, ("acc_subset", "acc_full")):
        x = np.arange(len(ms))
        use = [s for s in series if s in set(d.variant)] + ["random pool"]
        w = 0.84 / max(len(use), 1)
        for i, v in enumerate(use):
            g = (d[d.method == "random pool"] if v == "random pool"
                 else d[(d.method == "full model") if v == "full model"
                        else (d.method == "selected") & (d.variant == v)])
            vals = [g[g.model == m][col].mean() for m in ms]
            c = VARIANT_COLOR.get(v, "0.4" if v == "full model" else "0.75")
            ax.bar(x + (i - (len(use) - 1) / 2) * w, vals, w * 0.9, label=v, color=c)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_ylabel("top-1 (%)")
        ax.set_title("the 10 chosen classes" if col == "acc_subset"
                     else "the full 1,000-way problem", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle(f"ImageNet specialisation, {level} level: what the subset gain costs elsewhere",
                 y=1.03)
    fig.tight_layout()
    _save(fig, f"6_2_insub_{level}")


def fig_insub_trajectory(models=MODELS, level="component"):
    """Fig: the search trajectory - calibration loss against both accuracies.

    The loss falls monotonically by construction, so it proves nothing on its own. The two
    accuracies on the same x-axis are the actual result: where the 1,000-way curve turns down
    while the 10-way curve is still rising is the point specialisation starts costing generality."""
    d = _cat("6_2", "insub_trace", models)
    if d is None:
        return
    d = d[d.level == level]
    ms = [m for m in models if m in set(d.model)]
    if not ms:
        return
    fig, axes = plt.subplots(2, len(ms), figsize=(2.6 * len(ms), 5.2), squeeze=False, sharex="col")
    for ci, m in enumerate(ms):
        g = d[d.model == m]
        ax = axes[0][ci]
        for v in VARIANT_ORDER:
            gg = g[g.variant == v]
            if gg.empty:
                continue
            mm = gg.groupby("step")["loss"].mean()
            ax.plot(mm.index, mm.values, lw=1.4, color=VARIANT_COLOR[v], label=v)
        ax.set_ylabel("calibration loss (10-way CE)"); ax.set_title(m, fontsize=8)
        ax = axes[1][ci]
        for v in VARIANT_ORDER:
            gg = g[g.variant == v]
            if gg.empty:
                continue
            for col, ls in (("acc_subset", "-"), ("acc_full", ":")):
                mm = gg.groupby("step")[col].mean()
                ax.plot(mm.index, mm.values, lw=1.3, ls=ls, color=VARIANT_COLOR[v])
        ax.set_xlabel("search step"); ax.set_ylabel("top-1 (%)")
    axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"ImageNet specialisation ({level} level). Bottom: solid = the 10 classes, "
                 f"dotted = all 1,000", y=1.01, fontsize=8.5)
    fig.tight_layout()
    _save(fig, f"6_2_insub_trajectory_{level}")


def fig_counting(models=MODELS):
    """Fig: counting 1..10. The dashed line is chance (10%) with the object held fixed - if the
    full-model bar sits on it, this set cannot support a claim about counting."""
    d = _cat("6_2", "counting_eval", models)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2))
    series = ["full model"] + VARIANT_ORDER + ["random pool"]
    for ax, col, ttl in zip(axes, ("count_acc", "full_acc", "mae"),
                            ("count, object fixed (chance 10%)", "joint 200-way",
                             "mean |predicted - true| count")):
        x = np.arange(len(ms))
        use = [s for s in series if s in set(d.variant) or s in ("random pool", "full model")]
        w = 0.84 / max(len(use), 1)
        for i, v in enumerate(use):
            g = (d[d.method == "random pool"] if v == "random pool"
                 else d[d.method == "full model"] if v == "full model"
                 else d[(d.method == "selected") & (d.variant == v)])
            vals = [g[g.model == m][col].mean() for m in ms]
            c = VARIANT_COLOR.get(v, "0.4" if v == "full model" else "0.75")
            ax.bar(x + (i - (len(use) - 1) / 2) * w, vals, w * 0.9, label=v, color=c)
        if col == "count_acc":
            ax.axhline(10, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, fontsize=7)
        ax.set_ylabel("MAE (counts)" if col == "mae" else "top-1 (%)")
        ax.set_title(ttl, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Counting 1-10 on rendered scenes with exact ground truth", y=1.03)
    fig.tight_layout()
    _save(fig, "6_2_counting")


def fig_counting_per_n(models=MODELS):
    """Fig: accuracy per count value - small n is where any count signal lives."""
    d = _cat("6_2", "counting_eval", models)
    if d is None:
        return
    cols = [f"count_{k}" for k in range(1, 11)]
    if not set(cols) <= set(d.columns):
        return
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.4 * len(ms), 2.8), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        g = d[d.model == m]
        for v, c in ([("full model", "0.4")] +
                     [(x, VARIANT_COLOR[x]) for x in ("forward", "backward")]):
            gg = (g[g.method == "full model"] if v == "full model"
                  else g[(g.method == "selected") & (g.variant == v)])
            if gg.empty:
                continue
            ax.plot(range(1, 11), [gg[c_].mean() for c_ in cols], marker="o", ms=3, lw=1.4,
                    color=c, label=v)
        ax.axhline(10, color="k", ls="--", lw=1)
        ax.set_xlabel("true count"); ax.set_title(m, fontsize=8)
    axes[0][0].set_ylabel("accuracy (%)"); axes[0][0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Counting accuracy by count value (dashed = chance)", y=1.05)
    fig.tight_layout()
    _save(fig, "6_2_counting_per_n")


def fig_bias_pc_anatomy(models=MODELS):
    """Fig: where the PC units the bias search drops actually live."""
    d = _cat("6_2", "bias_pc_moves", models)
    if d is None:
        return
    ms = [m for m in models if m in set(d.model)]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.8 * len(ms), 2.9), squeeze=False)
    for ax, m in zip(axes[0], ms):
        g = d[(d.model == m) & (d.variant == "backward") & (d.action == "drop") &
              (d.rep == d.rep.min())]
        for i, tw in enumerate(("vision", "text")):
            gg = g[g.tower == tw]
            if gg.empty:
                continue
            h = gg.groupby("layer").size()
            ax.bar(h.index + (i - 0.5) * 0.4, h.values, 0.4, label=tw, color=OKABE_ITO[i])
        ax.set_xlabel("layer"); ax.set_ylabel("# PC units dropped"); ax.set_title(m, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Bias audit, backward search: depth of the dropped PC directions", y=1.04)
    fig.tight_layout()
    _save(fig, "6_2_bias_pc_anatomy")


# =========================================================================== driver
def run(sections, models):
    if "6.2" in sections:
        fig_baselines(models)
        fig_concentration(models)
        fig_ranking_quality(models)
        for m in models:
            for src in ("self", "ref"):
                fig_layer_ablation(m, src)
            for mode in ("ablate", "keep"):
                fig_component_ranking(m, "self", mode)
            fig_component_map(m)
            fig_loud_vs_informative(m)
            fig_pair_ranking(m)
            fig_pair_map(m)
            fig_loss_ranking(m)
            fig_greedy_order(m)
            fig_pool_trajectory(m)
        fig_pool_bars(models)
        fig_pool_retrieval(models)
        fig_waterbirds(models)
        for lvl in ("comp", "pc"):
            fig_waterbirds_variants(models, lvl)
            for reg in ("class", "group"):
                fig_waterbirds_trace(models, reg, lvl)
        fig_waterbirds_pc_vs_comp([m for m in models if m in ("ViT-B-32", "ViT-L-14")])
        for reg in ("class", "group"):
            fig_waterbirds_table(models, reg)
            fig_waterbirds_table_groups(models, reg)
        for m in models:
            for reg in ("class", "group"):
                fig_pc_moves(m, reg)
        fig_bias_audit(models)
        fig_bias_pc_anatomy(models)
        fig_typographic(models)
        fig_counting(models)
        fig_counting_per_n(models)
        for lvl in ("component", "pc"):
            fig_insub(models, lvl)
            fig_insub_trajectory(models, lvl)
        fig_hardneg(models)
        fig_hardneg_negation_split(models)
        fig_mnist_prompt_control(models)
        for tag, title in (("_mnist", "MNIST (bare classnames)"),
                           ("_mnistp", "MNIST (prompt template)"),
                           ("_inbig", "ImageNet, uncapped pool"),
                           ("_typo", "Typographic attack: label pasted onto the image"),
                           ("_count", "Counting"), ("_neg", "Negation")):
            fig_task_pool(tag, f"Greedy pool - {title}", models)
    if "6.3" in sections:
        for col in ("twonn", "pca95", "ratio"):
            fig_id_layer_own(models, col)
        for tower in ("vision", "text"):
            fig_id_table(models, tower)
            fig_ratio_depth(models, tower)
        for m in models:
            fig_id_components(m)
            fig_modality_geometry(m)
    if "6.4" in sections:
        fig_k90(models)
        for m in models:
            for target in ("late4", "topB32"):
                fig_linear_approx(m, target)
            fig_approx_heat(m)


def get_args_parser():
    p = argparse.ArgumentParser("paper figures 6.2-6.4", add_help=False)
    p.add_argument("--sections", nargs="+", default=["6.2", "6.3", "6.4"])
    p.add_argument("--models", nargs="+", default=MODELS)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    run(a.sections, a.models)
