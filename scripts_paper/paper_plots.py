"""
Paper-quality figures (paper section: all). One function per figure; each reads
ONLY results/paper/ (stats are decoupled from plotting) and writes a PDF.

Figures (data source in parentheses):
  mean_ablation        (ablation_orderings_*.json)
  interaction_matrix   (shared_pairs_*.json -> C_hat)
  h1_scatter           (shared_pairs_*.json -> routing_H1)   [prints stats; scatter needs raw pairs]
  paired_ablation      (shared_pairs_*.json -> paired_ablation)
  ranked_pairs         (shared_pairs_*.json -> score_B_top)
  checkpoint_traj      (checkpoints_*.json)
  random_control       (query_random_control.csv)
  alive_fraction       (RN50/alive_fraction.json)

Usage:
  python -m scripts_paper.paper_plots --results results/paper --model ViT-B-32 --figures all
"""
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colorblind-safe (Okabe-Ito) + paper rcParams (>= 8pt at column width, PDF export)
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    "axes.grid": True, "grid.alpha": 0.3,
})


def _load_json(path):
    if not os.path.exists(path):
        print(f"[skip] missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, out):
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def mean_ablation(results, model, seed, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=False)
    panels = [("vision", "attn"), ("vision", "mlp"), ("text", "attn"), ("text", "mlp")]
    for ax, (tower, kind) in zip(axes.flat, panels):
        d = _load_json(os.path.join(results, model, f"ablation_orderings_{tower}_{kind}_seed_{seed}.json"))
        if d is None:
            ax.set_visible(False); continue
        cur = d["curves"]
        x = np.arange(len(cur.get("forward", cur.get("random_mean", []))))
        for key in ("forward", "backward"):
            if key in cur:
                ax.plot(x, cur[key], marker="o", ms=3, label=key)
        if "random_mean" in cur:
            m, s = np.array(cur["random_mean"]), np.array(cur["random_std"])
            ax.plot(x, m, marker="o", ms=3, label="random")
            ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_title(f"{tower} {kind}"); ax.set_xlabel("# layers mean-ablated"); ax.set_ylabel("acc (%)")
        ax.legend()
    fig.suptitle(f"Mean-ablation ({model})")
    _save(fig, os.path.join(outdir, f"mean_ablation_{model}.pdf"))


def interaction_matrix(results, model, seed, outdir):
    d = _load_json(os.path.join(results, model, f"shared_pairs_seed_{seed}.json"))
    if d is None:
        return
    C_hat = np.array(d["C_hat"])
    cl_v, cl_t = d["comp_labels_v"], d["comp_labels_t"]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    vmax = np.abs(C_hat).max()
    im = ax.imshow(C_hat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(0, len(cl_t), 4)); ax.set_xticklabels(cl_t[::4], rotation=90)
    ax.set_yticks(range(0, len(cl_v), 4)); ax.set_yticklabels(cl_v[::4])
    ax.set_xlabel("text components"); ax.set_ylabel("vision components")
    # annotate top-5 |C_hat| cells
    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(C_hat), axis=None), C_hat.shape))[0][:5]
    for (i, j) in flat:
        ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\hat{C}_{ij}$")
    ax.set_title(f"Interaction matrix ({model})")
    _save(fig, os.path.join(outdir, f"interaction_matrix_{model}.pdf"))


def paired_ablation(results, model, seed, outdir):
    d = _load_json(os.path.join(results, model, f"shared_pairs_seed_{seed}.json"))
    if d is None:
        return
    ab = d["paired_ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax, tower in zip(axes, ["vision", "text", "joint"]):
        x = np.arange(len(ab["high2low"][tower]))
        for order in ["high2low", "low2high", "random"]:
            ax.plot(x, ab[order][tower], marker="o", ms=3, label=order)
        ax.set_title(tower); ax.set_xlabel("# pairs ablated"); ax.set_ylabel("acc (%)")
    axes[-1].legend()
    fig.suptitle(f"Paired-ablation ({model})")
    _save(fig, os.path.join(outdir, f"paired_ablation_{model}.pdf"))


def ranked_pairs(results, model, seed, outdir):
    d = _load_json(os.path.join(results, model, f"shared_pairs_seed_{seed}.json"))
    if d is None:
        return
    top = d["score_B_top"]
    labels = [f"{p['vision']}\n{p['text']}" for p in top]
    vals = [p["score"] for p in top]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(range(len(vals)), vals, color=OKABE_ITO[0])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Metric B"); ax.set_title(f"Top shared head-pairs ({model})")
    r = d.get("routing_H1", {})
    if r:
        ax.text(0.98, 0.95, f"H1 spearman={r['spearman']:.2f} (z={r['z']:.1f})",
                transform=ax.transAxes, ha="right", va="top")
    _save(fig, os.path.join(outdir, f"ranked_pairs_{model}.pdf"))


def checkpoint_traj(results, outdir, fname="checkpoints_vitb32.json"):
    d = _load_json(os.path.join(results, fname))
    if d is None:
        return
    metricB = np.array(d["metricB"]); acc = np.array(d["accuracy"]); tags = d["tags"]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for p in range(metricB.shape[1]):
        ax.plot(x, metricB[:, p], marker="o", ms=3, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30)
    ax.set_ylabel("Metric B"); ax.set_xlabel("checkpoint")
    axt = ax.twinx(); axt.plot(x, acc, "k--", lw=2, label="zero-shot acc"); axt.set_ylabel("acc (%)")
    ax.set_title("Emergence of top metric-B head-pairs")
    _save(fig, os.path.join(outdir, "checkpoint_trajectories.pdf"))


def random_control(results, model, outdir, fname="query_random_control.csv"):
    path = os.path.join(results, model, fname)
    if not os.path.exists(path):
        print(f"[skip] missing {path}"); return
    agg = defaultdict(list)  # method -> recon_cosine values
    with open(path) as f:
        for row in csv.DictReader(f):
            agg[row["method"]].append(float(row["recon_cosine"]))
    methods = [m for m in ["query", "random_late", "random_uniform"] if m in agg]
    means = [np.mean(agg[m]) for m in methods]
    stds = [np.std(agg[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(methods, means, yerr=stds, capsize=4, color=OKABE_ITO[:len(methods)])
    ax.set_ylabel("reconstruction cosine"); ax.set_title(f"QuerySystem vs random ({model})")
    _save(fig, os.path.join(outdir, f"random_control_{model}.pdf"))


def alive_fraction(results, outdir, model="RN50"):
    d = _load_json(os.path.join(results, model, "alive_fraction.json"))
    if d is None:
        return
    frac = np.array(d["alive_frac"])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(frac, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("head h"); ax.set_ylabel("block l (0=stem)")
    ax.set_title(f"alive fraction $\\pi_{{l,h}}$ ({model}, min={d['min_alive_fraction']:.3f})")
    fig.colorbar(im, ax=ax, label=r"$\pi$ = active/N")
    _save(fig, os.path.join(outdir, f"alive_fraction_{model}.pdf"))


ALL = ["mean_ablation", "interaction_matrix", "paired_ablation", "ranked_pairs",
       "checkpoint_traj", "random_control", "alive_fraction"]


def get_args_parser():
    p = argparse.ArgumentParser("Paper plots", add_help=False)
    p.add_argument("--results", default="results/paper", type=str)
    p.add_argument("--outdir", default="results/paper/figures", type=str)
    p.add_argument("--model", default="ViT-B-32", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--figures", nargs="+", default=["all"])
    return p


def main(args):
    figs = ALL if args.figures == ["all"] else args.figures
    for name in figs:
        if name == "mean_ablation":
            mean_ablation(args.results, args.model, args.seed, args.outdir)
        elif name == "interaction_matrix":
            interaction_matrix(args.results, args.model, args.seed, args.outdir)
        elif name == "paired_ablation":
            paired_ablation(args.results, args.model, args.seed, args.outdir)
        elif name == "ranked_pairs":
            ranked_pairs(args.results, args.model, args.seed, args.outdir)
        elif name == "checkpoint_traj":
            checkpoint_traj(args.results, args.outdir)
        elif name == "random_control":
            random_control(args.results, args.model, args.outdir)
        elif name == "alive_fraction":
            alive_fraction(args.results, args.outdir)
        else:
            print(f"[skip] unknown figure {name}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
