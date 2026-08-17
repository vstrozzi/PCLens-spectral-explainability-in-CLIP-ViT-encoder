"""Generate paper_experiments_6_2_to_6_4.ipynb (results notebook for sections 6.2-6.4).

The notebook is a *reader* of output_dir/results_paper/**.csv: heavy compute runs through
scripts_paper/run_6_2_6_4.sbatch, the notebook shows the tables and the figures.

  python -m scripts_paper.gen_notebook_6_2_6_4
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "paper_experiments_6_2_to_6_4.ipynb")


def _lines(src):
    """nbformat wants one entry per line, each newline-terminated except the last."""
    ls = src.split("\n")
    return [l + "\n" for l in ls[:-1]] + ls[-1:]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src.strip())}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _lines(src.strip("\n"))}


CELLS = [
    md("""
# PCLens - paper experiments 6.2 - 6.4

Retrieval-grounded component analysis of both CLIP towers on **COCO Karpathy-test**,
using the exact residual-stream decomposition (components sum to the L2-normalised embedding).

| Section | Question | Output |
|---|---|---|
| 6.2 | Where are the top components? | `6_2/*.csv` - layer & component mean-ablation curves, Metric A/B scores |
| 6.3 | What is their intrinsic dimensionality? | `6_3/intrinsic_dim_*.csv` - PCA 80/95/99, TwoNN, ratio, EVR1, geometry |
| 6.4 | How well are they linearly approximated? | `6_4/linear_approx_*.csv` - rank-k PC vs text/image embedding spans |

**Setup.** 5,000 images x 1 caption each (one caption per image, drawn with seed 69, so retrieval
is strictly 1-to-1 in both directions). Retrieval = cosine ranking of the renormalised embeddings,
i.e. exactly CLIP's inference rule; ties count against the model, so a fully mean-ablated
(constant) embedding reads as chance rather than 100%.

**Compute path.** All of it runs on the GPU under SLURM:

```bash
sbatch scripts_paper/extract_coco.sbatch     # once: COCO activations, 6 models, both towers
sbatch scripts_paper/run_6_2_6_4.sbatch      # 6.2 + 6.3 + 6.4 + every figure
```
"""),
    code("""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath("")) if "scripts_paper" in os.getcwd() else os.getcwd())

from scripts_paper.pclens_core import RES_DIR, MODELS
from scripts_paper import exp_plots
from IPython.display import Image, display, Markdown

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
FIG = os.path.join(RES_DIR, "figures")

def load(section, name, models=MODELS):
    "Concatenate one CSV family across models."
    fs = [os.path.join(RES_DIR, section, f"{name}_{m}.csv") for m in models]
    ds = [pd.read_csv(f) for f in fs if os.path.exists(f)]
    if not ds: return pd.DataFrame()
    d = pd.concat(ds, ignore_index=True)
    if "R@1_i2t" in d: d["R@1"] = (d["R@1_i2t"] + d["R@1_t2i"]) / 2
    return d

def show(name):
    p = os.path.join(FIG, f"{name}.png")
    display(Image(p)) if os.path.exists(p) else print(f"[missing] {p}")

AVAIL = sorted({os.path.basename(f).split("_")[-1][:-4]
                for f in glob.glob(os.path.join(RES_DIR, "6_2", "baseline_*.csv"))})
print("models with results:", AVAIL)
"""),

    md("""
## 0. The retrieval task

One caption per image makes image->text and text->image symmetric, each with 5,000 candidates
and exactly one correct answer. These baselines are the reference every ablation is measured
against.
"""),
    code("""
base = load("6_2", "baseline")
display(base[["model", "R@1_i2t", "R@5_i2t", "R@10_i2t", "R@1_t2i", "R@5_t2i", "R@10_t2i",
              "medr_i2t", "rsum"]].round(2))
show("6_2_baselines")
"""),

    md("""
## 6.2 Where are the top components?

Two granularities, both mean-ablation (replace a component's per-sample output by its dataset
mean: the input-specific information goes, the bias stays).

**(a) Layer level.** Cumulative ablation of whole layers, forward / backward / random(5), for MSA
heads and MLPs of each tower separately. This is the retrieval analogue of the paper's Fig. 1.
"""),
    code("""
lay = load("6_2", "layer_ablation")
lay = lay[lay.mean_source == "self"]
for model in AVAIL[:1]:                       # per-model tables; loop over AVAIL for all
    for (tw, kd), g in lay[lay.model == model].groupby(["tower", "kind"]):
        display(Markdown(f"**{model} - {tw} / {kd}** (R@1, mean of both directions)"))
        display(g.pivot_table(index="order", columns="step", values="R@1").round(2))
"""),
    code("""
for m in AVAIL:
    show(f"6_2_layer_ablation_{m}_self")
"""),
    md("""
The **mean source** control: `self` uses COCO's own component means, `ref` the wide-dataset means
(D_I = ImageNet, D_T = the 1500-noun sentence set). If the conclusion depended on which mean we
substitute, the ablation would be measuring the mean rather than the component.
"""),
    code("""
cmp = load("6_2", "layer_ablation")
piv = (cmp[cmp.order == "backward"]
       .pivot_table(index=["model", "tower", "kind"], columns="mean_source", values="R@1",
                    aggfunc=lambda s: s.iloc[-1]).round(2))
piv["delta"] = (piv.get("ref", np.nan) - piv.get("self", np.nan)).round(2)
display(piv)
"""),

    md("""
**(b) Component level.** Every head and MLP slot is scored, then ablated cumulatively in that order.

- **Metric B** (literal Eq. 19-20 at the checkpoint's learned tau): since
  `exp(l_ij - l_ii) = prod_ab r_ab`, the per-component factor is the product over partners, which
  telescopes to `r_a(i,j) = exp((1/tau) w_a(i) [cos(c_a(i), y_j) - cos(c_a(i), y_i)])`.
  `B(a) = E[r_a] < 1` means the component pushes positives above negatives.
- **Metric A**: the PVE-weighted Hungarian PC-cosine against the other tower, max over partners.
- **norm**: `E||c_a||`, the loudness control.
- **random** (3 seeds) and **metricB_rev** (worst-first) bound the useful range.
"""),
    code("""
sc = load("6_2", "component_scores")
for m in AVAIL[:1]:
    for tw in ["vision", "text"]:
        g = sc[(sc.model == m) & (sc.tower == tw)].nsmallest(8, "metricB")
        display(Markdown(f"**{m} / {tw} - most discriminative components (lowest E[r_a])**"))
        display(g[["kind", "layer", "head", "metricB", "metricB_log", "metricA", "lambda_geo",
                   "norm"]].round(4).reset_index(drop=True))
"""),
    code("""
comp = load("6_2", "component_ablation")
comp = comp[(comp.mean_source == "self")]
for m in AVAIL[:1]:
    for mode in ["ablate", "keep"]:
        g = comp[(comp.model == m) & (comp["mode"] == mode) & (comp.tower == "joint")]
        display(Markdown(f"**{m} - joint {mode} curve** (R@1 vs # components)"))
        display(g.pivot_table(index="order", columns="step", values="R@1").round(2).iloc[:, :10])
"""),
    code("""
for m in AVAIL:
    show(f"6_2_component_ablate_{m}_self")
    show(f"6_2_component_keep_{m}_self")
"""),
    md("""
**How much does the decomposition cancel?** The components sum to a unit-norm embedding, so
`sum_a E||c_a||` *is* the cancellation ratio. It is the health check on any per-component
attribution: the larger it is, the more each direct effect is a small difference of large
numbers. Causal mean-ablation curves are unaffected (they intervene rather than attribute),
but a ranking computed on the ResNet vision tower has to be read with this in mind.
"""),
    code("""
rows = []
for m in AVAIL:
    sc_m = load("6_2", "component_scores", [m])
    for tw in ["vision", "text"]:
        g = sc_m[sc_m.tower == tw]
        if g.empty: continue
        rows.append(dict(model=m, tower=tw, n_components=len(g),
                         cancellation=round(g["norm"].sum(), 2)))
canc = pd.DataFrame(rows).pivot(index="model", columns="tower",
                                values=["n_components", "cancellation"])
display(canc)
"""),
    md("""
**Cross-model summary.** How concentrated importance is, and which ranking actually finds the
load-bearing components (lower retained R@1 after ablating the same budget = better ranking).
"""),
    code("""
show("6_2_concentration_joint")
show("6_2_ranking_quality_joint")
"""),
    md("""
**Where they sit, and the loud-vs-informative dissociation.** The map is the literal answer to the
section's title; the scatter is the paper's "norm hiding" claim - the norm-fraction weights of
Eq. (12) are not a distribution, so a component can be loud and carry no pair-discriminative signal.
"""),
    code("""
for m in AVAIL:
    show(f"6_2_component_map_{m}")
    show(f"6_2_loud_vs_informative_{m}")
"""),

    md("""
### 6.2b A pair-level ranking from r_ab * r~_ab

A second, pair-level score built from the loss factors of Eq. (19)-(20): for a component pair
(a, b), sum over every positive index i and both negative indices j and k,

$$\\mathrm{score}(a,b)=\\mathbb{E}_{i,j,k}\\big[r_{ab}(i,j)\\,\\tilde r_{ab}(k,i)\\big],
\\qquad \\tau\\log[\\cdot]=S_{ab}(i,j)+S_{ab}(k,i)-2S_{ab}(i,i)$$

with $S_{ab}(i,j)=\\langle c_a(m_i), d_b(m'_j)\\rangle$ (the stored components are already divided
by the output norm). **Small = aligned**: the pair scores matched pairs above mismatched ones, in
*both* directions at once - where the per-component Metric B fixes only one. Since j and k are
independent given i the triple sum factorises into one logsumexp each, evaluated in log space so
the loud scaffold pairs cannot overflow.

Three orderings come out of the [P, Q] score matrix. `pair_min` and `pair_mean` reduce it to one
number per component (best partner / aggregate over all partners) and then pool **both towers into
a single ranked list, one component ablated per step** - exactly the protocol the Metric B curves
use, so the two are directly comparable. `pair_matched` is the new part: a greedy one-to-one
matching, so each step removes one vision **and** one text component.
"""),
    code("""
rows = []
for m in AVAIL:
    d = load("6_2", "pair_ablation", [m])
    if d.empty: continue
    d = d[d["mode"] == "ablate"]; base = d[d.step == 0]["R@1"].mean()
    r = {"model": m, "R@1": round(base, 2)}
    for o in ["metricB", "pair_min", "pair_mean", "pair_matched", "random"]:
        g = d[d.order == o].groupby("step")["R@1"].mean().sort_index()
        if g.empty: continue
        # normalised area under the ablation curve; LOWER = ranking removes the signal sooner
        r[o] = round(float(np.trapz(g.values, g.index) / (base * max(g.index))), 3)
    rows.append(r)
display(pd.DataFrame(rows).set_index("model"))
"""),
    md("""
The same thing in the units that matter - **R@1 points lost** after mean-ablating the first N
pooled components (bigger = the ranking found the load-bearing components sooner):
"""),
    code("""
STEPS = [1, 2, 4, 8, 16, 32]
for m in AVAIL:
    d = load("6_2", "pair_ablation", [m])
    if d.empty: continue
    d = d[d["mode"] == "ablate"]; base = d[d.step == 0]["R@1"].mean()
    tab = {}
    for o in ["metricB", "pair_min", "pair_mean", "pair_matched", "random"]:
        g = d[d.order == o].groupby("step")["R@1"].mean()
        if g.empty: continue
        tab[o] = [round(base - g[s], 2) if s in g.index else np.nan for s in STEPS]
    print(f"{m}  (baseline R@1 = {base:.2f})  -- R@1 points lost")
    display(pd.DataFrame(tab, index=[f"N={s}" for s in STEPS]).T)
"""),
    md("""
The ranking is **not** uniformly better than Metric B, and the pattern is worth stating plainly:
the pair score wins on the two small ViTs, `pair_mean` roughly ties on L-14/H-14, and Metric B
stays clearly ahead on both ResNets. `pair_mean` is the most robust of the three variants, which
is the expected direction - it aggregates over all partners, like Metric B's product, whereas
`pair_min`/`pair_matched` commit to a single partner.

This is not a sample-size artifact: subsampling the triple expectation to n=384/256/128 leaves the
per-component ranking at Spearman 0.997/0.994/0.986 against n=512, with an identical top pair.
"""),
    code("""
for m in AVAIL:
    show(f"6_2_pair_ranking_{m}")
    show(f"6_2_pair_map_{m}")
"""),

    md("""
### 6.2c The loss restricted to one component

Metric B is already per-component *and* already aggregates over every partner: its `r_a` equals
`prod_b r_ab(i,j)`, because the product telescopes (`sum_b S_ab(i,j) = <c_a(i), y_j>`). Taking the
unrolled objective (Eqs. 26-27) and keeping only component `a` fixes the one remaining difference -
the loss is a **product over positives**, not a mean of exponentials:

$$\\log \\mathrm{score}_a=\\sum_i\\Big[\\mathrm{LSE}_j\\big(G_{ij}-G_{ii}\\big)
+\\mathrm{LSE}_k\\big(G_{ki}-G_{ii}\\big)\\Big],\\qquad G=\\hat c_a\\hat Y^{\\top}/\\tau$$

So it weights every positive equally instead of being dominated by the worst one. `lossB_greedy`
re-scores after each ablation: a component's score depends on the *other* tower, so each text
ablation moves `Y` and re-scores every vision component. Scoring uses 256 COCO pairs; every curve
below is the full 5,000.
"""),
    code("""
rows = []
for m in AVAIL:
    d = load("6_2", "loss_ablation", [m])
    if d.empty: continue
    d = d[(d["mode"] == "ablate") & (d.tower == "joint")]
    base = d[d.step == 0]["R@1"].mean(); r = {"model": m, "R@1": round(base, 2)}
    for o in ["lossB", "lossB_greedy", "pair_matched", "random"]:
        g = d[d.order == o].groupby("step")["R@1"].mean()
        if not g.empty and 16 in g.index:
            r[o] = round(base - g[16], 2)
    rows.append(r)
display(Markdown("**R@1 points lost after ablating 16 pooled components**"))
display(pd.DataFrame(rows).set_index("model"))
for m in AVAIL:
    show(f"6_2_loss_ranking_{m}")
"""),

    md("""
### 6.2d Orderings that account for what is already selected

Every ranking so far is *marginal*: it asks what a component is worth alone, never what it adds
given the ones already chosen, so redundant components crowd the top. These two orderings apply
the greedy rule instead, at all three levels of 6.2 (`vision`, `text`, `joint`):

* **forward** - start fully mean-ablated, repeatedly ADD the component that most reduces the loss
  given the current set (matched to the reconstruction curve);
* **backward** - start intact, repeatedly REMOVE the component whose removal most INCREASES the
  loss given what is already gone (matched to the ablation curve).

Selection runs on 256 COCO pairs, the curves on the full 5,000, so the order is not read off the
data it is scored on.
"""),
    code("""
for m in AVAIL:
    show(f"6_2_greedy_order_{m}")
d = load("6_2", "greedy_order_order", AVAIL)
if not d.empty:
    display(Markdown("**First 8 components of the forward ordering (joint)**"))
    for m in sorted(set(d.model)):
        g = d[(d.model == m) & (d.tower == "joint") & (d["mode"] == "forward")].nsmallest(8, "rank")
        print(m, " -> ".join(f"{r.comp_tower[0]}:{r.kind}{r.layer}"
                             f"{'' if r.head < 0 else 'h' + str(r.head)}" for r in g.itertuples()))
"""),

    md("""
### 6.2e Greedy pool: a training-free subset chosen on a tiny calibration set

Same greedy rule, but stopped when the loss no longer improves, and driven by a calibration set of
**correct pairs only** (1 per class). Nothing is trained. The pool is then evaluated on the FULL
task with everything outside it mean-ablated, against the full model, same-budget random pools,
and the top-|pool| of the marginal ranking.

Two caveats to read the numbers with. (i) The calibration set is drawn from the evaluation set, so
there is contamination: negligible for FairFace (7/3,500) and small for CIFAR-100 (5%) and Caltech
(4.9%), but **20% for ImageNet** (1,000 of 5,000). (ii) The pool was capped at 96 components, and
on ImageNet the loss was still improving at the cap, so that column is truncated rather than
converged - the `_inbig` re-run lifts the cap.
"""),
    code("""
show("6_2_pool_bars")
d = load("6_2", "greedy_pool_eval", AVAIL)
if not d.empty:
    display(d.pivot_table(index=["model", "task"], columns="method", values="acc",
                          aggfunc="mean").round(2))
    display(Markdown("**Pool size at the accuracy peak (mean over calibration draws)**"))
    b = d[d.method == "greedy pool (best prefix)"]
    display(b.pivot_table(index="model", columns="task", values="size", aggfunc="mean").round(1))
"""),
    md("""
The trajectory is the informative part: the calibration loss falls monotonically by construction,
but task accuracy need not. Where the calibration set is tiny (FairFace, 7 pairs) accuracy peaks
early and then decays - the pool starts fitting the calibration set. Where it is larger
(CIFAR-100, 100 pairs) the peak sits near the cap.
"""),
    code("""
for m in AVAIL:
    show(f"6_2_pool_trajectory_{m}")
show("6_2_pool_retrieval")
"""),

    md("""
### 6.2f Waterbirds: the same selector against a spurious correlation

Waterbirds correlates the bird class with the background, so CLIP leans on the background and the
metric that matters is **worst-group** accuracy over the four (class, background) cells. The four
groups are recovered by joining the saved subset index map back to the dataset metadata.

Two changes are forced by the task. With two classes the symmetric InfoNCE degenerates (a 2x2
matrix, and with several images per class the "negative" texts are copies of the positive), so the
calibration objective is the zero-shot classification loss `CE(E_img E_txt^T / tau, labels)` over
the C unique class names. And two calibration regimes are reported because they assume different
supervision: **class-balanced** uses only class labels (the same supervision as everywhere else),
**group-balanced** also uses the background label and is therefore an upper bound, not a method.

Calibration images are **excluded from the evaluation**, and the accuracy-optimal prefix is
labelled `ORACLE` in the CSV because it is chosen by reading the test metric - it bounds the
headroom, it is not a result.
"""),
    code("""
show("6_2_waterbirds")
wb = pd.concat([pd.read_csv(f) for f in
                glob.glob(os.path.join(RES_DIR, "6_2", "waterbirds_eval_*.csv"))],
               ignore_index=True) if glob.glob(
    os.path.join(RES_DIR, "6_2", "waterbirds_eval_*.csv")) else pd.DataFrame()
if wb.empty:
    print("[pending] waterbirds not run yet")
else:
    honest = wb[~wb.method.str.contains("ORACLE")]
    display(Markdown("**Worst-group accuracy** (calibration images excluded)"))
    display(honest.pivot_table(index="model", columns=["regime", "method"],
                               values="worst_group", aggfunc="mean").round(2))
    display(Markdown("**Total accuracy** - the gain is not a worst-group / average trade-off"))
    display(honest.pivot_table(index="model", columns=["regime", "method"],
                               values="total", aggfunc="mean").round(2))
    display(Markdown("**Oracle prefix (upper bound only, selected on the test metric)**"))
    display(wb[wb.method.str.contains("ORACLE")].pivot_table(
        index="model", columns="regime", values=["worst_group", "size"], aggfunc="mean").round(2))
"""),

    md("""
## 6.3 Intrinsic dimensionality of the components

Per component, per cumulative layer, and for the final embedding:
linear ID (#PCs at 80/95/99% variance), nonlinear ID (TwoNN), their ratio, EVR_1, and the
component's position relative to the two modality centroids.
"""),
    code("""
idf = load("6_3", "intrinsic_dim")
display(Markdown("**Final embeddings**"))
display(idf[idf.level == "final"][["model", "tower", "pca80", "pca95", "pca99", "twonn", "ratio",
                                   "evr1", "dist_img_mean", "dist_txt_mean"]].round(3))
display(Markdown("**Per-head means by depth (vision)**"))
g = idf[(idf.level == "component") & (idf.tower == "vision") & (idf.kind == "attn")]
display(g.groupby(["model", "layer"])[["pca95", "twonn", "ratio", "evr1"]].mean()
        .round(2).groupby("model").tail(4))
"""),
    md("""
The heat-table below is the depth profile of the cumulative residual stream: **L** = #PCs at 95%
variance, **N** = TwoNN, **Ratio** = L/N, **EVR_1** = leading eigenvalue share. Each column has its
own sequential ramp normalised across models, so colour reads as magnitude within a quantity.
"""),
    code("""
show("6_3_id_table_vision")
show("6_3_id_table_text")
show("6_3_ratio_depth_vision")
show("6_3_ratio_depth_text")
"""),
    md("""
**What each layer contributes on its own (not cumulative).** The cumulative view answers "how big
is the residual stream by layer l"; this one answers "how big is what layer l adds". Three values
per layer, in the same grid: the sum of that layer's **heads**, its **MLP**, and the **two
together**. Note the MLP curve is a single component while the heads curve is a sum of H of them,
so the gap between them is a statement about how much structure the MSA block adds over the MLP at
the same depth - not a like-for-like count.
"""),
    code("""
idl = idf[idf.level == "layer_own"] if "level" in idf else pd.DataFrame()
if idl.empty:
    print("[pending] per-layer (non-cumulative) rows not computed yet")
else:
    for col in ("twonn", "pca95", "ratio"):
        show(f"6_3_id_layer_own_{col}")
    display(Markdown("**Per-layer own contribution, last 4 layers (vision)**"))
    g = idl[idl.tower == "vision"]
    display(g.groupby(["model", "layer", "kind"])[["pca95", "twonn", "ratio", "evr1"]]
            .mean().round(2).groupby("model").tail(12))
"""),
    md("""
**Before vs after the output projection.** The same statistics computed on the *pre-projection*
residual stream (the encoder's own width, `space="pre"`), where available. Those components do
not live in the shared space and do not sum to the CLIP embedding, so they are used for
dimensionality only - never for retrieval.
"""),
    code("""
pre = pd.concat([pd.read_csv(f) for f in
                 glob.glob(os.path.join(RES_DIR, "6_3", "intrinsic_dim_*_pre.csv"))],
                ignore_index=True) if glob.glob(os.path.join(RES_DIR, "6_3", "intrinsic_dim_*_pre.csv")) else pd.DataFrame()
if pre.empty:
    print("[pending] pre-projection run not available yet")
else:
    post = load("6_3", "intrinsic_dim")
    both = pd.concat([post.assign(space="post"), pre], ignore_index=True)
    fin = both[both.level == "final"]
    display(fin.pivot_table(index=["model", "tower"], columns="space",
                            values=["d", "pca95", "twonn", "ratio"]).round(2))
"""),
    code("""
for m in AVAIL:
    show(f"6_3_id_components_{m}")
    show(f"6_3_modality_geometry_{m}")
"""),

    md("""
## 6.4 Linear approximation of the top components

Replace the target components by a rank-k approximation and re-measure retrieval; everything else
stays exact. Four bases:

- **pc** - the component's own top-k PCs (variance-optimal, the upper bound);
- **text / image / both** - for each PC, the pool embedding maximising `|cos(p_j, e)|`
  (greedy, no repeats), i.e. PCLens' labelling rule. Pools are mean-centred, as PCLens does, so the
  modality cone does not dominate the selection.

Targets: `late4` (every head of the last 4 layers) and `topB32` (top-32 components by Metric B).
`cos_recon` is the cosine between the approximated and the exact embedding.
"""),
    code("""
la = load("6_4", "linear_approx")
for m in AVAIL[:1]:
    for target in ["late4", "topB32"]:
        g = la[(la.model == m) & (la.target == target) & (la.tower == "joint")]
        if g.empty: continue
        display(Markdown(f"**{m} / {target} / joint - R@1 (baseline "
                         f"{la[(la.model==m)&(la.basis=='exact')]['R@1'].iloc[0]:.2f})**"))
        display(g.pivot_table(index="basis", columns="k", values="R@1").round(2))
        display(g.pivot_table(index="basis", columns="k", values="cos_recon").round(3))
"""),
    code("""
for m in AVAIL:
    show(f"6_4_linear_approx_{m}_late4_joint")
    show(f"6_4_approx_heat_{m}_joint")
show("6_4_k90_late4_joint")
"""),

    # ================================================================= 6.5 applications
    md("""
---
# 6.5 Does the ranking DO anything? Four downstream failure modes

Everything above measures components. This section asks whether the loss-driven selector fixes
things CLIP is documented to get wrong - with no training, no labels beyond a handful of
calibration examples, and no change to the weights. Only which components stay un-ablated changes.

**The four selectors.** The state is a subset of KEPT components; forward and backward enter the
same space from opposite ends, and the `+swap` variants may undo an earlier decision:

| selector | starts from | moves |
|---|---|---|
| `forward` | nothing kept | add only |
| `forward+swap` | nothing kept | add, or drop one already chosen (budget of 100 undos) |
| `backward` | everything kept | remove only |
| `backward+swap` | everything kept | remove, or restore one removed (budget of 100 undos) |

All four run to **convergence** - they stop when no single move lowers the calibration loss. The
undo budget exists so a swap-enabled run cannot spend its steps oscillating between two states; it
does not stop the search finishing.

**Two controls carried throughout, because without them the numbers mean nothing:**

- **random pool** - a random subset of the *same size*. Mean-ablating most of a network changes the
  output a lot; the question is whether the *choice* matters, not the count.
- **held-out evaluation** - calibration items are excluded from every reported number.

Where an `ORACLE` row appears it is the best prefix chosen *by reading the test metric*. It is an
upper bound on available headroom, **not a method**.
"""),

    md("""
### Integrity check: did the searches converge, or just run out of steps?

A greedy search that stops because it hit its step budget looks exactly like one that stopped
because no move improved the loss - same CSV, same monotone loss curve, plausible numbers. But the
first is a truncated search whose answer depends on an arbitrary budget, and nothing drawn from it
is safe. This has caught real errors in this project more than once, so it runs as a standing check
over every trace family rather than being trusted per-script.

The signature of a cap is that a large share of one model's runs stop at *exactly* the same,
largest step. Converged runs scatter.
"""),
    code("""
from scripts_paper import audit_convergence
audit_convergence.main()
"""),

    md("""
## 6.5.1 Waterbirds - a spurious correlation

Waterbirds correlates the bird class with the background, so CLIP leans on the background. The
metric is **worst-group** accuracy over the four (class, background) cells - total accuracy can
rise while the shortcut gets worse.

Two calibration regimes: `class` uses only class labels (the supervision the rest of the pipeline
assumes); `group` also uses the background label, so it is an upper bound, not a free method.
"""),
    code("""
wb = load("6_2", "waterbirds_eval")
if wb.empty:
    print("[pending] waterbirds not run yet")
else:
    sel = wb[wb.method.isin(["full model", "selected"])]
    piv = sel.pivot_table(index=["regime", "variant"], columns="model",
                          values="worst_group", aggfunc="mean").round(1)
    display(Markdown("**Worst-group accuracy (%)**")); display(piv)
show("6_2_waterbirds_variants_comp")
"""),
    md("""
**The results table**, in the layout the paper uses: each cell is *total, worst-group*, shaded per
column so colour reads as "which method wins for this model" rather than "which model is best".
"""),
    code("""
for reg in ("class", "group"):
    show(f"6_2_waterbirds_table_{reg}")
"""),
    md("""
**Every group, not just the worst.** A method can lift the worst cell by flattening a strong one,
and the summary numbers hide that. The starred columns are the minority cells - the bird on the
background it is *not* correlated with - which is where the shortcut lives.
"""),
    code("""
wt = load("6_2", "waterbirds_table")
if wt.empty:
    print("[pending] run scripts_paper.exp_waterbirds_table")
else:
    cols = ["landbird_land", "landbird_water", "waterbird_land", "waterbird_water",
            "class_0", "class_1", "total", "worst_group"]
    g = wt[wt.method.isin(["full model", "selected"]) & (wt.regime.isin(["-", "class"]))]
    display(g.groupby(["model", "variant"])[cols].mean().round(1))
for m in AVAIL:
    show(f"6_2_waterbirds_groups_{m}_class")
"""),
    md("""
**The search trajectory.** The calibration loss falls monotonically by construction; held-out
worst-group accuracy does not have to follow it. Where the two part company, the selector is
overfitting a handful of calibration images.
"""),
    code("""
show("6_2_waterbirds_trace_class_comp")
show("6_2_waterbirds_trace_group_comp")
"""),

    md("""
## 6.5.2 Finer units: (component, PC) instead of whole components

A component is not atomic. Writing it through its own spectral basis,
`c_a(m) = mean_a + sum_k alpha_k^a(m) p_k^a + resid`, makes the unit a pair (component a, PC k),
and mean-ablation becomes zeroing `alpha` - exactly PCSelection, but driven by the loss instead of
by a hand-written query. Everything not selectable (component means + the residual beyond the kept
PCs) is a fixed scaffold, so the decomposition stays exact.

This matters for *explanation*, not only for accuracy: a PC is a direction, and a direction can be
labelled with the texts and images at each of its poles.
"""),
    code("""
pc = load("6_2", "waterbirds_pc_eval")
if pc.empty:
    print("[pending] PC-level run not available yet")
else:
    display(pc[pc.method.isin(["full model", "selected"])]
            .pivot_table(index=["regime", "variant"], columns="model",
                         values="worst_group", aggfunc="mean").round(1))
show("6_2_waterbirds_pc_vs_comp")
show("6_2_waterbirds_variants_pc")
"""),
    md("""
**What the search actually picked.** Forward ADDS units (these carry the task), backward DROPS them
(these carry the spurious cue), so the two lists answer different questions and both are worth
reading.
"""),
    code("""
for m in AVAIL:
    show(f"6_2_pc_moves_{m}_class")
"""),
    code("""
mv = load("6_2", "waterbirds_pc_moves")
if mv.empty:
    print("[pending] no PC moves yet")
else:
    for (m, v), g in mv[(mv.regime == "class") & (mv.rep == 0)].groupby(["model", "variant"]):
        if v not in ("forward", "backward"): continue
        act = "add" if v == "forward" else "drop"
        g = g[g.action == act].head(8)
        display(Markdown(f"**{m} / {v}: first {len(g)} units {act}ed**"))
        display(g[["step", "tower", "kind", "layer", "head", "pc", "loss"]]
                .reset_index(drop=True).round(4))
"""),
    md("""
The labelled version of those units - top texts and images at *both* poles of each PC - is printed
by `scripts_paper/explain_pcs.py` and appears in the job logs (`logs/allB_*.out`), because it needs
the activation files rather than a CSV:

```bash
python -m scripts_paper.explain_pcs --task waterbirds --model ViT-L-14 \\
    --variant forward --regime class --top 15
```
"""),

    md("""
## 6.5.3 Bias audit - dehumanisation and crime labels

The CLIP paper reports that the model assigns non-human and crime categories to photographs of
people, at different rates for different groups. This runs that audit on FairFace faces against
ImageNet primates, with 12 competing prompts (person / man / woman / child, animal / gorilla /
chimpanzee / orangutan / monkey, thief / criminal / suspicious person).

A two-class "person vs monkey" contrast shows nothing - it is trivially separable (100% accuracy,
0% error for every group), which is why the multi-class protocol is the one that measures anything.

**Two numbers must move together for a claimed fix to be real:** the harmful rate on faces has to
fall, *and* the primate images have to stay correctly classified. Otherwise the model has merely
collapsed the label space, which would look like a fix and be worthless.

**The finding splits by checkpoint, and the split matters for how the result is read.** The
CLIP paper studied the *OpenAI* models, which here are RN50 and RN101; the ViTs are LAION
checkpoints. The dehumanisation effect reproduces on the OpenAI models and essentially vanishes on
the LAION ones:

- **RN101 (OpenAI)** - 3.69% of faces get a non-human label overall, but **10.51% of Black faces**
  against 2.09-3.26% for every other group: a 3-5x disparity, which is the CLIP paper's result.
- **RN50 (OpenAI)** - 2.37% overall, again highest on Black faces (4.85%).
- **LAION ViTs** - 0.00% non-human for *every* group. Whatever else these checkpoints do, they do
  not make this error.

The crime categories behave differently again: the OpenAI models assign one to ~44-48% of all faces
(RN101 peaks at 55.5% for "east asian"), the LAION ViTs to 0.9-5.8%. So the two harms do not travel
together, and reporting only an aggregate "bias score" would merge two unrelated effects.

**What selection does, and why the random control is what makes it a result.** Held-out, averaged
over 3 repetitions:

| model | | human | non-human | crime | worst group | primates OK |
|---|---|---|---|---|---|---|
| RN101 | full | 48.6 | 3.69 | 47.7 | 10.51 | 100.0 |
| | **forward+swap** | **94.2** | **0.55** | **5.2** | **1.46** | **100.0** |
| | random, same size | 34.3 | 14.12 | 51.6 | 23.28 | 82.9 |
| RN50 | full | 53.8 | 2.37 | 43.9 | 4.85 | 98.9 |
| | **forward** | **99.7** | **0.27** | **0.01** | **0.62** | **100.0** |
| | random, same size | 47.4 | 18.00 | 34.6 | 25.87 | 76.1 |

The random pool is the reason this is not just "ablating things changes the output": at an
identical budget it makes the harm **worse** (RN101 3.69 -> 14.12% non-human) and breaks the
primate check (100 -> 82.9%), while the selected pool improves both at once. On the LAION ViTs the
crime rate goes to 0.00% and random pools again collapse the primate check - to 34.8% on ViT-B-16.

Forward beats backward on the OpenAI checkpoints here (RN101: 7.6% vs 11.8% crime) using half the
components, which is the opposite of the Waterbirds ordering - neither direction dominates in
general.
"""),
    code("""
ba = load("6_2", "bias_audit_eval")
if ba.empty:
    print("[pending] bias audit not run yet")
else:
    cols = ["size", "face_human", "face_crime", "face_nonhuman", "primate_nonhuman", "disparity"]
    display(ba.groupby(["model", "variant"])[cols].mean().round(2))
    display(Markdown("**Per-group rate of a CRIME label on faces (%)**"))
    gcols = [c for c in ba.columns if c.startswith("crime_")]
    display(ba[ba.variant.isin(["full model", "forward", "backward"])]
            .groupby(["model", "variant"])[gcols].mean().round(2))
show("6_2_bias_audit")
show("6_2_bias_audit_sanity")
"""),
    md("""
**What is being removed?** At PC granularity the dropped units are directions, so they can be
named. This is the part a bias audit actually needs: not "head 9.3 was dropped" but what that
direction responds to.
"""),
    code("""
bp = load("6_2", "bias_pc_eval")
if bp.empty:
    print("[pending] PC-level bias audit not run yet")
else:
    display(bp.groupby(["model", "variant"])[["size", "face_crime", "face_nonhuman",
                                              "primate_nonhuman", "disparity"]].mean().round(2))
bm = load("6_2", "bias_pc_moves")
if not bm.empty:
    for (m, v), g in bm[bm.rep == 0].groupby(["model", "variant"]):
        if v not in ("forward", "backward"): continue
        act = "add" if v == "forward" else "drop"
        g = g[g.action == act].head(10)
        display(Markdown(f"**{m} / {v}: first {len(g)} units {act}ed**"))
        display(g[["step", "tower", "kind", "layer", "head", "pc", "loss"]]
                .reset_index(drop=True).round(4))
show("6_2_bias_pc_anatomy")
"""),

    md("""
## 6.5.4 Matched caption pairs - negation, counting, word order

Three failure modes that a single-caption retrieval score cannot separate from ordinary difficulty.
Each item is (image, correct caption, minimally-edited wrong caption), so **chance is exactly 50%**
and only the edited property is being tested:

| task | the edit | why CLIP struggles |
|---|---|---|
| `negation` | "a photo of a dog" vs "a photo with no dog" | web captions almost never say what is *absent*, so the word "dog" fires the dog concept either way |
| `count` | the caption's own number word swapped | number words are weakly grounded |
| `order` | the two noun phrases around a relation swapped | bag-of-words matching cannot tell them apart |

All three are built from the COCO captions we already have (`scripts_paper/build_hardneg.py`), so
the images are the same ones used everywhere else - no new image extraction, at most 2,000 items
per task.
"""),
    code("""
hn = pd.concat([load("6_2", f"hardneg_eval_{t}") for t in ("negation", "count", "order")],
               ignore_index=True)
if hn.empty:
    print("[pending] hard-negative tasks not run yet")
else:
    display(Markdown("**Pair accuracy (%) - chance is 50**"))
    display(hn.pivot_table(index=["task", "variant"], columns="model",
                           values="acc", aggfunc="mean").round(1))
show("6_2_hardneg")
"""),
    md("""
**Negation, split by whether the object is there.** The usual claim is that this failure is
one-sided - near-perfect when the object is present, near-chance when it is absent. On these
checkpoints it is *not*: the two sides come out close to each other and well below ceiling (ViT-B-32
sits near 67% both ways), so the model is not simply ignoring the negation, it is weak on the
contrast in both directions.

The split is still reported separately, because a selector can raise the average by trading one
side against the other and the average would hide it.
"""),
    code("""
hnn = load("6_2", "hardneg_eval_negation")
if hnn.empty:
    print("[pending]")
else:
    display(hnn.groupby(["model", "variant"])[["size", "acc", "acc_present", "acc_absent"]]
            .mean().round(1))
show("6_2_hardneg_negation_split")
"""),

    md("""
## 6.5.5 The typographic attack - photo vs printed word

Every image is a real ImageNet photograph with the name of a **different** class printed across it
(`scripts_paper/build_typographic.py`, 2,000 images, the standard white-on-black band). CLIP reads
text in images, so each picture now supports two defensible answers and only one is right. The
classifier is the full 1,000-way ImageNet class-name tower we already have, so the printed label is
just one of the thousand - no special-cased binary choice.

**Two numbers, always together, because neither alone can distinguish a fix from damage:**

- `acc_true` - assigned the class the photograph shows (higher is better);
- `attack_rate` - assigned the class printed on it (lower is better).

They are not complements: 998 other classes are available, so both can fall at once. A selector
that drops both is destroying the model. Only one that drops `attack_rate` while holding
`acc_true` is actually suppressing the text-reading route.

The calibration objective is ordinary zero-shot cross-entropy toward the true label on 64 images.
Nothing tells the selector which word was printed - it never sees the attack label.
"""),
    code("""
ty = load("6_2", "typographic_eval")
if ty.empty:
    print("[pending] typographic run not available yet")
else:
    display(ty.groupby(["model", "variant", "method"])[["size", "acc_true", "attack_rate",
                                                        "acc_other"]].mean().round(2))
show("6_2_typographic")
"""),

    md("""
## 6.5.6 Counting, 1 to 10

COCO's own captions cannot carry this: only 543 of the 5,000 contain a number word and 70% of
those are "two", so counts above four are essentially absent. The set is therefore composed -
n instances of one (colour, shape) on a plain canvas, n balanced over 1..10, 2,000 images.

**Why rendered and not photographs.** Pasting ImageNet crops does not give a countable scene: a
"banana" validation image is a market shelf holding dozens of them, so an image labelled *one
banana* shows a pile and the ground truth is simply wrong. Exactness is the one property this task
cannot do without, so the objects are rendered.

**What that costs - and it turned out to cost less than expected.** Rendered images sit outside
CLIP's training distribution, so the risk was that the full model would be at chance and the set
would support no claim at all. The script warns when that happens. It does not happen:

| | full-model `count_acc` | `obj_acc` | MAE |
|---|---|---|---|
| ViT-B-32 | 52.3% | 100.0% | 0.67 |
| ViT-B-16 | 47.1% | 100.0% | 0.87 |

Chance is 10%, so the models carry a real count signal, and `obj_acc = 100%` says the colour and
shape are recognised perfectly - the images are legible to CLIP, and what is hard about them is the
counting specifically. That is exactly the property the task needs.

(`full_acc` equals `count_acc` here for the same reason: with the object identified every time, the
200-way problem collapses onto the count.)

Two scores: `count_acc` holds the object fixed so the 10 candidates differ in one token (chance
10%), and `full_acc` puts all 200 (object, count) pairs in competition. `mae` is the mean absolute
error in counts - a model that is wrong but close is doing something a model that is wrong at
random is not.
"""),
    code("""
cn = load("6_2", "counting_eval")
if cn.empty:
    print("[pending] counting run not available yet")
else:
    display(Markdown("**Chance on `count_acc` is 10%. Read the full-model row first.**"))
    display(cn.groupby(["model", "variant", "method"])[["size", "count_acc", "full_acc",
                                                        "obj_acc", "mae"]].mean().round(2))
show("6_2_counting")
show("6_2_counting_per_n")
"""),

    md("""
## 6.5.7 Specialising on 10 ImageNet classes - and what it costs on the other 990

Pick 10 random ImageNet classes, run the selectors against the 10-way problem, then re-score the
**same kept set** on the full 1,000-way problem. A selector tuned on a narrow task will improve
that task; the only interesting question is what it destroys outside it, so both numbers are
always reported together.

Held fixed so the two evaluations are comparable:

- **means and PC bases come from the FULL ImageNet activations**, never from the 10-class slice.
  Mean-ablation is defined relative to a distribution, so if the subset defined it the "same"
  model would differ between the two evaluations and the comparison would be void;
- calibration images are excluded from **both** evaluations;
- both granularities run: whole components and (component, PC) units.

**Sample size, stated rather than hidden.** The decomposed ImageNet set holds 5 images per class,
so a 10-class subset is 50 images: 2 per class calibrate and 30 are held out. That is a small
evaluation, which is why everything is averaged over **5 independent class draws x 3 calibration
draws** and why the subset column should be read as a trend rather than a precise number.
"""),
    code("""
ins = load("6_2", "insub_eval")
if ins.empty:
    print("[pending] ImageNet subset run not available yet")
else:
    display(Markdown("**10 chosen classes vs the full 1,000-way problem**"))
    display(ins.groupby(["level", "model", "variant", "method"])[["size", "acc_subset",
                                                                 "acc_full"]].mean().round(2))
for lvl in ("component", "pc"):
    show(f"6_2_insub_{lvl}")
"""),
    md("""
**The loss trajectory, with both accuracies on the same axis.** The calibration loss falls
monotonically by construction, so on its own it proves nothing. Putting the 10-class and
1,000-class curves beside it is what makes the plot informative: the point where the 1,000-way
curve turns down while the 10-way curve is still rising is where specialisation starts costing
generality.
"""),
    code("""
for lvl in ("component", "pc"):
    show(f"6_2_insub_trajectory_{lvl}")
tr = load("6_2", "insub_trace")
if not tr.empty:
    g = tr[tr.level == "component"].groupby(["model", "variant", "step"])[
        ["loss", "acc_subset", "acc_full"]].mean().reset_index()
    display(g.groupby(["model", "variant"]).tail(1).round(3))
"""),

    md("""
## 6.5.8 MNIST - and whether the gain is real

CLIP is famously weak on MNIST (the paper reports 88% for the best model; these checkpoints do far
worse with bare digit words). The selector gains a lot here - but bare class names are also a *bad
prompt*, and a method that merely compensates for a bad prompt is not a method.

The control is the same run with a proper prompt template. If the gain survives it, it is real.
"""),
    code("""
for tag in ("_mnist", "_mnistp"):
    d = pd.concat([pd.read_csv(f) for f in
                   glob.glob(os.path.join(RES_DIR, "6_2", f"greedy_pool_eval_*{tag}.csv"))],
                  ignore_index=True) if glob.glob(
                      os.path.join(RES_DIR, "6_2", f"greedy_pool_eval_*{tag}.csv")) else pd.DataFrame()
    if d.empty:
        print(f"[pending] {tag}"); continue
    display(Markdown(f"**MNIST {'bare classnames' if tag=='_mnist' else 'with prompt template'}**"))
    display(d.pivot_table(index="method", columns="model", values="acc", aggfunc="mean").round(1))
show("6_2_mnist_prompt_control")
"""),

    md("""
## Summary table

One row per model: the numbers these three sections contribute to the paper.
"""),
    code("""
def _num(x, nd=1):
    "round, tolerating empty selections (a section may not have run for a model yet)"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if np.isnan(v) else round(v, nd)

rows = []
for m in AVAIL:
    b = load("6_2", "baseline", [m]).iloc[0]
    cj = load("6_2", "component_ablation", [m])
    if not cj.empty:
        cj = cj[(cj.mean_source == "self") & (cj["mode"] == "ablate") & (cj.tower == "joint")]
    def at(order, frac):
        if cj.empty: return np.nan
        g = cj[cj.order == order]
        if g.empty: return np.nan
        j = (g.frac - frac).abs().idxmin()
        return g.loc[j, "R@1"] / b["R@1"] * 100
    idm = load("6_3", "intrinsic_dim", [m])
    fin = idm[idm.level == "final"] if not idm.empty else idm
    def idv(tower, col):
        return fin[fin.tower == tower][col].mean() if not fin.empty else np.nan
    la = load("6_4", "linear_approx", [m])
    lj = la[(la.target == "late4") & (la.tower == "joint")] if not la.empty else la
    base = la[la.basis == "exact"]["R@1"].iloc[0] if not la.empty else np.nan
    def k_for(basis, th=0.9):
        if lj.empty: return np.nan
        g = lj[lj.basis == basis].sort_values("k")
        hit = g[g["R@1"] >= th * base]
        return int(hit.k.iloc[0]) if len(hit) else np.nan
    rows.append(dict(
        model=m,
        R1_i2t=_num(b["R@1_i2t"], 2), R1_t2i=_num(b["R@1_t2i"], 2),
        retained_metricB_10pct=_num(at("metricB", 0.10)),
        retained_random_10pct=_num(at("random", 0.10)),
        twonn_img=_num(idv("vision", "twonn")), twonn_txt=_num(idv("text", "twonn")),
        ratio_img=_num(idv("vision", "ratio")),
        k90_pc=k_for("pc"), k90_text=k_for("text"), k90_both=k_for("both"),
    ))
summary = pd.DataFrame(rows)
out = os.path.join(RES_DIR, "summary_6_2_6_4.csv")
summary.to_csv(out, index=False)
print("wrote", out)
display(summary)
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {NB} ({len(CELLS)} cells)")
