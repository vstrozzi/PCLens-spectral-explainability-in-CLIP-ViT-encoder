# Minimal-example mechanistic debiasing — project plan

**Claim.** Given a handful of examples — as few as one pair — we identify the internal components a
frozen CLIP uses as a shortcut, interpret them, and ablate them, with no training. The same
algorithm transfers across shortcut *types* (background → gender → occupation), across datasets,
and across architectures.

The accuracy numbers become evidence for the mechanism. They are not the contribution.

---

## 0. What already exists (so the plan starts from an honest baseline)

| Piece | Status | Where |
|---|---|---|
| Exact residual-stream decomposition, 6 models, both towers | done | `pclens_core.py` |
| Four selectors (fwd / bwd / ±swap, undo budget, run to convergence) | done | `exp_waterbirds.py` |
| PC-level units `(component, PC)` — the interpretation substrate | done | `exp_waterbirds_pc.py` |
| PC labelling (top texts/images at both poles) | done | `explain_pcs.py` |
| Waterbirds, 6 models, per-group cells | re-running uncapped (job 4145) | `exp_waterbirds_table.py` |
| Bias audit (FairFace × primates × 12 prompts), 6 models | done; RN101/H-14 re-running | `exp_bias_audit.py` |
| Convergence audit (cap ≠ convergence) | done, standing check | `audit_convergence.py` |
| Held-out discipline (calibration excluded everywhere) | done | all `exp_*` |
| Random-pool control at matched budget | done | all `exp_*` |

Roughly **60% of the machinery this plan needs is already built and validated.** What is missing is
mostly *experiments*, not *infrastructure* — plus two datasets.

---

## 1. The framing: one-shot specification, stated plainly

The pair is **not hidden supervision — it is the input**. The user supplies one example per class of
*what they want the model to key on*, exactly as they would write a prompt:

> `(image A, class A) + (image B, class B)` — a one-shot specification of the intended decision rule.

Framed this way the method is a **one-shot mechanistic intervention**: the pair carries the intent,
the algorithm finds the components that implement it, and nothing is trained. This is the honest and
the strong reading at once, and it should be stated in the abstract so no reviewer thinks it is being
smuggled past them.

The comparison that makes the framing meaningful is against methods needing *thousands of
group-annotated* images and gradient steps (GroupDRO, JTT, DFR).

**Two arms remain, now as an ablation rather than a defence:**

- **(a) informative pair** — one minority-group example per class: the user knows what they want.
- **(b) random pair** — one example per class at random: the user picks carelessly.

The gap prices *how much the choice of example matters*. If (b) nearly matches (a) the method is
robust to a careless user; if (a) is far better, "choose an example that exposes the failure" becomes
a documented usage instruction — which is a normal thing for a one-shot method to have.

---

## 2. Datasets

| Dataset | Task (retain) | Shortcut (remove) | Status |
|---|---|---|---|
| **Waterbirds** | bird identity | background water/land | **have it**, decomposed, 6 models |
| **CelebA** blond×gender | hair colour | gender | download + extract (canonical, low-risk) |
| **Doctor–Nurse** | profession | gender | **construct + verify** (see below) |
| **VisoGender** | occupation | gender | **verify availability first** |

### Why CelebA is in the table
The user's three-dataset structure needs a *reliable* demographic shortcut. CelebA blond-hair × gender
is the benchmark every group-robustness paper reports (GroupDRO, JTT, DFR), so our numbers are directly
comparable to published ones, and it downloads cleanly. Doctor–Nurse and VisoGender are more novel but
riskier; CelebA de-risks the middle row so the paper survives if the third row falls through.

### The Doctor–Nurse / VisoGender caveat is real — gate it
Different papers use different subsets, prompts, and even different definitions of the task, exactly
as flagged. **Phase 1 is a verification spike, not a commitment**: download, count images per
(occupation × gender) cell, check the cells are non-empty and balanced enough to compute a worst-group
number, and confirm the licence permits use. VisoGender ships as *URLs*, so link rot is the specific
risk — measure the live fraction before building on it. If it fails, fall back to a
FairFace-conditioned occupation-prompt audit, which we can already run.

---

## 3. Experiments, mapped to the reviewer questions

### Q1 — Why do two images suffice? *(the headline experiment)*
Sweep `n_pairs ∈ {1, 2, 5, 10, 25, 100, all}` × arms {oracle, random} × 4 selectors × 6 models ×
5 seeds. Plot worst-group accuracy vs `n_pairs`, log x-axis.

- **Result shape we want:** the curve is flat from `n=1`.
- **Control that makes it believable:** random pool at matched size, at every `n`.
- **Trap:** with 2 calibration images the loss is nearly degenerate — verify the search does not
  simply memorise them. The held-out gap already tests this; report train/held-out both.
- New script: `exp_npairs.py` (thin loop over the existing selector — ~1 day).

### Q2 — Are the discovered components stable?
Draw `K = 20` independent one-shot pairs, run the selector on each, and measure overlap of the
selected sets — always against a random baseline drawn at the same size.

> **Trap that will otherwise sink this:** full pools are large (~150 of 266), so two *random* pools
> already overlap at Jaccard ≈ 0.43. Raw Jaccard on full pools will look impressive and mean nothing.
> Every overlap number is reported as **excess over the hypergeometric null**, with the null band
> drawn on the same axes.

**The number that carries the claim is the prefix overlap.** Report overlap of the **top-10 and
top-15 components** (and of the prefix at which held-out accuracy first plateaus — the *effective*
set, which is the fairest choice because it is where the method has already done its work). At
k = 10 of 266 the chance Jaccard is ≈ 0.02, so any real agreement is unmistakable and needs no
statistical apparatus to be legible:

| set | chance Jaccard | interpretation if we beat it |
|---|---|---|
| top-10 | ≈ 0.02 | different examples → same heads |
| top-15 | ≈ 0.03 | same, slightly more forgiving |
| plateau prefix | varies, computed per run | the honest "effective set" |
| full pool | ≈ 0.43 | nearly uninformative — reported only for completeness |

New script: `exp_stability.py` (~1 day).

### Q3 — Are the components interpretable, and is the effect selective?
Two halves:

**(a) Interpretation.** For each discovered unit, label it via `explain_pcs.py` (top texts/images at
both poles) and — new — its **spatial attention map** from the saved `*_cls_attn_*` arrays, which we
already extract for the ViTs. This is what turns "head 9.3" into "attends to the water background".

**(b) Selectivity — the strong form.** Build a **double dissociation**: for every candidate component
measure
- Δ accuracy on the *task* (bird / profession), and
- Δ accuracy on the *shortcut attribute*, read out zero-shot ("a photo of a water background" vs
  "a land background"; "a photo of a man" vs "a woman").

A component that removes shortcut accuracy while leaving task accuracy is a *causal, targeted*
intervention; one that drops both is just damage. **Plot the two deltas against each other** — that
scatter is the single most convincing figure in the paper, and it is cheap because both readouts are
zero-shot over embeddings we already have.
New script: `exp_selectivity.py` (~2 days).

### Q4 — Is the intervention causal / counterfactual?
Present the 2×2 transition table (before → after) per group, not just worst-group scalars. We
**already compute every cell** (`exp_waterbirds_table.py`); this is a presentation change plus the
per-image prediction-flip counts:

| | before | after |
|---|---|---|
| waterbird + land | landbird | **waterbird** |
| landbird + water | waterbird | **landbird** |

Report *flip rates in both directions*, including harmful flips (correct → incorrect). Net accuracy
hides those. (~0.5 day, mostly plotting.)

### Q5 — Discovery ≠ evaluation
Already enforced everywhere via the `held` mask. Make it explicit in the paper text and keep the
assertion in code. **No new work.**

### Q6 — Across datasets
Same algorithm, no per-dataset tuning, on all three (four) benchmarks. The one thing to hold fixed
and *say* we held fixed: no hyperparameter is tuned per dataset — same `k`, same seeding rule, same
undo budget.

### Q7 — Across architectures
Already have ViT-B/32, B/16, L/14, H/14, RN50, RN101 — **two families**, stronger than the minimum
asked. Add the cross-architecture consistency question: do the *same functional* components get
discovered (always late-layer, vision-side)? The layer/depth data to answer this already exists.

#### The ResNet caveat — what it does and does not invalidate
The ModifiedResNet decomposition splits a path containing ReLUs, and the earlier frozen-ReLU
attribution audit failed badly (NMAE ≈ 144%, with component cancellation of 9–53× — ViTs included).
So RN50/RN101 results must be split into two categories, because they are affected very differently:

| claim | depends on | status |
|---|---|---|
| **Full-model behaviour** — e.g. RN101 labels 10.5% of Black faces non-human | nothing but a forward pass | **sound**, decomposition irrelevant |
| **Mean-ablation interventions** — kept set → accuracy | only the *sum identity* (`Σ components = embedding`), which `verify_decomposition` asserts per run | **valid as an intervention** |
| **Attribution** — "this component carries the shortcut" | per-component contributions being individually meaningful | **provisional** — heavy cancellation makes single-component attribution unreliable |

Practical consequence: RN results stay in every run (they are already queued), but the paper's
**mechanistic/interpretation claims are ViT-first**, with ResNets reported as a transfer check on the
*intervention*, explicitly flagged. Before any RN attribution claim ships, add a gate:

- **`audit_decomposition.py`** (new, ~0.5 day): assert the sum identity per model, and report the
  cancellation ratio `Σ_a E‖c_a‖ / E‖Σ_a c_a‖`. Models above a stated threshold are labelled
  "intervention valid, attribution provisional" in every table, rather than silently mixed in.

If the RN numbers come out clean under the intervention framing, that is a *bonus* (the method works
even where attribution is murky). If they come out erratic — RN50 forward on Waterbirds already goes
40.7 → 31.7 — the decomposition is the first suspect, and the ViT story is unaffected either way.

---

## 4. Phasing

| Phase | Work | Depends on | Est. |
|---|---|---|---|
| **P0** | Land the running jobs; regenerate figures; re-run convergence audit | jobs 4136/4143/4144/4145/4146 | in flight |
| **P1** | **Data spike**: CelebA download+extract; VisoGender live-link count; Doctor–Nurse construction + protocol verification | — | 2–3 days |
| **P2** | Q1 `exp_npairs.py` on Waterbirds, both arms, 6 models | P0 | 2 days |
| **P3** | Q3 `exp_selectivity.py` — the double-dissociation scatter | P0 | 2 days |
| **P4** | Q2 `exp_stability.py` with the hypergeometric null | P2 | 1 day |
| **P5** | Q4 counterfactual flip tables + figures | P0 | 1 day |
| **P6** | Repeat P2–P5 on CelebA, then Doctor–Nurse / VisoGender | P1 | 3–4 days |
| **P7** | Cross-architecture consistency (Q7) + `audit_decomposition.py` RN gate | P6 | 1.5 days |
| **P8** | Attention-map interpretation figures (Q3a) | P3 | 2 days |

P1 runs in parallel with P2–P5, since those only need Waterbirds.

---

## 5. What would falsify the claim

Written down in advance so we cannot rationalise afterwards:

1. **Q1 fails** — one shot is much worse than 100. Then the contribution is "cheap debiasing", not
   "one-shot discovery". Still a paper; different title.
2. **Q2 fails** — top-10 prefix overlap sits at the chance line (≈ 0.02). Then we are fitting
   image-specific noise and the mechanistic claim collapses. This is the most dangerous one, so run
   it early (P4 immediately after P2).
3. **Q3b fails** — no selectivity; every component that removes the shortcut also removes the task.
   Then the intervention is damage that happens to help worst-group by flattening a majority group.
   **The random-pool control and the per-group table already partly test this**, and RN50 forward on
   Waterbirds (40.7 → 31.7) is an existing example of exactly this failure mode, so it is a live risk.
4. **Q6 fails** — works on Waterbirds only. Then it is a background-detector, not a shortcut-detector.

Risk 3 is the one I would watch hardest: several of our current wins come from *backward* removal of
many components, and "remove 60% of the network and worst-group improves" needs the selectivity
experiment to distinguish mechanism from blunt regularisation.

---

## 6. Relationship to the current PCLens framing

The spectral machinery is not discarded — it becomes **the interpretation step (Q3)**. The pipeline
reads:

```
2 images + 2 class names
   → loss-driven search over (component, PC) units      [§6.5 machinery, built]
   → labelled directions + attention maps               [PCLens labelling, built]
   → ablate / retain, frozen weights                    [built]
   → evaluate on unseen data                            [built]
```

So the paper's spine changes from "here is a lens" to "here is a lens, and here is what it lets you
*do* with two examples". §6.2–6.4 become the method section that justifies why component-and-PC units
are the right granularity.
