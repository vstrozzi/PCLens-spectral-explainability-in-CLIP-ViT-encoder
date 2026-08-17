# Cross-Encoder Component Alignment in Contrastive VLMs
## Formal research plan: math, experiments, and positioning

---

## 0. Notation and setup

Two encoders map into a joint space of dimension $d$:

- **Image encoder** $f_I$: ViT with $L$ layers, $H$ heads per layer. The output representation is read at the **CLS** token.
- **Text encoder** $f_T$: (causal) transformer, $L'$ layers, $H'$ heads. Output read at the **EOS** token.

Each encoder ends with a LayerNorm and a linear projection into the joint space:
$$\hat z_I = P_I\,\mathrm{LN}_I(z_I), \qquad \hat z_T = P_T\,\mathrm{LN}_T(z_T),$$
where $z_I, z_T$ are the final residual-stream states at CLS / EOS.

### 0.1 Residual-stream decomposition (both encoders)

Because every block writes additively into the residual stream (Elhage et al., 2021; Gandelsman et al., ICLR 2024), the final state at the readout token decomposes exactly as

$$z_I \;=\; e_{\mathrm{CLS}} \;+\; \sum_{l=1}^{L}\sum_{h=1}^{H} a^{l,h} \;+\; \sum_{l=1}^{L} m^{l},
\qquad
a^{l,h} = \sum_{t=1}^{N} c^{\,l,h}_{t},$$

where $a^{l,h}$ is the direct write of head $(l,h)$ to the CLS position (itself a sum over source tokens $t$), and $m^l$ is the MLP write at CLS. Identically for the text encoder at EOS:

$$z_T \;=\; e_{\mathrm{EOS}} \;+\; \sum_{l,h} b^{l,h} \;+\; \sum_{l} n^{l}.$$

**Status vs literature.** The image-side identity and the empirical facts that (i) late attention layers carry nearly all the direct effect and (ii) MLP direct writes are near-constant across inputs are established (Gandelsman et al. 2024; confirmed again in the second-order-lens paper, 2406.04341). **The EOS-side decomposition with the same mean-ablation methodology is the novel mirror experiment (E1).**

### 0.2 Handling the final LayerNorm (Gandelsman-style: it is all linear)

Following Gandelsman et al. (2024, Eq. 9), the LayerNorm is rewritten as a **per-sample affine map**. With $\mu_l, \sigma_l \in \mathbb{R}$ the mean and standard deviation of the input token, computed **during the forward pass and then frozen**:

$$\mathrm{LN}(x) \;=\; \gamma \odot \frac{x - \mu_l\mathbf{1}}{\sqrt{\sigma_l^2+\epsilon}} + \beta
\;=\; \underbrace{\left[\frac{\gamma}{\sqrt{\sigma_l^2+\epsilon}}\right]}_{\text{multiplicative}} \odot\; x \;-\; \underbrace{\left[\frac{\mu_l\,\gamma}{\sqrt{\sigma_l^2+\epsilon}} - \beta\right]}_{\text{constant}}.$$

Since $\mu_l$ and $\sigma_l$ are treated as constants (evaluated once on the forward pass), $\mathrm{LN}$ is **exactly affine** in $x$, and the multiplicative term is absorbed into the projection matrix. Define the per-sample effective projection

$$\tilde P_I \;:=\; P_I\,\mathrm{diag}\!\left(\frac{\gamma}{\sqrt{\sigma_l^2+\epsilon}}\right).$$

Then the decomposition distributes over all components **with no residual nonlinearity**:

$$\hat z_I \;=\; \sum_{c \in \mathcal{C}_I} \tilde P_I\, z_c \;-\; P_I\!\left[\frac{\mu_l\,\gamma}{\sqrt{\sigma_l^2+\epsilon}} - \beta\right]
\;=\; \sum_{c\in\mathcal{C}_I} \hat c,$$

where $\mathcal{C}_I$ indexes all components (heads, MLPs, plus a constant pseudo-component $c_0$ absorbing $e_\mathrm{CLS}$ and the LN constant term), and the **effective joint-space contribution** of component $c$ is

$$\boxed{\;\hat c := \tilde P_I\, z_c\;}$$

Everything downstream of the residual stream is therefore **one linear map per sample**; the decomposition is exact given the forward-pass statistics. The per-sample scalar $1/\sqrt{\sigma_l^2+\epsilon}$ inside $\tilde P_I$ is the shared normalizer that makes your "components normalized by the total residual stream" phrasing precise. Two consequences to state in the paper:

1. **Statistics convention.** $\mu_l, \sigma_l$ are computed once on the clean forward pass and frozen. Under interventions (mean-ablation of a component), the standard convention — following Gandelsman et al. — is to keep the frozen statistics, so the decomposition and the intervention share the same linear map. (Recomputing statistics after ablation is a robustness check, not the default.)
2. All component statistics (PCA etc.) must be computed **after** $\tilde P$ (i.e., post-projection, in the joint space), otherwise image and text components live in different spaces and their cosines are meaningless. The projection matrices $P_I \ne P_T$ — the joint space is the only common ground.

---

## 1. The loss decomposition (the methodological core)

### 1.1 Similarity decomposes into pairwise component cosines — exact

The (pre-temperature) CLIP logit for image $i$, text $j$:

$$s_{ij} = \cos(\hat z_I^{(i)}, \hat z_T^{(j)}) = \frac{\hat z_I^{(i)} \cdot \hat z_T^{(j)}}{\|\hat z_I^{(i)}\|\,\|\hat z_T^{(j)}\|}.$$

Substituting the sums:

$$\hat z_I \cdot \hat z_T = \sum_{a\in\mathcal{C}_I}\sum_{b\in\mathcal{C}_T} \hat a \cdot \hat b
= \sum_{a,b} \|\hat a\|\,\|\hat b\|\,\cos(\hat a, \hat b),$$

hence

$$\boxed{\;\cos(\hat z_I, \hat z_T) \;=\; \sum_{a\in\mathcal{C}_I}\sum_{b\in\mathcal{C}_T} \omega_{ab}\,\cos(\hat a,\hat b),
\qquad \omega_{ab} = \frac{\|\hat a\|\,\|\hat b\|}{\|\hat z_I\|\,\|\hat z_T\|}\;}$$

**Validation: this identity is correct** (bilinearity of the dot product; the constant pseudo-components must be included or the identity fails). It is *exact per sample*, requiring no approximation beyond the LN bookkeeping of §0.2.

### 1.2 Per-component contribution: norm × cosine

Marginalizing over one side gives the contribution of a single image component:

$$\phi(a) := \frac{\hat a \cdot \hat z_T}{\|\hat z_I\|\,\|\hat z_T\|}
= \frac{\|\hat a\|}{\|\hat z_I\|}\,\cos(\hat a, \hat z_T),
\qquad \sum_{a} \phi(a) = \cos(\hat z_I, \hat z_T).$$

**Validation: correct**, and this is your "L2 norm multiplied by cosine with the other modality" metric. It cleanly factorizes *magnitude* ($\|\hat a\|$) from *alignment* ($\cos$), which is what lets you find **high-norm but orthogonal components** ($\|\hat a\|$ large, $\cos \approx 0$) — candidates for modality-specific computation (E7).

**Relation to literature.** Gandelsman et al. compute $\langle c, \hat z_T\rangle$ as a direct-effect attribution of the *representation*. The two deltas here are: (i) the norm/cosine factorization used as a taxonomy axis, and (ii) the **double** decomposition over pairs $(a,b)$ of components across the two encoders — no prior work decomposes the *bilateral* similarity into a component–component matrix.

### 1.3 How the loss "sees" this sum — gradient analysis

InfoNCE over a batch of $N$ pairs, temperature $\tau$:

$$\mathcal{L} = -\frac{1}{2N}\sum_{i}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right].$$

The loss is **not** linear in the $s_{ij}$ (log-sum-exp), so strictly the model does not "optimize the sum of pairwise cosines"; it optimizes margins between $s_{ii}$ and $s_{ij}$. The precise statement is at the gradient level:

$$\frac{\partial \mathcal{L}}{\partial s_{ij}} = \frac{1}{\tau}\big(\bar p_{ij} - \delta_{ij}\big)\cdot\frac{1}{2N} \quad\text{(with } \bar p_{ij}\text{ the averaged row/column softmax weights)},$$

and since $s_{ij} = \sum_{a,b}\omega_{ab}\cos(\hat a,\hat b)$, **every component pair inside $s_{ij}$ receives the same scalar gradient weight** $(\bar p_{ij}-\delta_{ij})/\tau$. The pressure toward pairwise cross-encoder alignment is therefore distributed uniformly across the component-pair matrix, modulated only by the norm weights $\omega_{ab}$. This is the defensible version of your claim "implicitly the model is optimizing over this sum."

**A sharper proposition worth including.** For a component $a$ whose only path to the loss is direct (final-layer components), the sample gradient of $s$ w.r.t. its output is

$$\frac{\partial s}{\partial \hat a} = \frac{1}{\|\hat z_I\|}\left(\frac{\hat z_T}{\|\hat z_T\|} - s\,\frac{\hat z_I}{\|\hat z_I\|}\right),$$

which is **identical for all direct-only components of that sample** (it depends only on the totals). Differentiation between late heads can therefore arise *only* through their input-dependence (Jacobians), not through the loss signal itself. This predicts and explains your empirical observations that (i) late components are more constant across the dataset and (ii) some of them converge to shared cross-encoder subspaces. Early components additionally receive indirect gradients through all downstream layers (standard direct/indirect path expansion, Elhage et al. 2021), which is your "earlier components are more interacting."

### 1.4 SigLIP

$$\mathcal{L}_{\mathrm{sig}} = -\frac{1}{N}\sum_{i,j} \log\,\sigma\!\big(z_{ij}\,(t\,s_{ij} + b)\big), \qquad z_{ij}=\begin{cases}+1 & i=j\\ -1 & i\ne j\end{cases}$$

Per-pair, no softmax coupling: the decomposition of §1.1 applies to each $s_{ij}$ independently, and the gradient weight is $z_{ij}\,t\,\big(1-\sigma(z_{ij}(t s_{ij}+b))\big)$ — again a single scalar shared by all component pairs. **The pairwise-alignment story is if anything cleaner for SigLIP**; testing both objectives (E3, E6) is the generality claim.

---

## 2. PC machinery: characterizing and comparing components

### 2.1 PCA of a component over a dataset

For component $a$ with joint-space outputs $\{\hat a_x\}_{x\in\mathcal{D}}$:

$$\hat a_x \approx \mu_a + \sum_{k=1}^{K}\alpha_k(x)\,u_k, \qquad \lambda_k^a = \mathrm{Var}_x[\alpha_k], \qquad \rho_k^a = \frac{\lambda_k^a}{\sum_{k'}\lambda_{k'}^a}\ \ (\text{PVE}).$$

Directions $u_k$ are labeled by nearest text embeddings (TextSpan-style) **and** — your novelty — nearest **image** embeddings, for components of *both* encoders. Same-modality and cross-modality labeling of the same $u_k$ is a built-in consistency check.

### 2.2 Why cosine between PCs is the right comparison — a bound, not an assumption

The dataset-average pairwise dot between components $a$ (image) and $b$ (text):

$$\mathbb{E}_x[\hat a_x\cdot \hat b_x] = \mu_a\cdot\mu_b \;+\; \sum_{k,k'} \mathrm{Cov}\big(\alpha_k,\beta_{k'}\big)\,\big(u_k\cdot v_{k'}\big),$$

and by Cauchy–Schwarz, $|\mathrm{Cov}(\alpha_k,\beta_{k'})| \le \sqrt{\lambda_k^a\,\lambda_{k'}^b}$, so

$$\Big|\mathbb{E}_x[\hat a\cdot\hat b] - \mu_a\cdot\mu_b\Big| \;\le\; \sum_{k,k'} \sqrt{\lambda_k^a\lambda_{k'}^b}\;\big|\cos(u_k,v_{k'})\big|.$$

**This validates your metric**: the PVE-weighted PC-cosine

$$A(a,b) := \sum_{k}\sqrt{\rho_k^a\,\rho_k^b}\;\big|\cos(u_k, v_k)\big|$$

is (the diagonal restriction of) the tight upper envelope on the *covariance channel* through which a head pair can contribute to the loss. Two required refinements:

1. **Don't restrict to diagonal $k\!\to\!k$ matching.** Eigenvectors with near-degenerate eigenvalues rotate arbitrarily between fits; use the full bipartite sum $\sum_{k,k'}$, or better an assignment (Hungarian) over $\sqrt{\rho_k\rho_{k'}}|\cos(u_k,v_{k'})|$.
2. **Report a basis-free robustness check**: linear CKA between component outputs,
$$\mathrm{CKA}(a,b) = \frac{\|C_{ab}\|_F^2}{\|C_{aa}\|_F\,\|C_{bb}\|_F},$$
which measures the same subspace overlap without eigen-decomposition brittleness. If your $A$ and CKA give the same head-pair rankings, the metric is solid. (ResiDual analyzes spectral geometry of visual heads but never compares heads *across encoders*; this cross-encoder matrix $A$ is the novel object.)

### 2.3 PC-level interventions (steering)

Mean-ablation of PC $k$ of component $a$ (projecting out its variance while keeping the mean):

$$\hat a' = \hat a - \big(u_k^\top(\hat a - \mu_a)\big)\,u_k.$$

Concept-conditional selection: given concept embedding $t$ (text or image), ablate $k^\star = \arg\min_k |\cos(u_k, t)|$ (keep the concept, remove the rest) or $\arg\max_k$ (destroy the concept). Effect measured as $\Delta \cos(\hat z_I', \hat z_T^{(t)})$. **Convention**: LN statistics stay frozen from the clean forward pass (§0.2), so the intervention acts through the same linear map $\tilde P$; only the final embedding L2-norm used in the cosine is recomputed. Attributions still do not sum linearly across simultaneous interventions because of that final re-normalization.

**Relation to literature.** ResiDual *rescales* PCs with learned weights $\lambda$ to improve zero-shot accuracy; you perform *concept-conditional, training-free ablations, on both encoders, validated causally on a concept set*. That triple (conditional / bilateral / causal) is the differentiation — make it explicit or reviewers will collapse you into ResiDual.

---

## 3. Experiments

### E1 — Component localization in the **text** encoder (mirror of Gandelsman)
- **Motivation.** All component-level decomposition work is on the image encoder; surveys explicitly note the text side is unexplored (the only head-level text-encoder study is negation-specific, arXiv:2407.10488).
- **Goal.** Establish where the direct effect lives at EOS: confirm late-attention dominance and MLP near-constancy, or find a different profile (interesting either way — causal masking might shift the profile earlier).
- **Procedure.** Mean-ablate $\{n^l\}$ (all text MLPs), $\{b^{l,h}\}$ layer-by-layer, and the constant term, over ImageNet class prompts + a caption corpus (avoid prompt-template bias); measure zero-shot accuracy and retrieval R@k. Repeat on the image side only as sanity replication.
- **Deliverable.** A "direct-effect map" figure for both encoders side by side.
- **Risk.** Text prompts for ImageNet are short and templated → low variance; use richer captions (COCO/Flickr30k) as the dataset for statistics.

### E2 — PC characterization of heads in both encoders, labeled by both modalities
- **Motivation.** TextSpan labels image heads with text; nobody labels text-encoder heads with images, and nobody cross-labels. Your observation that PC *signs* are not semantically symmetric (e.g., "black" and "white" not at opposite ends) is itself a reportable geometric finding about high-dimensional concept encoding.
- **Goal.** A catalog: for every late head of both encoders, top-$K$ PCs, PVE, text label, image label, cross-consistency score.
- **Procedure.** PCA per §2.1 on ImageNet-scale data; label $u_k$ (and $-u_k$ separately, given your sign finding) by nearest neighbors in a large text set and image set.
- **Differentiation.** vs TextSpan: bilateral and image-based labeling; vs ResiDual: they characterize specialization, you attach *directional semantic labels* and use them for interventions.

### E3 — Empirical validation of the loss decomposition
- **Motivation.** §1 is exact math, but the paper needs the numerics: show the identity reconstructs $s_{ij}$ to machine precision, then show the *approximation quality* when truncating to top PCs / top pairs.
- **Goal.** (i) Exactness check; (ii) show the component-pair matrix $S_{ab} = \omega_{ab}\cos(\hat a,\hat b)$ is sparse/structured (few pairs carry most of $\cos(\hat z_I,\hat z_T)$); (iii) repeat for SigLIP.
- **Procedure.** Heatmaps of $\mathbb{E}_x[S_{ab}]$ over positive pairs; cumulative curves (fraction of similarity recovered by top-$m$ pairs); compare CLIP vs SigLIP vs model scale.
- **This experiment is the paper's Figure 2.** No prior work has this object.

### E4 — Causal PC steering on both encoders
- **Motivation/Goal.** Show PCs are not just descriptive: ablating the least-concept-similar PCs preserves/enhances the concept ($\cos$ up), ablating the most-similar destroys it, on a full concept set, in **both** encoders.
- **Procedure.** Concept set $\mathcal{T}$ (e.g., 100 concepts, each with text and image exemplars). For each concept, intervene per §2.3, report $\Delta\cos$ distributions with random-PC and random-head controls, both intervention directions.
- **Differentiation.** ResiDual (learned reweighting, image side) → you: training-free, concept-conditional, bilateral, with destroy-direction controls.

### E5 — Spurious-correlation removal by cosine-ranked PC ablation
- **Motivation.** Your finding that ranking PCs by $|\cos(u_k, t_{\mathrm{spur}})|$ beats head-level ablation.
- **Goal.** Worst-group accuracy on Waterbirds (and CelebA) via PC-level ablation, image **and** text encoder.
- **Procedure.** $t_{\mathrm{spur}}$ = embedding of spurious attribute ("water/land background"); ablate top-ranked PCs across heads; report WG / Avg / Gap.
- **Mandatory baselines** (this space is crowded): Gandelsman head-ablation, RepDecompose/CompAlign (NeurIPS 2024, CLIP WG 0.507→0.744), Debiasing-CLIP/LTC (2505.17425), and prompt-space methods (Orth-Cali, RoboShot, Perception CLIP). **Judgment: keep this as a supporting application, not the headline** — a Waterbirds-only paper will be rejected as incremental.

### E6 — Cross-encoder head-pair matching and causal collapse (headline experiment)
- **Motivation.** §1–2 predict that specific head *pairs* $(a,b)$ across encoders carry the alignment.
- **Goal.** (i) Compute the $A(a,b)$ matrix (with CKA check); (ii) show ablating the top-$m$ aligned pairs collapses zero-shot accuracy far faster than (a) random pairs, (b) top-norm heads per layer, (c) top single-side heads; (iii) characterize the shared subspaces semantically (colors, animals — note and disclose the ImageNet bias of the statistics).
- **Procedure.** Accuracy-vs-$m$ curves for each ablation policy; per-pair semantic labels from E2; repeat on SigLIP and ≥2 model scales.
- **Novelty.** No published work matches components across the two encoders of a contrastive VLM, let alone causally validates the matching. This is the paper.

### E7 — Taxonomy: cross-modal vs modality-specific components
- **Motivation.** Embedding-level work already shows the final space splits into bimodal atoms that carry all cross-modal alignment and unimodal atoms that explain the modality gap (arXiv:2602.06218; same comet-shaped structure in audio-text CLAP, arXiv:2605.29628). Your hypothesis lifts this to the *component* level: heads with low $A$ (orthogonal cross-encoder subspaces) but high output variance are modality-specific.
- **Goal.** Quantify: scatter each head on the plane (cross-encoder alignment $\max_b A(a,b)$) × (dataset variance $\sum_k\lambda_k^a$) × (mean norm $\|\hat a\|$); test the predicted anti-correlation.
- **Causal test (E11 merged here).** Build a modality-stress benchmark: rendered letters/digits as images vs letter/digit strings as text (SVHN digits vs digit words; typographic-attack stimuli à la Goh et al.'s multimodal neurons). Prediction: ablating low-$A$/high-variance heads hurts modality-heavy tasks (OCR-like, texture) while leaving cross-modal semantic tasks intact — and vice versa for high-$A$ pairs.

### E8 — LLaVA: token-dependence of component importance
- **Motivation.** You found mean-ablating ATTN direct writes barely affects LLaVA, unlike CLIP — consistent with the fact that LLaVA consumes all patch tokens (penultimate layer), not the projected CLS, and that detailed information resides in patch tokens (arXiv:2411.05195). The second-order lens already shows MLP information reaches CLS *through* later attention — so for CLS, ablating attention also kills the MLP channel, while for patch tokens MLPs write directly.
- **Goal.** Formalize "attention gathers (CLS), MLPs refine (per-token)": decompose each patch-token stream $z_p = e_p + \sum a^{l,h}_p + \sum m^l_p$ and measure LLaVA benchmark deltas (VQA, POPE, MME) under ATTN-vs-MLP direct-write ablation at patch positions vs at CLS.
- **Controls (important).** High-norm register/artifact tokens (Darcet et al. 2023; test-time registers, arXiv:2506.08010) will dominate naive statistics — exclude or model them separately.
- **Framing.** This is the "transferability warning": CLS-based CLIP interpretability does not transfer to MLLMs. Valuable as a section, probably not a standalone paper.

### E9 — Training intervention: sparse component-pair alignment loss
- **Motivation.** If alignment pressure is uniform across pairs (§1.3), concentrating it should yield cleaner, more interpretable head-pair structure. Precedent that interpretability can be co-optimized with performance during CLIP training exists (Sparse CLIP, ICLR 2026 — but they sparsify the *output representation*; you sparsify the *component-pair interaction*, a different and more mechanistic knob).
- **Formalization.** Let $S_{ab}^{(ij)} = \hat a^{(i)}\cdot\hat b^{(j)}$. Replace the logit with a top-$k$ restricted sum, $\tilde s_{ij} = \frac{1}{Z}\sum_{(a,b)\in \mathrm{TopK}(|S^{(ij)}|)} S_{ab}^{(ij)}$ (straight-through or soft top-$k$ via temperature), **or** keep $s_{ij}$ and add a concentration regularizer
$$\mathcal{L} = \mathcal{L}_{\mathrm{CLIP}} + \eta \sum_{(a,b)\notin \mathrm{TopK}} \big|\mathbb{E}_{i}[S_{ab}^{(ii)}]\big| \quad\text{or}\quad \eta\sum_a H\big(\mathrm{softmax}_b |S_{ab}|\big)$$
(row-entropy penalty: each image component may align with few text components).
- **Procedure.** Small scale (CC3M/CC12M, ViT-B/32), $k\in\{1,2,4\}$; track the $A$ matrix **during training** (emergence dynamics — nobody has shown when cross-encoder head alignment appears); evaluate zero-shot + retrieval + interpretability metrics from E2/E6.
- **Judgment.** High-interest, medium-risk, compute-bound. Run after E1–E7 are solid.

### E10 — Initialization intervention
- **Motivation.** The cone effect: modality gap originates largely at random initialization (Liang et al. 2022), and even same-modality contrastive training creates gaps (arXiv:2405.18570). Hypothesis: pairing heads at init helps.
- **Procedure.** For chosen pairs $(h_I, h_T)$, draw a shared orthonormal basis $U \in \mathbb{R}^{d\times r}$ and initialize both output projections $W_O$ to write into $\mathrm{span}(U)$, keeping other pairs in (approximately) orthogonal complements; train as in E9; compare final $A$ matrix, gap size, and accuracy vs standard init.
- **Judgment.** Cheap ablation inside E9's runs; speculative as a headline, fine as a section.

---

## 4. Validation caveats to state in the paper (pre-empt reviewers)

1. **LN handling.** LN is rewritten as a per-sample affine map with forward-pass statistics $\mu_l,\sigma_l$ frozen and the multiplicative term $\gamma/\sqrt{\sigma_l^2+\epsilon}$ absorbed into $P$ (§0.2) — the decomposition is fully linear and exact under this convention (Gandelsman et al. 2024, Eq. 9). State the convention once; optionally show as a robustness check that recomputing statistics after ablation changes results negligibly.
2. **Loss ≠ linear in the sum.** Claim the *gradient-level* uniform-weighting statement (§1.3), not "the model minimizes the sum of cosines."
3. **PC identifiability.** Degenerate eigenvalues rotate PCs; always pair the $A$ metric with CKA/Procrustes; use bipartite matching, not diagonal $k\!\to\!k$.
4. **Sign asymmetry of PCs.** Label $\pm u_k$ separately (your black/white observation).
5. **Dataset bias.** ImageNet-derived PCs bias the semantic catalog (your "dogs" subspaces); replicate statistics on LAION subsets or COCO.
6. **Mean- vs zero-ablation.** Report both; mean-ablation is the default (removes variance, keeps operating point).
7. **Multiple simultaneous ablations don't add** (norm renormalization); report joint interventions empirically, not by summing attributions.

---

## 5. Positioning summary

| Piece | Closest prior work | Your delta |
|---|---|---|
| Image-side head/MLP localization | Gandelsman et al. 2024 | replication only — cite, don't claim |
| PCs of heads, stability, PC reweighting | ResiDual (arXiv:2411.00246) | bilateral (text encoder), concept-conditional ablation, causal destroy-controls |
| Text-encoder components | negation heads (2407.10488) only | full EOS decomposition + image-based labeling: **novel** |
| Loss/similarity decomposition into cross-encoder component pairs | none found | **novel — the paper's core** |
| Head-pair matching + causal collapse | none found | **novel** |
| Cross-modal vs modality-specific split | bimodal/unimodal atoms (2602.06218), COMET | lifted from embedding level to component level, with causal task tests |
| Waterbirds debiasing | Gandelsman; RepDecompose; LTC; prompt methods | PC-granularity; keep as application only |
| LLaVA token dependence | patch-token MLLM literature; second-order lens | ablation-profile flip CLS↔patch; transferability warning |
| Training for interpretability | Sparse CLIP (2601.20075) | pair-level (mechanistic) sparsity vs output sparsity; alignment emergence dynamics |

**Recommended paper 1:** §1 math + E1 + E2 + E3 + E6 (+ E7), with E4/E5 as applications. **Paper 2 (later):** E9 + E10 + emergence dynamics.
