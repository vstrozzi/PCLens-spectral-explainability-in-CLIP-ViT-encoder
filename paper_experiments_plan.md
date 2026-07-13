# Code Plan — Experiments Missing for the CVPR2026 Paper

Each item lists: what to add, where, CLI/function signatures, outputs, the paper
section/figure it feeds (red `\red{}` notes in the template), and compute notes.
Priority order = reviewer risk.

Convention for all new results: `results/paper/{model}/{experiment}.json|.npy`
(stats decoupled from plotting), figures regenerated only by `scripts_paper/paper_plots.py`.

---

## P1 — All-architecture dual-tower pipeline (feeds §6.1, §6.8, Fig. mean-ablation / interaction / H1 / paired-ablation)

Today the dual-tower analysis lives only in `playground_all.ipynb`, hard-wired to one
model per run. Promote it to scripts so it loops over B-32 / B-16 / L-14 / H-14.

**1a. `utils/scripts/shared_pairs_analysis.py`** — extract cells 70–89 of
`playground_all.ipynb` into importable functions + CLI:
- `interaction_matrix(attns_v, mlps_v, attns_t, mlps_t, labels, C) -> (C_mat, C_hat, W_mat)`
  (block-wise class means, never materialize `[N,K,d]`; code exists in cell 70).
- `head_pc_bundle(jsonl_path, k=50, var_cutoff=0.99)`, `metric_A(bundles_v, bundles_t)`,
  `metric_B(score_A, norm_v, norm_t)` (cells 79–80).
- `subspace_similarity(bundles_v, bundles_t, k=8, n_rotations=50)` + z-scores,
  `routing_test(C_hat, subspace_sim, n_shuffles=1000) -> spearman, z` (cells 74–75).
- `hungarian_match(...)`, `paired_mean_ablation(pairs, order='high2low|low2high|random', trials)` (cells 76–77, 82).
- CLI: `python -m utils.scripts.shared_pairs_analysis --model ViT-L-14 --seed 0 --pair_k 50 --cutoff 0.99 --subspace_k 8 --out results/paper/ViT-L-14/shared_pairs.json`

**1b. `utils/scripts/ablation_orderings.py`** — extract cell 43 (forward/backward/random
mean-ablation curves, both towers, MSA+MLP, 5 trials):
`--tower vision|text --kind attn|mlp --orders forward backward random --trials 5`.

**1c. Driver `scripts_paper/run_all_archs.sh`** — per model × seed:
1. `compute_activation_values` (vision) → 2. `compute_activation_values_text` →
3. `compute_text_embeddings` + `compute_classes_embeddings` →
4. `compute_text_explanations` + `compute_text_explanations_text` (`svd_data_approx`, `--num_of_last_layers 11`) →
5. `verify_decomposition_activations` (assert cos=1.0 both towers, log to JSON) →
6. `ablation_orderings` → 7. `shared_pairs_analysis`.

**Memory note (blocker for H-14):** vision activations `[50000, 32, 16, 1024]` fp32 ≈ 105 GB.
Add to `compute_activation_values.py`: `--last_layers_only k` (save only last k layers)
and `--dtype fp16`. For L-14/H-14 use 10 img/class (10k images) + last 11 layers →
`[10000, 11, 16, 1024]` fp16 ≈ 3.6 GB. Keep B-32/B-16 at 50k for comparability with current numbers.

---

## P2 — ResNet full run (feeds "Beyond transformers", appendix:resnet, §6.5)

`playground_resnet.ipynb` §A–§D are verified on a pilot (~5 img/class, RN50);
`compute_activation_values_resnet.py` exists.

- Scale the pilot: `samples_per_class=50` on RN50 (RN101 second), rerun §B–§D.
- **Add alive-fraction export**: dump `pi_lh` (`alive_frac`, cell 12) to
  `results/paper/RN50/alive_fraction.json` — the paper sentence "gates essentially never
  all-zero" currently has no citable number.
- §G (cross-encoder pairs): replace the notebook-local metrics with imports from
  `shared_pairs_analysis.py` (the RN50 text tower is a standard transformer, so the text
  side of P1 runs unchanged). This gives Metric A/B + interaction matrix + paired
  ablation for RN50 → completes the "every residual model, not only transformers" claim.
- Waterbirds on RN50: extend `bias_removal_test.py` to accept the resnet activation
  files (`--backbone resnet`); the PCSelection math is unchanged.

---

## P3 — Checkpoint emergence (feeds §6.9; abstract promises this)

`playground_checkpoints_vitb32.ipynb` stages 0–3 have outputs; stages 4–7 (the actual
tracking) have never run.

- Run Stage 1 extraction for the 7 tags (init, 2 early, n/4, n/2, 3n/4, final) into
  `output_dir/ck_<tag>/`. Heavy step; cache flag already exists (`RE_EXTRACT`).
- Add a consolidation cell/script `utils/scripts/checkpoint_trajectories.py`:
  per checkpoint compute Metric B for the 10 reference pairs (fixed at final ckpt),
  zero-shot accuracy, subspace similarity of matched pairs, top-10-head Jaccard vs final.
  Output `results/paper/checkpoints_vitb32.json`.
- Define the paper's emergence statistic explicitly: first checkpoint where a pair's
  Metric B ≥ 80% of its final value; plus the sudden-vs-continuous ratio already coded
  (largest single-step jump / total rise, cell 16).
- Figure: metric-B trajectories (10 lines) vs samples seen, accuracy overlay; small
  multiples of the architecture map per checkpoint (Stage 6).

---

## P4 — Random-PC control for QuerySystem (feeds §6.4(ii); cheap, do first)

- Add to `algorithms_text_explanations_funcs.py`:
  `select_random_pcs(data, n, layers='late'|'all', seed)` returning a set with the same
  shape as the QuerySystem selection (so the existing reconstruction path is reused).
- New `utils/scripts/query_random_control.py`: for ~20 concepts (colors, objects,
  abstract, biased), compare {QuerySystem top-n} vs {n random, late layers} vs
  {n random, uniform} with n = 30 and 75, 10 seeds each. Metrics per concept:
  (a) reconstruction cosine to the query embedding, (b) precision@k of
  concept-consistent retrievals (labels from ImageNet classes matching the concept),
  (c) zero-shot accuracy restricted to concept-related classes.
  Output CSV → grouped bar/violin figure.

---

## P5 — CIFAR-100 / FairFace characterization (feeds §6.5)

- `utils/datasets_constants/cifar_100_classes.py` (classnames + reuse OpenAI templates);
  add a `cifar100` branch next to the existing `cifar10` in
  `compute_activation_values.py` / `compute_classes_embeddings.py` (torchvision auto-download).
- `utils/datasets/fairface.py`: `FairFace(VisionDataset)` over the padded-images CSV
  (gender/race/age labels) + `fairface_classes.py`. Experiment mirrors Waterbirds:
  QuerySystem on "a man"/"a woman" (and race terms), PCSelection removal, report accuracy
  + demographic-gap metrics before/after. Reuses `bias_removal_test.py` with a
  `--dataset fairface` flag.

---

## P6 — Multi-seed small models (feeds §6.10; biggest compute)

- New `training/train_small_clip.py`: thin wrapper over `open_clip.training` — model
  `ViT-S-16`-scale (or B-32 with width/2), dataset CC3M (webdataset), 3–5 seeds,
  identical hyperparameters, ~30 epochs, batch 1024. Save 4–6 intermediate checkpoints
  per run (feeds P3-style analysis too).
- Post-hoc: run P1 pipeline per seed. Seed-invariance is tested on *structure*, not
  indices: distribution of Metric B (heavy-headedness), number of dominant pairs,
  and concept overlap of top pairs across seeds (match pairs by their PC0 text labels,
  e.g. Jaccard over top-20 label tokens).
- Fallback if compute is short: two public B-32 checkpoints trained on different data
  (laion400m vs laion2B) as a weaker "different training run" comparison.

---

## P7 — Loss variants (feeds §6.11; depends on P6 harness)

- In `training/`, add `--component_loss {none,orth}` and `--component_lambda λ`:
  per batch, hook the per-head [CLS] contributions of the last k layers (lightweight
  version of the PRS hook: only OV output + projection, no spatial), and penalize
  Σ_{a≠a'} |cossim(c_a, c_{a'})| within each tower (optionally PVE-weighted using a
  running covariance). Start with k=2 last layers to keep overhead <15%.
- Evaluate: zero-shot accuracy delta, sparsity of Metric A/B distribution,
  interventions efficacy (Waterbirds worst-group after PCSelection), mean top-PC text
  correlation as an interpretability proxy.

---

## P8 — Paper-quality plots (`scripts_paper/paper_plots.py`)

One function per paper figure, reading only `results/paper/`:
- Consistent rcParams: ≥8 pt fonts at column width, colorblind-safe palette, PDF export.
- Interaction-matrix + z-score heatmaps: label every 4th tick, annotate the top-5 cells,
  optional late-layers-only crop (current PNGs from the notebook are unreadable at column width).
- Multi-arch versions of mean-ablation / paired-ablation as arch × panel grids.
- Regenerate: `mean_ablation`, `interaction_matrix`, `h1_scatter`, `paired_ablation`,
  `ranked_pairs`, `architecture_map`, `accuracy_recovered`, waterbird grids, checkpoint
  trajectories, random-control bars.

---

## Suggested execution order

| Order | Item | Compute | Unblocks |
|---|---|---|---|
| 1 | P4 random control | hours, CPU/1 GPU | §6.4 control claim |
| 2 | P1 scripts + B-16 run | 1–2 days | multi-arch story |
| 3 | P2 RN50 full | 1 day | "not only transformers" |
| 4 | P3 checkpoints | 1–2 days (downloads heavy) | abstract claim |
| 5 | P1 L-14 / H-14 (fp16, last-11) | 2–3 days | full table |
| 6 | P5 CIFAR-100/FairFace | 1 day | §6.5 |
| 7 | P8 plots | 1 day | camera-ready figures |
| 8 | P6 multi-seed training | 1–2 weeks GPU | §6.10 |
| 9 | P7 loss variants | 1 week GPU | §6.11 |
