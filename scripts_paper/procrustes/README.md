# Orthogonal-Procrustes alignment of CLIP encoders

Closed-form study of `W* = argmax_{WᵀW=I} Σ cossim(eᵢ, W fᵢ) = VUᵀ` (M = Σ fᵢeᵢᵀ = USVᵀ)
over ViT-B/16 & ViT-B/32, ImageNet (5000) and CIFAR100 (2000), image + text towers.
Reads the PRS activations under `output_dir/activations_and_datasets_idxs_69/`
(components are already `/‖output‖`, so component-sum = unit embedding and `ΣM = cosθ`).

## Run order (conda env `MT`; set BLAS threads first — see note)
```
python procrustes_align.py        # accuracy table (pos/neg/combos) + held-out split + shift baseline  -> procrustes_results.json
python procrustes_variants.py     # (A) hardest-neg repulsion  (B) per-sample-img vs per-class-txt fit -> procrustes_variants.json
python circuit_compute.py         # chunked, cgroup-safe: M(y,c)=A(c)·W·B(y)ᵀ reshaping              -> circuit_<model>_imagenet.npz
python fig_plots.py               # figures (SymLogNorm heatmaps)                                     -> figures/*.png
python build_report.py            # self-contained HTML report (figures inlined as data URIs)         -> report.html
```

## Findings
- **W_pos** attracts image→true class: mean true-class cos 0.27→0.72, but zero-shot helps
  **only transductively** (fit=eval). Held-out split: **hurts** (ViT-B/16 IN −0.126 test vs +0.158 train).
  Transfer ImageNet→CIFAR100 −0.077; CIFAR100→ImageNet −0.54. **Does not generalise.**
- **W_neg** = repel from a wrong class (min cossim / max L2 = −VUᵀ): zero-shot ≈0, and every product
  with it ≈0. `M_neg` is **stable-rank ≈1** — the random negative *is* the global text centroid
  (cos 0.9999). Variant (A): the **hardest** negative is degenerate too (rank ≈1) — CLIP's text cone
  is narrow, so any repulsion collapses to the modality-gap axis.
- **Image tower ≡ text tower**: same pairs ⇒ `W_text = W_imageᵀ`, `|Δacc| = 0.000` exactly.
  Variant (B): fitting per-sample image (N pts) vs per-class text (C pts) genuinely differs but only
  marginally — the coarser text fit over-fits slightly less, both still net-negative.
- **Circuit**: verified `Δcos = Σdimg = Σdtxt = ΣΔM`. W acts **~70% on the 13 MLP components**
  (enforces m0, m8–m11; destroys m12), attention heads ~untouched → it rearranges the class-independent
  DC/scaffold stream (arg-max-invisible), never the semantic heads.

## Env note
Set `OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_NUM_THREADS` **before** importing
numpy (the scripts do via `setdefault`) and keep float32 + chunking — the 12 GB login cgroup OOM-kills
otherwise. SVDs are 512×512 (CPU is fine).
