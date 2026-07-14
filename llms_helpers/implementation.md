# Implementation Overview — PCLens

> Spectral Explainability and Concept-Level Interventions in CLIP ViT Multi-Head Self-Attention (MSA)

This document explains the **goal** of the project, the **file structure** of the
repository, and **what each file contains**. It is meant as a map for anyone who
needs to understand or extend the codebase.

---

## 1. Goal

CLIP image encoders (ViTs) produce a single image embedding, but *how* that
embedding is built from the model's internal components is opaque. This project
(**PCLens**) opens up that black box.

The core idea builds on the **residual-stream decomposition** of a CLIP ViT: the
final image embedding can be written as a **sum of the direct contributions** of
every multi-head self-attention (MSA) head and every MLP across all layers, each
projected into the shared CLIP text–image space:

```
image_embedding ≈ Σ_layers Σ_heads  attn[layer, head]  +  Σ_layers  mlp[layer]
```

(This is the "Projected Residual Stream", PRS, from Gandelsman et al.
*clip_text_span* — the code is adapted from that MIT-licensed project.)

On top of this decomposition PCLens adds its own contribution:

1. **Spectral / Concept decomposition (`svd_data_approx`).** For each
   `(layer, head)`, collect its per-image contribution vectors over a dataset,
   mean-center them, and run an **SVD**. The top singular directions (principal
   components) are the recurring "concepts" that head writes into the residual
   stream. Each principal direction is then *labeled* with natural language by
   finding the text descriptions (from a large text dataset) whose CLIP embeddings
   align most/least with it. This yields a **human-readable, per-head, per-PC
   text explanation** of what each attention head represents.

2. **Concept-level interventions / reconstruction.** Because the embedding is an
   explicit sum of head contributions (and each head an explicit sum of PCs), you
   can **mean-ablate**, **keep**, **amplify**, or **invert** specific heads or
   principal components and re-assemble the embedding. This is used to:
   - measure *completeness* (how much zero-shot accuracy a few PCs recover),
   - remove **spurious/bias** directions (e.g. Waterbirds background bias),
   - perform **zero-shot semantic segmentation** by scoring per-patch
     contributions against a text query.

3. **Comparison to TextSpan.** The repo benchmarks PCLens against the prior
   "TextSpan" greedy algorithm; `scripts_thesis/` holds the accuracy-recovered
   results that show PCLens recovers more accuracy with fewer text explanations.

The whole pipeline runs on CLIP variants (`ViT-B-16/32`, `ViT-L-14`, `ViT-H-14`,
OpenCLIP/LAION weights) and also integrates with **LLaVA** (CLIP as the vision
tower of a VLM) to test whether the same decomposition affects downstream
language generation.

---

## 2. Top-level workflow

The intended usage path (see `README.md`):

1. **`prepare_data.ipynb`** — precompute everything needed:
   - per-head/per-MLP activations for a dataset (`compute_activation_values`),
   - final image embeddings (`compute_images_embedding`),
   - class-label text embeddings for zero-shot (`compute_classes_embeddings`),
   - text-dataset embeddings (`compute_text_embeddings`),
   - text explanations per head (`compute_text_explanations`).
2. **`playground.ipynb`** — interactively explore head/PC text explanations,
   reconstruct embeddings, run ablations, NN search, completeness curves.
3. **`zero_shot_segment.ipynb`** — spatial decomposition → segmentation demo.
4. **`playground_llava.ipynb`** — apply the decomposition inside LLaVA.

Precomputed artifacts are written to `output_dir/` as `.npy` / `.jsonl` files
keyed by `{dataset}_{type}_{model}_seed_{seed}`.

---

## 3. File structure & contents

### 3.1 Root

| File | Contents |
|------|----------|
| `README.md` | User-facing instructions: setup, dataset download, and CLI invocations for every preprocessing/evaluation script. |
| `experiments_info.json` | **Central experiment hub.** Single source of truth for the run: `models_config` (each model's `name`, `slug`, `pretrained` tag, `type` = vit/resnet), `datsets_config` (each dataset's `element_per_class`, `num_classes`, pipeline `dataset_arg`, `data_path`, presence `folder`, download `size`) and the default `seed`. Read by `utils.datasets.download_datasets` (sizes/catalogue) and `utils.scripts.extract_activations` (models/datasets/pretrained/seed) so those scripts hold no hardcoded model or dataset tables. |
| `CLAUDE.md` | Generic LLM coding-assistant behavioral guidelines (not project logic). |
| `environment.yml` | Conda env `MT`: Python 3.10, PyTorch 2.1.2 / cu11.7, plus LLaVA-pinned deps (transformers 4.37.2, timm, deepspeed, gradio, etc.). |
| `LICENSE.txt` | License. Much code is adapted from `clip_text_span` (MIT, Yossi Gandelsman) and OpenAI CLIP (MIT). |
| `playground.ipynb` | **Main interactive notebook.** Loads a model + precomputed data, prints top PCs/text explanations per head, reconstructs embeddings, runs mean-ablation completeness tests, nearest-neighbor search, and zero-shot accuracy checks. 54 cells. |
| `prepare_data.ipynb` | **Data-prep notebook.** Runs the `utils.scripts.*` CLI tools (activations, image embeddings, class embeddings, text embeddings, text explanations) for a chosen dataset/model. Requires ≥32 GB RAM. |
| `zero_shot_segment.ipynb` | **Spatial decomposition demo.** Uses `attn_method='head'` (spatial) to get per-patch contributions and produces zero-shot segmentation heatmaps from a text query. |
| `playground_llava.ipynb` | **LLaVA integration.** Loads LLaVA, swaps/inspects its CLIP vision tower, uses an `InversionNet` to map between embedding spaces, and tests how concept interventions affect generated text. |

### 3.2 `images/`
Sample input images (animals, people, hearts, walls, etc.) used by the notebooks,
plus output figures (`teaser.png`, `layer_11_head_*.png` head visualizations,
`eq_res_stream.png` residual-stream equation, a results PDF).

### 3.3 `datasets/`
Local dataset archives (`cifar-10-python.tar.gz`, `imagenet.zip`,
`waterbird_complete95_forest2water2.tar.gz`). ImageNet/Waterbirds are downloaded
per README; CIFAR is auto-downloaded by torchvision.

### 3.4 `output_dir/`
Precomputed artifacts (chunked then concatenated `.npy`, plus `.jsonl`
explanations). Naming convention:
`{dataset}_{attn|mlp|cls_attn|labels|classifier}_{model}_seed_{seed}.npy` and
`{dataset}_completeness_{textset}_{model}_algo_{algorithm}_seed_{seed}.jsonl`.
Runs launched via `utils.scripts.extract_activations` nest their outputs under
`activations_and_datasets_idxs_{seed}/`, adding per-dataset `{dataset_arg}_idx_to_class_seed_{seed}.{npy,json}`
subset maps, `{dataset_arg}_classnames.txt` (text-tower input), and a `parameters.jsonl` run manifest.

### 3.5 `results/` and `scripts_thesis/`
- `results/cifar10_acc_LLAVA_CLIP.txt` — recorded accuracy numbers.
- `scripts_thesis/plot_accuracy_recovered.py` — seaborn heatmap comparing
  **TextSpan vs PCLens** accuracy recovered vs. number of text explanations,
  across the four ViT sizes (`accuracy_recovered.png`).

---

## 4. `utils/` — the library

### 4.1 `utils/models/` — CLIP/OpenCLIP model + hooks
Adapted from OpenCLIP / `clip_text_span`.

| File | Contents |
|------|----------|
| `model.py` | `CLIP` module definitions, config dataclasses (`CLIPVisionCfg`), state-dict conversion, pos-embed resizing. |
| `transformer.py` | `VisionTransformer`, `TextTransformer`, `Attention`, `LayerNorm`, `QuickGELU`. The attention forward exposes per-head outputs via `attn_method` (`head`, `head_no_spatial`). |
| `modified_resnet.py` | ResNet CLIP backbone variant. |
| `timm_model.py` | Wrapper for timm vision backbones. |
| `factory.py` | `create_model_and_transforms`, `get_tokenizer`; reads `model_configs/*.json`, downloads pretrained weights. |
| `pretrained.py` | Registry of pretrained tags / HF download helpers. |
| `openai_models.py` | Loading original OpenAI CLIP checkpoints. |
| `openai_templates.py` | `OPENAI_IMAGENET_TEMPLATES` prompt templates for zero-shot. |
| `hook.py` | **`HookManager`** — registers named forward hooks (supports `*` wildcards over resblocks and "forks"); the mechanism that lets PRS read intermediate activations without modifying forward code. |
| `prs_hook.py` | **`PRSLogger` / `hook_prs_logger`** — the heart of the decomposition. Hooks each block's attention output and MLP output, applies the correct share of the final `ln_post` (mean-centering, scaling, bias) and the visual projection so every head/MLP contribution lands in CLIP space. Supports spatial (per-patch) vs non-spatial (CLS-only) and projected vs raw output. |

### 4.2 `utils/model_configs/`
JSON architecture configs for every supported model (`ViT-B-16.json`,
`ViT-L-14.json`, `ViT-H-14.json`, EVA, CoCa, RoBERTa-CLIP, Swin, etc.). Consumed
by `factory.py`.

### 4.3 `utils/scripts/` — the pipeline (run as `python -m utils.scripts.<name>`)

| File | Role |
|------|------|
| `extract_activations.py` | **Interactive one-command driver** (counterpart of `download_datasets`). Reads `experiments_info.json`, asks which models / downloaded datasets / seed / sentence sets to run and the `compute_activation_values` knobs (L2-projection, quantization, layers, batch sizes, FairFace label), then delegates per `(model, dataset)` to the scripts below: image tower (`compute_activation_values` for ViT, `compute_activation_values_resnet` for RN), text tower (`compute_activation_values_text` over each dataset's class names + any chosen sentence sets), and `verify_decomposition_activations`. Runs models sequentially or one-per-GPU in parallel. Writes everything to `output_dir/activations_and_datasets_idxs_{seed}/`, including per-dataset `{dataset_arg}_idx_to_class_seed_{seed}.{npy,json}` subset maps (recover the exact decomposed images + classes) and a `parameters.jsonl` describing the run. |
| `compute_activation_values.py` | For a dataset, run the model with the PRS hook and save per-head attention contributions `[b, l, h, d]`, MLP contributions `[b, l+1, d]`, CLS→CLS attention, and labels. Writes RAM-friendly chunk files then concatenates. |
| `compute_activation_values_resnet.py` | ResNet (RN50/RN101) counterpart: exact per-input layer×head decomposition (`utils.models.resnet_prs`) saved in the same `[N,L+1,H,d]` / `[N,1,d]` layout so the ViT tooling is reused (TF32 off). `--normalize` (default True) divides every component by `‖visual(image)‖` so they sum to the L2-normalized output like the ViT/text pipeline; `--normalize False` keeps the raw exact decomposition summing to `visual(image)`. `--vision_proj False` saves the **pre-projection** per-head value stream (dim `C//H`, e.g. 64) with the attnpool output projection `W_o` factored out — `W_o`/`b_o` (`ModifiedResNet.get_output_projection()`, state-dict `visual.attnpool.c_proj.{weight,bias}`, `[out_dim, 2048]`/`[out_dim]`) are dumped to `{dataset}_out_proj_{model}_seed_{seed}.npz` so the CLIP embedding stays recoverable. |
| `compute_activation_values_text.py` | **Text-tower** counterpart of `compute_activation_values`: decomposes the CLIP text encoder over a `.txt` sentence set into per-head/per-MLP contributions to the EOS token (`_text` tag). Components sum to the L2-normalized text embedding (same `/‖·‖` step as the ViT logger). |
| `verify_decomposition_activations.py` | Sanity-checks BOTH encoders by re-running `n_samples` real forwards and asserting `sum(components) == encode_*(x)`. `--image_normalize` (default True) picks whether the image side is compared to the normalized or raw output, matching how the activations were saved; ResNet also forces TF32 off. |
| `compute_mlps_attns_hidden_mean.py` | Variant that computes the **mean** activation per head/MLP on the fly (used for mean-ablation baselines). |
| `compute_images_embedding.py` | Reassemble the saved activations into final image embeddings. |
| `compute_classes_embeddings.py` | `zero_shot_classifier`: build the `(N, C)` class-projection matrix from class names + templates for zero-shot classification. |
| `compute_text_embeddings.py` | Embed an arbitrary text dataset (`text_descriptions/*.txt`) into CLIP space — the candidate "concept labels". |
| `run_explanations.py` | **Interactive analysis driver** (counterpart of `extract_activations`). Discovers which activations exist under `output_dir` (seeds/models/datasets/towers), asks what to analyse (models, datasets, `--components` attn/mlp/all, modality = image vs text tower, algorithm, candidate-text set, `n_explanations` = PCs/unit or `auto`=99% variance, layers, device / one-per-GPU), ensures prerequisites (`compute_classes_embeddings`, `compute_text_embeddings`), then delegates each run to `compute_text_explanations` (image) / `compute_text_explanations_text` (text). Writes `output_dir/current_analyzed_dir/{algorithm}_{n_explanations}/{dataset}_{model}_{component}_{modality}.jsonl` + one `zero_shot_accuracy.txt` table (`model / pc / image / text` reconstruction accuracies). |
| `compute_text_explanations.py` | **Main PCLens driver.** Mean-ablates early layers, then for each unit in the last `num_of_last_layers` (`--components` attn heads and/or mlp layers) runs the chosen `algorithm` (default `svd_data_approx`) to decompose it and label its PCs; each PC records both its top/bottom **texts** and **images** (index+class, via the idx→class map) for visualization. Final line reports four zero-shot accuracies: `full` (real model) ≥ `pc` (rank-r subspace) ≥ `image`/`text` (top-1 example-span reconstructions, mean restored). `--out_file` sets the path. |
| `algorithms_text_explanations.py` | **`svd_data_approx`** — the PCLens algorithm: mean-center a head's activations, SVD, keep PCs up to 99% variance (capped), project the text dataset into PC space, pick top/bottom texts per PC (positive/negative poles), and least-squares reconstruct the head from the selected text span. Returns reconstruction + JSON metadata (`vh`, `project_matrix`, strengths, texts). |
| `algorithms_text_explanations_prev.py` | Earlier/alternative algorithms (incl. the baseline **TextSpan** greedy span), imported via `*` so they're selectable by `--algorithm`. |
| `algorithms_text_explanations_funcs.py` | **Large analysis/visualization toolkit (~2200 lines).** Loads `.jsonl` explanations (`get_data*`), prints PC text tables, plots singular-value curves, and—critically—**reconstructs embeddings under interventions**: `reconstruct_all_embeddings_mean_ablation_pcs/heads` (keep/ablate/amplify chosen heads & PCs), `reconstruct_embeddings`, NN-search dataset builders (`create_dbs`, `visualize_dbs`), and per-PC visualization. Backs the notebooks. |
| `compute_components_per_image.py` | Per-image (rather than dataset-aggregate) component decomposition. |
| `compute_components_for_topic.py` | Find the heads/PCs most aligned with a given **topic** text embedding (top-k cosine search across heads). |
| `compute_segmentations.py` | Zero-shot **segmentation** evaluation: spatial PRS → per-patch scores vs. ground-truth masks; reports pixel accuracy, IoU, AP (`--save_img` to dump overlays). |
| `bias_removal_test.py` | **Bias-removal experiment** (e.g. Waterbirds): identify and ablate spurious-correlation directions, measure accuracy change. |
| `utils_llava.py` | LLaVA helpers: `InversionNet` (768→1024 MLP to bridge embedding spaces) and getters for the CLIP visual projection / post-LayerNorm params. |

### 4.4 `utils/datasets/`
| File | Contents |
|------|----------|
| `binary_waterbirds.py` | `BinaryWaterbirds` VisionDataset (waterbird vs landbird, with background-bias metadata). |
| `dataset_helpers.py` | `dataset_to_dataloader` / `dataset_subset` — build class-balanced subsets (`samples_per_class` out of `tot_samples_per_class`) for reproducible experiments. |

### 4.5 `utils/datasets_constants/`
Class-name lists and dataset stats: `imagenet_classes.py`, `cifar_10_classes.py`,
`cub_classes.py` (CUB + `waterbird_classes`), and `constants.py`
(`OPENAI_DATASET_MEAN/STD` normalization constants).

### 4.6 `utils/text_descriptions/`
The **text datasets** used as candidate concept labels, plus their cleaned variants:
`top_1500_nouns_5_sentences_imagenet*.txt`, `google_3498_english.txt`,
`laion.txt`, `mscoco.txt`, `visual_descriptions.txt`, `bias_words.txt`,
`topics.txt`, etc. Embedded by `compute_text_embeddings.py`.

### 4.7 `utils/generate_text_dataset/`
Tooling that *built* the text datasets above:
- `scrape_dictionary.py` — scrape a top-1500-nouns list from the web.
- `generate_text_dataset.py` — call the OpenAI API to expand each noun into 5
  simple CLIP-friendly descriptions.
- `clean_text_dataset.py` — dedup/sort/clean the generated text files.

### 4.8 `utils/misc/`
| File | Contents |
|------|----------|
| `misc.py` | `accuracy`, `accuracy_correct`, batch-norm freezing, `to_2tuple`, small helpers. |
| `transform.py` | Image preprocessing transforms (`image_transform`, `AugmentationCfg`). |
| `visualization.py` | `image_grid`, `visualization_preprocess` and heatmap/overlay helpers for the notebooks. |
| `tokenizer.py` | CLIP BPE tokenizer + HF tokenizer wrappers. |
| `vocab/bpe_simple_vocab_16e6.txt.gz` | BPE merge table for the tokenizer. |
| `imagenet_segmentation.py` | `ImagenetSegmentation` dataset loader (gtsegs mat) for the segmentation eval. |
| `segmentation_utils.py` | Metrics: `batch_pix_accuracy`, `batch_intersection_union`, `get_ap_scores`, `Saver`. |

---

## 5. Key data shapes & conventions

- **Attentions** saved as `[b, l, h, d]` (batch, layers, heads, CLIP dim) — CLS
  token only in the non-spatial path; `[b, l, n, h, d]` with patches `n` in the
  spatial path used for segmentation.
- **MLPs** saved as `[b, l+1, d]` (the extra `+1` is the `ln_pre` embedding
  contribution).
- A head's contribution is the sum over its PCs; the full embedding is the sum of
  all heads + all MLPs (the residual-stream identity in §1).
- **Mean-ablation** of a head = replace its per-image contribution by the dataset
  mean, removing its image-specific information while preserving the bias term.
- Explanation `.jsonl`: one line per `(layer, head)` with `vh` (PC basis),
  `project_matrix`, singular strengths (`strength_abs`/`strength_rel`), and the
  positive/negative text labels per PC; a final `head=-1` line holds the overall
  CLIP-output decomposition and recovered accuracy.

---

## 6. Provenance

Substantial portions (model code, hooks, PRS logger, several scripts) are
**adapted from `clip_text_span`** (Gandelsman et al., MIT License) and OpenAI
CLIP / OpenCLIP. The original contribution of this thesis is the
`svd_data_approx` spectral decomposition, the concept-level
intervention/reconstruction machinery in `algorithms_text_explanations_funcs.py`,
the bias-removal and LLaVA experiments, and the PCLens-vs-TextSpan evaluation.
