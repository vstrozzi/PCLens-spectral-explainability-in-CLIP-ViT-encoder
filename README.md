# PCLens: Spectral Explainability and Concept-Level Interventions in CLIP ViT MSA

![Teaser](images/PC_Lens_Schema.png)

## Overview

This repository contains the implementation of **PCLens**, a mechanistic and spectral framework for interpreting and intervening in the latent space of the CLIP Vision Transformer (ViT).  
The project introduces a principled decomposition of multi-head self-attention (MSA) activations into principal components, enabling fine-grained semantic analysis, automated concept localization, and targeted concept-level interventions.

PCLens extends prior text-based explainability approaches by integrating both textual and visual semantics, revealing the internal structure of polysemantic attention heads and enabling direct manipulation of semantic directions in the residual stream.

By Virgilio Strozzi at ETH (Medical Data Science Lab, 2025. See full report in the repository.

## Abstract

Leveraging the shared image–text representation space, recent work proposes to interpret the role of multi-head self-attention (MSA) in the CLIP vision encoder by reconstructing the activation output of individual heads using text embeddings. However, this approach fails to capture the semantic structure of polysemantic heads, where no single dominant concept emerges.

We introduce **PCLens**, a mechanistic and spectral framework that decomposes the activation space of each MSA head into principal components (PCs), each bidirectionally interpreted with both text and image semantics. This method reveals highly specialized, semantically meaningful directions distributed across the ViT residual stream, encoding consistent semantic meaning in both MSA heads and the final CLIP embedding space.

We further propose **QuerySystem**, which automatically identifies PCs encoding a given visual–text concept across different heads, and **PCSelection**, which enables targeted manipulation of PCs to amplify or remove concepts without fine-tuning. Together, these tools uncover the latent structure of CLIP representations and support automatic concept-level interventions, such as mitigating spurious correlations.

Finally, we investigate the transferability of these insights to Large Vision–Language Models (LVLMs) using frozen CLIP encoders and show that the residual stream dynamics differ significantly between [CLS] tokens and patch tokens, suggesting new directions for multimodal interpretability research.

## Results and Key Findings

![Teaser](images/ViT_Head.png)

### Explainability and Latent Space Analysis
- PCLens provides **finer-grained interpretability** of CLIP ViT attention heads compared to text-only methods.
- Polysemantic heads are decomposed into multiple semantically coherent principal components.
- Principal components show **semantic alignment** between MSA head activations and the final CLIP embedding space.

### Automated Concept Localization
- QuerySystem identifies concept-related PCs across multiple heads and layers.
- Concepts are shown to be **distributed and redundant** across the ViT architecture.
- Only a small subset of PCs is often sufficient to represent a concept.

### Concept-Level Interventions
- PCSelection enables targeted removal or amplification of concepts in the residual stream.
- Applications include:
  - Removal of spurious correlations in zero-shot classification.
  - Concept enhancement and suppression without model fine-tuning.
  - Qualitative manipulation of semantic content in CLIP embeddings.

### Transfer to LVLMs (LLaVA)
- Interpretability insights from CLIP do not directly transfer to LVLM image tokens.
- MLP contributions dominate patch-token representations in LLaVA, unlike the [CLS]-token behavior in CLIP.
- This suggests that ViT encoders play different functional roles in CLIP and LVLM pipelines.

## Experimental Setup and Evaluation

### Models
Experiments are conducted on multiple CLIP ViT variants:
- ViT-B-16
- ViT-L-14
- ViT-H-14

### Datasets
- ImageNet (classification and segmentation)
- Waterbirds (spurious correlation analysis)
- Custom text datasets for concept probing

### Tasks
- Zero-shot classification reconstruction
- Semantic characterization of attention heads
- Concept localization across heads and layers
- Targeted concept intervention in the residual stream
- Transfer analysis to LLaVA

### Metrics
- Cosine similarity in CLIP embedding space
- Zero-shot accuracy recovery
- Qualitative semantic coherence of PCs
- Intervention effectiveness (spurious correlation mitigation)

## General
Information on how to run and play with our approach.

### Setup dependencies
Use the provided [`environment.yml`](environment.yml) file to create a Conda environment with all the dependencies:

```bash
conda env create -f environment.yml
conda activate MT
pip install -e llava-fork
```
### Download Dataset(s)
Download all datasets listed in `experiments_info.json` into `./datasets/`.

```bash
python -m utils.datasets.download_datasets
```

### Derive activation(s)
Derive activations for all the models listed in `experiments_info.json` over the datasets previously downloaded,
using a subset of data if required.

```bash
python -m utils.scripts.extract_activations
```


### Derive Explanations:
Run PCLens (`svd_data_approx`) or another algorithm over the gathered activations and get per model-dataset pair `.jsonl` explanations (each PC labelled by top texts **and** images) plus a `zero_shot_accuracy.txt` table of the reconstructions.
```bash
python -m utils.scripts.run_explanations
```

### Run PCLens


### Test
First run the Notebook prepare_data.ipynb to setup all the necessary data for the experiments.
Then use the Notebook playground.ipynb to play around with the different approaches.


### Preprocessing
To obtain the projected residual stream components for the ImageNet validation set, including the contributions from multi-head attentions and MLPs, please run one of the following instructions:

```bash
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k --data_path <PATH>
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path <PATH>
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k --data_path <PATH>
```

To obtain the precomputed text representations of the ImageNet classes, please run:
```bash
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k
```


### Convert text labels to representation 
To convert the text labels to CLIP text representations, please run:

```bash
python -m utils.scripts.compute_text_explanations --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path utils/text_descriptions/google_3498_english.txt
python -m utils.scripts.compute_text_explanations --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path utils/text_descriptions/top_1500_nouns_5_sentences_imagenet_bias_clean.txt
```

### ImageNet segmentation
To get the evaluation results, please run:

```bash
python compute_segmentations.py --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
python compute_segmentations.py --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
python compute_segmentations.py --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
```
Save the results with the `--save_img` flag.

### Spectral Decomposition

Explain the internal components of the CLIP-embeddings ViT-Encoder's for images using with text (ours: svd_data_approx, their: text_span)
```bash
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-H-14 --algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-L-14 --algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-B-16--algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
```

### Spatial decomposition
Please see a demo for the spatial decomposition of CLIP in `demo.ipynb`. 


### Nearest neighbors search
Please see the nearest neighbors search demo in `nns.ipynb`.
