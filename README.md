# PCLens: Spectral Explainability and Concept-Level Interventions in CLIP ViT MSA

![Teaser](images/teaser.png)

## General
Information on how to run and play with our approach.

### Setup dependencies
Use the provided [`environment.yml`](environment.yml) file to create a Conda environment with all the dependenctios:

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
