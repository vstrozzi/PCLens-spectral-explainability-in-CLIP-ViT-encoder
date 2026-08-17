#!/usr/bin/env bash
# Submit the spectral-explanation analysis (run_explanations) to SLURM, non-interactively.
# All parameters are left at their defaults EXCEPT top-10 texts/images per PC.
#
#   sbatch scripts_paper/run_explanations.sbatch
#
# pro_qos caps the user at 2 GPUs / 200G, so we take 2 GPUs and run one model per GPU
# (--parallel uses every visible GPU). Add --partition=... if your cluster requires it.
#SBATCH --job-name=pclens_explain
#SBATCH --qos=pro_qos
#SBATCH --gpus=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_explanations_%j.out
#SBATCH --error=logs/run_explanations_%j.err
set -euo pipefail

cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
mkdir -p logs

# Non-interactive (no TTY): every prompt takes its default; the flags below set those defaults.
# Defaults => seed 69, all towers/models/datasets, mode=both (self + transfer tables), components=all,
# num_of_last_layers=all, svd_data_approx, textset=top_1500_nouns_5_sentences_imagenet_clean,
# image_set=imagenet (image_ref pool), n_explanations=auto, save_jsonl=true. Text-tower targets are
# scored on their own paired images; fairface classifiers auto-match the extracted attribute.
# Each accuracy cell is image->label / label->image.
"$PY" -m utils.scripts.run_explanations \
    --text_per_princ_comp 10 \
    --parallel \
    --rerun true \
    < /dev/null
