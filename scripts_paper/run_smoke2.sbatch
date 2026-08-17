#!/usr/bin/env bash
#SBATCH --job-name=smoke2
#SBATCH --qos=pro_qos
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/smoke2_%j.out
#SBATCH --error=logs/smoke2_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
"$P" -m scripts_paper.exp_imagenet_subset --models ViT-B-32 --device cpu \
    --subsets 1 --reps 1 --max_steps 6 --eval_every 2
echo SMOKE2_OK
