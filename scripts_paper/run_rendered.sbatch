#!/usr/bin/env bash
#SBATCH --job-name=rend_run
#SBATCH --qos=pro_qos
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/rend_run_%j.out
#SBATCH --error=logs/rend_run_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_counting --models "$M" --device cpu --reps 3 --n_cal 4 || true
done
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_typographic --models "$M" --device cpu --reps 3 --n_cal 64 || true
done
echo ALLDONE
