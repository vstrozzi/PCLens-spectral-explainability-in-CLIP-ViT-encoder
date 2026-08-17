#!/usr/bin/env bash
#SBATCH --job-name=insub
#SBATCH --qos=pro_qos
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --output=logs/insub_%j.out
#SBATCH --error=logs/insub_%j.err
# 10-class ImageNet specialisation, component AND PC level, 5 class draws x 3 calibration draws.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_imagenet_subset --models "$M" --device cpu \
      --n_cls 10 --subsets 5 --reps 3 --n_cal 2 --k 8 --eval_every 10
done
echo ALLDONE
