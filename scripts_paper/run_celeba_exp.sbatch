#!/usr/bin/env bash
#SBATCH --job-name=celeba_exp
#SBATCH --qos=pro_qos
#SBATCH --mem=35GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/celeba_exp_%j.out
#SBATCH --error=logs/celeba_exp_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_celeba --models "$M" --device cpu --reps 5 --n_cal 1 || true
done
"$P" -m scripts_paper.audit_convergence || true
echo ALLDONE
