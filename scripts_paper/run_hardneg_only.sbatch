#!/usr/bin/env bash
#SBATCH --job-name=hn_only
#SBATCH --qos=pro_qos
#SBATCH --mem=45GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/hn_only_%j.out
#SBATCH --error=logs/hn_only_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_hardneg --models "$M" --tasks negation count order \
      --device cpu --reps 3 --n_cal 32 --max_items 2000
done
echo ALLDONE
