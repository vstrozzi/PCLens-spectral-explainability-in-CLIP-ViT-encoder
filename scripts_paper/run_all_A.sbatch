#!/usr/bin/env bash
#SBATCH --job-name=allA_wb_hn
#SBATCH --qos=pro_qos
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=24
#SBATCH --time=96:00:00
#SBATCH --output=logs/allA_%j.out
#SBATCH --error=logs/allA_%j.err
# Component-level: Waterbirds (all 4 selectors, to convergence) + per-group table + the three
# hard-negative caption tasks. Smallest models first so partial results land early.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
ORDER="ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14"

echo "############ Waterbirds, component level, to convergence"
for M in $ORDER; do
  "$P" -m scripts_paper.exp_waterbirds --models "$M" --n_cal 8 --reps 3 --device cpu
done

echo "############ Waterbirds per-group table (re-scores the saved pools, no search)"
"$P" -m scripts_paper.exp_waterbirds_table --device cpu

echo "############ Hard-negative caption pairs: negation / count / order"
for M in $ORDER; do
  "$P" -m scripts_paper.exp_hardneg --models "$M" --tasks negation count order \
      --device cpu --reps 3 --n_cal 32 --max_items 2000
done
echo ALLDONE
