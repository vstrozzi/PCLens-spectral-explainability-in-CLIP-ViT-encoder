#!/usr/bin/env bash
#SBATCH --job-name=hardneg
#SBATCH --qos=pro_qos
#SBATCH --mem=180GB
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/hardneg_%j.out
#SBATCH --error=logs/hardneg_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
# smaller models first so partial results land early
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_hardneg --models "$M" --tasks negation count order \
      --device cpu --reps 3 --n_cal 32 --max_steps 600 --max_items 2000
done
echo ALLDONE
