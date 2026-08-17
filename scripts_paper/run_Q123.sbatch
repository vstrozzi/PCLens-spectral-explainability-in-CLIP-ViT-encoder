#!/usr/bin/env bash
#SBATCH --job-name=Q123
#SBATCH --qos=pro_qos
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/Q123_%j.out
#SBATCH --error=logs/Q123_%j.err
# The critical path for the minimal-example paper: Q1 how many examples, Q2 stability, Q3b
# selectivity. All on Waterbirds, all 6 models, all cheap.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  echo "######## Q3b selectivity $M"
  "$P" -m scripts_paper.exp_selectivity --models "$M" --device cpu --n_runs 3 || true
  echo "######## Q1 n-pairs $M"
  "$P" -m scripts_paper.exp_npairs --models "$M" --device cpu --reps 5 || true
  echo "######## Q2 stability $M (informative arm)"
  "$P" -m scripts_paper.exp_stability --models "$M" --device cpu --n_runs 20 --arm informative || true
  echo "######## Q2 stability $M (random arm)"
  "$P" -m scripts_paper.exp_stability --models "$M" --device cpu --n_runs 20 --arm random || true
done
echo ALLDONE
