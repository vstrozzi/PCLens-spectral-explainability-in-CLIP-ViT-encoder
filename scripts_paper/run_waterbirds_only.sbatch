#!/usr/bin/env bash
#SBATCH --job-name=wb_only
#SBATCH --qos=pro_qos
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/wb_only_%j.out
#SBATCH --error=logs/wb_only_%j.err
# Waterbirds UNCAPPED (the earlier run truncated at 64/400 steps) + the per-group table.
# Split out from the hard-negative job because it needs far less memory and is the blocking result.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_waterbirds --models "$M" --n_cal 8 --reps 3 --device cpu
done
"$P" -m scripts_paper.exp_waterbirds_table --device cpu
"$P" -m scripts_paper.audit_convergence || true
echo ALLDONE
