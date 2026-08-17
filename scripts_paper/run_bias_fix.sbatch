#!/usr/bin/env bash
#SBATCH --job-name=bias_fix
#SBATCH --qos=pro_qos
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/bias_fix_%j.out
#SBATCH --error=logs/bias_fix_%j.err
# RN101 and ViT-H-14 hit the old 400-step cap, so their bias-audit numbers were truncated, not
# converged. The other four provably stopped on their own and are not re-run.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in RN101 ViT-H-14; do
  "$P" -m scripts_paper.exp_bias_audit --models "$M" --device cpu --reps 3
done
echo ALLDONE
