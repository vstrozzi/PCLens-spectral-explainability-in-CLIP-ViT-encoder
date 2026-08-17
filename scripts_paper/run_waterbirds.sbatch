#!/usr/bin/env bash
#SBATCH --job-name=pclens_wb4
#SBATCH --qos=pro_qos
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=logs/waterbirds4_%j.out
#SBATCH --error=logs/waterbirds4_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  "$PY" -m scripts_paper.exp_waterbirds --models "$M" --n_cal 8 --max_size 400 --reps 3 --device cpu
done
echo "ALL DONE"
