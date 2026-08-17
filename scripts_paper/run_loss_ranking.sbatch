#!/usr/bin/env bash
# Loss-restricted ranking: cheap scoring, but the evaluation sweep is many full 5000x5000
# retrievals + InfoNCE, which is what actually needs the GPU.
#SBATCH --job-name=pclens_loss
#SBATCH --qos=pro_qos

#SBATCH --mem=90GB
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/loss_rank_%j.out
#SBATCH --error=logs/loss_rank_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14 RN50 RN101 ViT-H-14; do
  "$PY" -m scripts_paper.exp_loss_ranking --models "$M" --n_samples 256 --device cpu
done
echo "ALL DONE"
