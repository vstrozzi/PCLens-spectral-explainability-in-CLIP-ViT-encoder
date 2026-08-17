#!/usr/bin/env bash
# Context-aware greedy orderings for the 6.2 curves. O(n^2/2) loss evaluations for selection
# (on a 256-pair subsample), then the curves on the full 5,000. Smallest models first so
# results land early.
#SBATCH --job-name=pclens_gord
#SBATCH --qos=pro_qos
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/greedy_order_%j.out
#SBATCH --error=logs/greedy_order_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14 RN50 RN101 ViT-H-14; do
  "$PY" -m scripts_paper.exp_greedy_order --models "$M" --n_samples 256 --device cpu
done
echo "ALL DONE"
