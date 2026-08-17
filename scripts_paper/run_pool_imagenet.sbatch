#!/usr/bin/env bash
# ImageNet with a much larger cap: at 1000 classes the pool was still improving when the
# 96-component cap truncated it, so the earlier ImageNet numbers were not a fair comparison.
#SBATCH --job-name=pclens_in
#SBATCH --qos=pro_qos
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/pool_imagenet_%j.out
#SBATCH --error=logs/pool_imagenet_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14; do
  "$PY" -m scripts_paper.exp_greedy_pool --models "$M" --tasks imagenet \
      --cal_reps 1 --max_size 100000 --seed_topk 48 --tag _inbig --device cpu
done
echo "ALL DONE"
