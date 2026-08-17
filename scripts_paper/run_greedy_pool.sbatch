#!/usr/bin/env bash
#SBATCH --job-name=pclens_pool
#SBATCH --qos=pro_qos

#SBATCH --mem=90GB
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/greedy_pool_%j.out
#SBATCH --error=logs/greedy_pool_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  "$PY" -m scripts_paper.exp_greedy_pool --models "$M" \
      --tasks fairface CIFAR100 caltech imagenet \
      --cal_reps 3 --max_size 96 --seed_topk 48 --retrieval --device cpu
done
echo "ALL DONE"
