#!/usr/bin/env bash
#SBATCH --job-name=bias_audit
#SBATCH --qos=pro_qos
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --output=logs/bias_audit_%j.out
#SBATCH --error=logs/bias_audit_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
# RN50/RN101 are the OpenAI checkpoints - the ones the CLIP paper's bias section actually studied
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  "$P" -m scripts_paper.exp_bias_audit --models "$M" --n_cal 16 --max_steps 400 --reps 3 --device cpu
done
echo ALLDONE
