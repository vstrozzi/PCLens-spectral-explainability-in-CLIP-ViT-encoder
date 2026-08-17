#!/usr/bin/env bash
#SBATCH --job-name=smoke
#SBATCH --qos=pro_qos
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
# Cheap correctness pass on the three NEW scripts before the long jobs reach them.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/home/virgilio/lib/minconda3/envs/MT/bin/python

echo "######## exp_hardneg (negation, ViT-B-32) - the FULL MODEL line is the finding"
"$P" -m scripts_paper.exp_hardneg --models ViT-B-32 --tasks negation --device cpu \
    --reps 1 --n_cal 32 --max_steps 4 --max_items 2000

echo "######## exp_waterbirds_table (ViT-B-32)"
"$P" -m scripts_paper.exp_waterbirds_table --models ViT-B-32 --device cpu --n_random 2

echo "######## exp_bias_pc (ViT-B-32)"
"$P" -m scripts_paper.exp_bias_pc --models ViT-B-32 --device cpu --k 8 --reps 1 --max_steps 4

echo "######## explain_pcs on the bias moves"
"$P" -m scripts_paper.explain_pcs --task bias --model ViT-B-32 --variant backward \
    --regime bias --rep 0 --top 3 --m 5 --device cpu
echo SMOKE_OK
