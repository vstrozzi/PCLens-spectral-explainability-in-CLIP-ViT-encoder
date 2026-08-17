#!/usr/bin/env bash
#SBATCH --job-name=smoke3
#SBATCH --qos=pro_qos
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/smoke3_%j.out
#SBATCH --error=logs/smoke3_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
echo "######## Q1 exp_npairs"
"$P" -m scripts_paper.exp_npairs --models ViT-B-32 --device cpu --reps 1 --npairs 1 10
echo "######## Q3b exp_selectivity  (the baseline readouts are the thing to check)"
"$P" -m scripts_paper.exp_selectivity --models ViT-B-32 --device cpu --n_runs 1
echo "######## Q2 exp_stability"
"$P" -m scripts_paper.exp_stability --models ViT-B-32 --device cpu --n_runs 4
echo SMOKE3_OK
