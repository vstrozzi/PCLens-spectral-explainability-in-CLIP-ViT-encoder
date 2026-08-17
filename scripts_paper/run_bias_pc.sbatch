#!/usr/bin/env bash
#SBATCH --job-name=bias_pc
#SBATCH --qos=pro_qos
#SBATCH --mem=180GB
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/bias_pc_%j.out
#SBATCH --error=logs/bias_pc_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-32 RN50 RN101 ViT-L-14; do
  "$P" -m scripts_paper.exp_bias_pc --models "$M" --device cpu --k 8 --reps 2 --max_steps 800
  # name the directions the search dropped, in both search directions
  for V in forward backward; do
    "$P" -m scripts_paper.explain_pcs --task bias --model "$M" --variant "$V" \
        --regime bias --rep 0 --top 15 --m 5 --device cpu || true
  done
done
echo ALLDONE
