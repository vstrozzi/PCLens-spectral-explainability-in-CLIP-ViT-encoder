#!/usr/bin/env bash
#SBATCH --job-name=allB_pc
#SBATCH --qos=pro_qos
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=24
#SBATCH --time=96:00:00
#SBATCH --output=logs/allB_%j.out
#SBATCH --error=logs/allB_%j.err
# PC-level: Waterbirds and the bias audit at (component, PC) granularity, all 4 selectors, to
# convergence, then the labelling pass that names the directions the search chose.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
ORDER="ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14"

echo "############ Bias audit at PC granularity"
for M in $ORDER; do
  "$P" -m scripts_paper.exp_bias_pc --models "$M" --device cpu --k 8 --reps 2
  for V in forward backward; do
    echo "======== labelled PC units, $M, $V"
    "$P" -m scripts_paper.explain_pcs --task bias --model "$M" --variant "$V" \
        --regime bias --rep 0 --top 15 --m 5 --device cpu || true
  done
done

echo "############ Waterbirds at PC granularity, all 4 selectors"
for M in $ORDER; do
  "$P" -m scripts_paper.exp_waterbirds_pc --models "$M" --device cpu --k 8 --n_cal 8 --reps 2
  for R in class group; do
    for V in forward backward; do
      echo "======== labelled PC units, $M, $V, $R"
      "$P" -m scripts_paper.explain_pcs --task waterbirds --model "$M" --variant "$V" \
          --regime "$R" --rep 0 --top 15 --m 5 --device cpu || true
    done
  done
done
echo ALLDONE
