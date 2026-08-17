#!/usr/bin/env bash
# Paper experiments 6.2 (retrieval ablation), 6.3 (intrinsic dimensionality),
# 6.4 (linear approximation) on the COCO Karpathy-test activations, all 6 models,
# then every figure.
#
#   sbatch scripts_paper/run_6_2_6_4.sbatch            # everything
#   MODELS="ViT-B-32" sbatch scripts_paper/run_6_2_6_4.sbatch
#
#SBATCH --job-name=pclens_6x
#SBATCH --qos=pro_qos
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_6x_%j.out
#SBATCH --error=logs/run_6x_%j.err
set -euo pipefail

cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs

MODELS=${MODELS:-"ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101"}
SECTIONS=${SECTIONS:-"6.2 6.3 6.4"}

for M in $MODELS; do
  echo "================ $M ================"
  for S in $SECTIONS; do
    case $S in
      6.2) "$PY" -m scripts_paper.exp_6_2 --models "$M" --device cuda:0 ;;
      6.3) "$PY" -m scripts_paper.exp_6_3 --models "$M" --device cuda:0 ;;
      6.4) "$PY" -m scripts_paper.exp_6_4 --models "$M" --device cuda:0 ;;
    esac
  done
done

"$PY" -m scripts_paper.exp_plots --sections $SECTIONS --models $MODELS
echo "ALL DONE"
