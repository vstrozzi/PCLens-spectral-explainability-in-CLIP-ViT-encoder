#!/usr/bin/env bash
#SBATCH --job-name=pclens_63b
#SBATCH --qos=pro_qos
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=logs/id_layerown_%j.out
#SBATCH --error=logs/id_layerown_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p logs
for M in ViT-B-32 ViT-B-16 ViT-L-14 RN50 RN101 ViT-H-14; do
  "$PY" -m scripts_paper.exp_6_3 --models "$M" --device cpu
done
"$PY" -m scripts_paper.exp_plots --sections 6.3
echo "ALL DONE"
