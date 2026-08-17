#!/usr/bin/env bash
# Re-run 6.4 with the extended rank range (k up to 256) for every model, then redraw all figures.
#
#   sbatch scripts_paper/rerun_6_4_plots.sbatch
#
#SBATCH --job-name=pclens_64
#SBATCH --qos=pro_qos
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/rerun_6_4_%j.out
#SBATCH --error=logs/rerun_6_4_%j.err
set -euo pipefail

cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs

for M in ViT-B-32 ViT-B-16 ViT-L-14 RN50 RN101; do
  "$PY" -m scripts_paper.exp_6_4 --models "$M" --device cuda:0
done

"$PY" -m scripts_paper.exp_plots --sections 6.2 6.3 6.4
"$PY" -m scripts_paper.gen_notebook_6_2_6_4
echo "ALL DONE"
