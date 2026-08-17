#!/usr/bin/env bash
#SBATCH --job-name=concept
#SBATCH --qos=pro_qos
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/concept_%j.out
#SBATCH --error=logs/concept_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for L in comp pc; do
  echo "######## $L level, margin objective"
  "$P" -m scripts_paper.exp_concept --models ViT-B-32 --level $L --device cpu \
      --obj margin --pool_M $([ $L = comp ] && echo 60 || echo 200)
done
echo SMOKE_CONCEPT_OK
