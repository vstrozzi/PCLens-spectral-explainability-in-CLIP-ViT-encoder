#!/usr/bin/env bash
#SBATCH --job-name=wb_pc
#SBATCH --qos=pro_qos
#SBATCH --mem=45GB
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/wb_pc_%j.out
#SBATCH --error=logs/wb_pc_%j.err
# PC-level Waterbirds ONLY - the interpretation substrate for Q3. The PC-level BIAS audit that used
# to share this job is not on the new critical path and was dropped; its ViT-B-32/B-16 results are
# already on disk.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
for M in ViT-B-16 RN50 RN101 ViT-H-14; do        # B-32 and L-14 already done
  "$P" -m scripts_paper.exp_waterbirds_pc --models "$M" --device cpu --k 8 --n_cal 8 --reps 2
done
echo ALLDONE
