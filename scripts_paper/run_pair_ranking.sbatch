#!/usr/bin/env bash
# CPU-only: the triple expectation is a stack of small GEMMs + logsumexps, and the GPU queue is
# the bottleneck, not the arithmetic. Needs real memory though (H-14/RN activations are ~11 GB),
# which is why this must be a SLURM job and not the login session.
#SBATCH --job-name=pclens_pair
#SBATCH --qos=pro_qos
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/pair_rank_%j.out
#SBATCH --error=logs/pair_rank_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p logs
# n_samples shrinks with model size: the triple expectation costs P*Q*n^2*d
"$PY" -m scripts_paper.exp_pair_ranking --models ViT-B-32 ViT-B-16 --n_samples 512 --device cpu
"$PY" -m scripts_paper.exp_pair_ranking --models ViT-L-14 RN101 --n_samples 384 --device cpu
"$PY" -m scripts_paper.exp_pair_ranking --models RN50 ViT-H-14 --n_samples 256 --device cpu
echo "ALL DONE"
