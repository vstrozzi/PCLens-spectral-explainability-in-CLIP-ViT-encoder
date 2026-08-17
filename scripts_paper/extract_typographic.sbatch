#!/usr/bin/env bash
#SBATCH --job-name=typo_ext
#SBATCH --qos=pro_qos
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_typo_%j.out
#SBATCH --error=logs/extract_typo_%j.err
# Typographic attack: build 2,000 ImageNet photos with a wrong class name printed on them, then
# decompose them. The TEXT side is free - the 1,000 ImageNet class names are already extracted.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p logs
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k \
                 [RN50]=openai [RN101]=openai )

rm -rf datasets/typographic
"$PY" -m scripts_paper.build_typographic --n 2000 --seed 69

for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  echo "=== typographic $M"
  "$PY" -m utils.scripts.compute_activation_values --dataset typographic --model "$M" \
      --pretrained "${PRE[$M]}" --data_path ./datasets/ --output_dir "$OUT" \
      --samples_per_class None --batch_size 8 --seed 69 --device cpu
done
echo ALLDONE
