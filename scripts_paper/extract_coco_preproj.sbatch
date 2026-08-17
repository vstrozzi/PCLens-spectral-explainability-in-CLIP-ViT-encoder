#!/usr/bin/env bash
# PRE-PROJECTION COCO activations, for section 6.3's "before vs after the output projection"
# question. Written to a SEPARATE directory: the file names carry no projection tag, so this
# must never share output_dir with the projected run.
#
# In this space the components live in the encoder's own width (e.g. 768 for ViT-B-32 vision,
# 512 for its text tower) and sum to the pre-projection residual, NOT to the shared embedding -
# so it is only used for dimensionality statistics, never for retrieval.
#
#   sbatch scripts_paper/extract_coco_preproj.sbatch
#
#SBATCH --job-name=coco_preproj
#SBATCH --qos=pro_qos
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_coco_pre_%j.out
#SBATCH --error=logs/extract_coco_pre_%j.err
set -euo pipefail

cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
OUT=output_dir/results_paper/preproj_acts
SEED=69
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs "$OUT"

declare -A PRETRAINED=(
  [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k
  [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k
)

for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14; do
  P=${PRETRAINED[$M]}
  if [[ ! -f "$OUT/coco_attn_${M}_seed_${SEED}.npy" ]]; then
    "$PY" -m utils.scripts.compute_activation_values \
      --dataset coco --model "$M" --pretrained "$P" --vision_proj False \
      --samples_per_class None --tot_samples_per_class None \
      --batch_size 8 --num_workers 4 --seed $SEED --output_dir "$OUT"
  fi
  if [[ ! -f "$OUT/coco_karpathy_test_attn_text_${M}_seed_${SEED}.npy" ]]; then
    "$PY" -m utils.scripts.compute_activation_values_text \
      --text_descriptions coco_karpathy_test --model "$M" --pretrained "$P" \
      --text_proj False --seed $SEED --output_dir "$OUT"
  fi
done

for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14; do
  "$PY" -m scripts_paper.exp_6_3 --models "$M" --device cuda:0 --act_dir "$OUT" --space pre
done
echo "ALL DONE"
