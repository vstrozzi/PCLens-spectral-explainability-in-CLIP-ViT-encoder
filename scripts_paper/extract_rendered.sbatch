#!/usr/bin/env bash
#SBATCH --job-name=rend_ext
#SBATCH --qos=pro_qos
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_rendered_%j.out
#SBATCH --error=logs/extract_rendered_%j.err
# Build and decompose the two rendered sets:
#   typographic  ImageNet photos with a WRONG class name printed on them
#   counting     1..10 instances of one (colour, shape), exact count ground truth
# ResNets take the separate ModifiedResNet decomposition path - the ViT script reaches into
# model.visual.transformer, which a ModifiedResNet does not have.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p logs
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k \
                 [RN50]=openai [RN101]=openai )

rm -rf datasets/counting
"$PY" -m scripts_paper.build_counting --n 2000 --seed 69
# the counting text side: "a photo of {n} {colour} {shape}s" for every (object, count) pair
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  "$PY" -m utils.scripts.compute_activation_values_text --text_descriptions counting_classnames \
      --model "$M" --pretrained "${PRE[$M]}" --text_dir "$OUT" --output_dir "$OUT" \
      --sentences_per_class 1 --batch_size 256 --seed 69 --device cpu
done

# typographic images already exist for the two ViT-B models; rebuild only if missing
if [ ! -f datasets/typographic/meta.json ]; then
  "$PY" -m scripts_paper.build_typographic --n 2000 --seed 69
fi

for DS in counting typographic; do
  for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
    if [ -f "$OUT/${DS}_attn_${M}_seed_69.npy" ]; then
      echo "=== $DS $M already done, skipping"; continue
    fi
    echo "=== $DS $M"
    if [[ "$M" == RN* ]]; then
      SCRIPT=utils.scripts.compute_activation_values_resnet
    else
      SCRIPT=utils.scripts.compute_activation_values
    fi
    "$PY" -m "$SCRIPT" --dataset "$DS" --model "$M" --pretrained "${PRE[$M]}" \
        --data_path ./datasets/ --output_dir "$OUT" \
        --samples_per_class None --tot_samples_per_class None \
        --batch_size 8 --num_workers 4 --seed 69 --device cpu
  done
done
echo ALLDONE
