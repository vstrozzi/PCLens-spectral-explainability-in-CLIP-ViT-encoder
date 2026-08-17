#!/usr/bin/env bash
#SBATCH --job-name=visogen
#SBATCH --qos=pro_qos
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/visogender_%j.out
#SBATCH --error=logs/visogender_%j.err
# Dataset 3: occupation retained, perceived gender removed. 228 images, so extraction is fast.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k \
                 [RN50]=openai [RN101]=openai )
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  if [[ "$M" == RN* ]]; then S=utils.scripts.compute_activation_values_resnet; else S=utils.scripts.compute_activation_values; fi
  [ -f "$OUT/visogender_attn_${M}_seed_69.npy" ] || \
    "$P" -m "$S" --dataset visogender --model "$M" --pretrained "${PRE[$M]}" \
      --data_path ./datasets/ --output_dir "$OUT" --samples_per_class None \
      --tot_samples_per_class None --batch_size 8 --num_workers 4 --seed 69 --device cpu
  "$P" -m utils.scripts.compute_activation_values_text --text_descriptions visogender_classnames \
      --model "$M" --pretrained "${PRE[$M]}" --text_dir "$OUT" --output_dir "$OUT" \
      --sentences_per_class 1 --batch_size 256 --seed 69 --device cpu
done
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  "$P" -m scripts_paper.exp_visogender --models "$M" --device cpu --reps 5 || true
done
echo ALLDONE
