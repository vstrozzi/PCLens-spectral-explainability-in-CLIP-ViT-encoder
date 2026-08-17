#!/usr/bin/env bash
#SBATCH --job-name=typo_clean
#SBATCH --qos=pro_qos
#SBATCH --mem=45GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/typo_clean_%j.out
#SBATCH --error=logs/typo_clean_%j.err
# The missing control: the SAME 2,000 images with no word pasted on. Without it we cannot say how
# much of the post-intervention gap is the attack and how much is just hard images.
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k \
                 [RN50]=openai [RN101]=openai )
rm -rf datasets/typographic_clean
"$P" -m scripts_paper.build_typographic --n 2000 --seed 69 --clean
for M in ViT-B-32 ViT-B-16 RN50 RN101 ViT-L-14 ViT-H-14; do
  if [[ "$M" == RN* ]]; then S=utils.scripts.compute_activation_values_resnet; else S=utils.scripts.compute_activation_values; fi
  "$P" -m "$S" --dataset typographic_clean --model "$M" --pretrained "${PRE[$M]}" \
      --data_path ./datasets/ --output_dir "$OUT" --samples_per_class None \
      --tot_samples_per_class None --batch_size 8 --num_workers 4 --seed 69 --device cpu
done
echo ALLDONE
