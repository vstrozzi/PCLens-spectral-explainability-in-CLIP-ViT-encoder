#!/usr/bin/env bash
# Control for the MNIST result: same greedy pool, but the text tower built from the CLIP-paper
# prompt instead of bare digit words, to separate "the pool helps" from "the prompt was bad".
#SBATCH --job-name=mnistp
#SBATCH --qos=pro_qos
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/mnistp_%j.out
#SBATCH --error=logs/mnistp_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/virgilio/lib/minconda3/envs/MT/bin/python
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k )
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14; do
  "$P" -m utils.scripts.compute_activation_values_text --text_descriptions MNISTp_classnames \
     --model "$M" --pretrained "${PRE[$M]}" --text_dir "$OUT" --output_dir "$OUT" \
     --sentences_per_class 1 --batch_size 256 --seed 69 --device cpu
done
"$P" -m scripts_paper.exp_greedy_pool --models ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 \
   --tasks MNISTp --cal_reps 3 --max_size 200 --seed_topk 48 --tag _mnistp --device cpu
echo ALLDONE
