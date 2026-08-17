#!/usr/bin/env bash
#SBATCH --job-name=pclens_mnist
#SBATCH --qos=pro_qos
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --output=logs/extract_mnist_%j.out
#SBATCH --error=logs/extract_mnist_%j.err
set -euo pipefail
cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p logs
OUT=output_dir/activations_and_datasets_idxs_69
declare -A PRE=( [ViT-B-32]=laion2b_s34b_b79k [ViT-B-16]=laion2b_s34b_b88k \
                 [ViT-L-14]=laion2b_s32b_b82k [ViT-H-14]=laion2b_s32b_b79k )
# class-name file for the text tower (index == label)
printf 'zero\none\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\n' > "$OUT/MNIST_classnames.txt"
for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14; do
  "$PY" -m utils.scripts.compute_activation_values --dataset MNIST --model "$M" \
      --pretrained "${PRE[$M]}" --data_path ./datasets/ --output_dir "$OUT" \
      --samples_per_class 100 --tot_samples_per_class 892 --batch_size 8 --seed 69 --device cpu
  "$PY" -m utils.scripts.compute_activation_values_text --text_descriptions MNIST_classnames \
      --model "$M" --pretrained "${PRE[$M]}" --text_dir "$OUT" --output_dir "$OUT" \
      --sentences_per_class 1 --batch_size 256 --seed 69 --device cpu
done
echo "ALL DONE"
