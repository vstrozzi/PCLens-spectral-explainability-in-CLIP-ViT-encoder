#!/usr/bin/env bash
# COCO Karpathy-test activation extraction for paper section 6.2 (retrieval ablation).
# Both towers, all 6 models, sequential on one GPU (pro_qos caps 2 GPUs total).
#
#   sbatch scripts_paper/extract_coco.sbatch
#
#SBATCH --job-name=coco_extract
#SBATCH --qos=pro_qos
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_coco_%j.out
#SBATCH --error=logs/extract_coco_%j.err
set -euo pipefail

cd /home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder
PY=/home/virgilio/lib/minconda3/envs/MT/bin/python
OUT=output_dir/activations_and_datasets_idxs_69
SEED=69
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs

declare -A PRETRAINED=(
  [ViT-B-32]=laion2b_s34b_b79k
  [ViT-B-16]=laion2b_s34b_b88k
  [ViT-L-14]=laion2b_s32b_b82k
  [ViT-H-14]=laion2b_s32b_b79k
  [RN50]=openai
  [RN101]=openai
)

for M in ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14 RN50 RN101; do
  P=${PRETRAINED[$M]}

  # ---- image tower (skip if final file exists) ----
  if [[ ! -f "$OUT/coco_attn_${M}_seed_${SEED}.npy" ]]; then
    if [[ $M == RN* ]]; then
      "$PY" -m utils.scripts.compute_activation_values_resnet \
        --dataset coco --model "$M" --pretrained "$P" \
        --samples_per_class None --tot_samples_per_class None \
        --batch_size 8 --num_workers 4 --seed $SEED --output_dir "$OUT"
    else
      "$PY" -m utils.scripts.compute_activation_values \
        --dataset coco --model "$M" --pretrained "$P" \
        --samples_per_class None --tot_samples_per_class None \
        --batch_size 8 --num_workers 4 --seed $SEED --output_dir "$OUT"
    fi
  else
    echo "skip image $M (exists)"
  fi

  # ---- text tower ----
  if [[ ! -f "$OUT/coco_karpathy_test_attn_text_${M}_seed_${SEED}.npy" ]]; then
    "$PY" -m utils.scripts.compute_activation_values_text \
      --text_descriptions coco_karpathy_test --model "$M" --pretrained "$P" \
      --seed $SEED --output_dir "$OUT"
  else
    echo "skip text $M (exists)"
  fi

  # ---- learned temperature (Metric B needs tau = 1/logit_scale) ----
  if [[ ! -f "$OUT/logit_scale_${M}.json" ]]; then
    MODEL="$M" PRETRAINED="$P" OUTDIR="$OUT" "$PY" - <<'PYEOF'
import json, os, torch
from utils.models.factory import create_model_and_transforms
m, p, out = os.environ["MODEL"], os.environ["PRETRAINED"], os.environ["OUTDIR"]
model, _, _ = create_model_and_transforms(m, pretrained=p)
with torch.no_grad():
    ls = model.logit_scale.exp().item()
json.dump({"model": m, "pretrained": p, "logit_scale": ls},
          open(os.path.join(out, f"logit_scale_{m}.json"), "w"))
print(m, "logit_scale =", ls)
PYEOF
  fi
done

echo "ALL DONE"
