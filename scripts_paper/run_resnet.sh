#!/usr/bin/env bash
# ResNet full run (paper: "Beyond transformers", appendix:resnet, section 6.5).
# Scales the RN50 pilot to samples_per_class=50 and runs the same dual-tower
# analysis as the ViTs (the RN50 text tower is a standard transformer, so the
# text side is unchanged). RN101 second by setting MODEL=RN101.
#
# Usage:
#   bash scripts_paper/run_resnet.sh
#   MODEL=RN101 SPC=50 bash scripts_paper/run_resnet.sh
set -euo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda:0}"
DATA_PATH="${DATA_PATH:-./datasets/}"
OUT="${OUT:-./output_dir}"
RES="${RES:-./results/paper}"
TEXTSET="${TEXTSET:-top_1500_nouns_5_sentences_imagenet_clean}"
ALGO="svd_data_approx"
MODEL="${MODEL:-RN50}"
PT="${PT:-openai}"
SEED="${SEED:-0}"
SPC="${SPC:-10}"
NLL="${NLL:-4}"     # ResNet has few blocks; decompose the last NLL of them

RDIR="$RES/$MODEL"; mkdir -p "$RDIR"
echo "================= $MODEL seed $SEED (pt=$PT, spc=$SPC) ================="

# 1) vision activations (ResNet layer x head decomposition)
$PY -m utils.scripts.compute_activation_values_resnet --model "$MODEL" --pretrained "$PT" \
    --dataset imagenet --samples_per_class "$SPC" --tot_samples_per_class 50 \
    --device "$DEVICE" --output_dir "$OUT" --seed "$SEED"

# 2) text-tower activations + explanations (standard transformer)
$PY -m utils.scripts.compute_activation_values_text --model "$MODEL" --pretrained "$PT" \
    --text_descriptions "$TEXTSET" --seed "$SEED" --device "$DEVICE" --output_dir "$OUT"

# 3) text-set + class embeddings
$PY -m utils.scripts.compute_text_embeddings --model "$MODEL" --pretrained "$PT" \
    --data_path "utils/text_descriptions/${TEXTSET}.txt" --device "$DEVICE" --output_dir "$OUT"
$PY -m utils.scripts.compute_classes_embeddings --model "$MODEL" --pretrained "$PT" \
    --dataset imagenet --device "$DEVICE" --output_dir "$OUT"

# 4) explanations, both towers
$PY -m utils.scripts.compute_text_explanations --model "$MODEL" \
    --dataset imagenet --text_descriptions "$TEXTSET" --algorithm "$ALGO" \
    --num_of_last_layers "$NLL" --seed "$SEED" --device "$DEVICE" --input_dir "$OUT" --output_dir "$OUT"
$PY -m utils.scripts.compute_text_explanations_text --model "$MODEL" \
    --dataset imagenet_descriptions_personal --text_descriptions "$TEXTSET" --algorithm "$ALGO" \
    --num_of_last_layers "$NLL" --seed "$SEED" --device "$DEVICE" --input_dir "$OUT" --output_dir "$OUT"

# 5) alive-fraction export (gates essentially never all-zero)
$PY -m utils.scripts.export_alive_fraction --model "$MODEL" --dataset imagenet --seed "$SEED" \
    --input_dir "$OUT" --out "$RDIR/alive_fraction.json"

# 6) shared-pairs dual-tower analysis (Metric A/B + interaction + paired ablation)
VIS_EXPL="imagenet_completeness_${TEXTSET}_${MODEL}_algo_${ALGO}_seed_${SEED}.jsonl"
TXT_EXPL="imagenet_descriptions_personal_completeness_text_${TEXTSET}_${MODEL}_algo_${ALGO}_seed_${SEED}.jsonl"
$PY -m utils.scripts.shared_pairs_analysis --model "$MODEL" --seed "$SEED" --input_dir "$OUT" \
    --vision_expl "$VIS_EXPL" --text_expl "$TXT_EXPL" \
    --pair_k 50 --cutoff 0.99 --subspace_k 8 --trials 5 --out "$RDIR/shared_pairs_seed_${SEED}.json"

# 7) Waterbirds bias removal on ResNet (PCSelection math unchanged)
$PY -m utils.scripts.bias_removal_test --model "$MODEL" --pretrained "$PT" \
    --dataset binary_waterbirds --dataset_text "$TEXTSET" --backbone resnet \
    --num_real_layer "$NLL" --seed "$SEED" --device "$DEVICE" --output_dir "$OUT" || \
    echo "WARN: bias_removal_test (waterbirds) needs the binary_waterbirds activations precomputed"

echo "run_resnet.sh complete."
