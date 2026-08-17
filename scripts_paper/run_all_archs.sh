#!/usr/bin/env bash
# All-architecture dual-tower pipeline (paper sections 6.1, 6.8).
# Loops the full precompute + analysis over B-32 / B-16 / L-14 / H-14 and seeds.
#
# Memory note: vision activations [50000,32,16,1024] fp32 ~= 105 GB. For L-14/H-14
# we use 10 img/class + last 11 layers + fp16 (~3.6 GB). B-32/B-16 stay at 50k for
# comparability with the existing numbers.
#
# Usage:
#   bash scripts_paper/run_all_archs.sh                 # all models, seed 0
#   MODELS="ViT-B-16" SEEDS="0 1 2" bash scripts_paper/run_all_archs.sh
set -euo pipefail

PY="${PY:-python}"                      # set PY=/path/to/envs/MT/bin/python if needed
DEVICE="${DEVICE:-cuda:0}"
DATA_PATH="${DATA_PATH:-./datasets/}"
OUT="${OUT:-./output_dir}"
RES="${RES:-./results/paper}"
TEXTSET="${TEXTSET:-top_1500_nouns_5_sentences_imagenet_clean}"
ALGO="svd_data_approx"
MODELS="${MODELS:-ViT-B-32 ViT-B-16 ViT-L-14 ViT-H-14}"
SEEDS="${SEEDS:-0}"

# per-model pretrained tag
pretrained_for() {
  case "$1" in
    ViT-B-32) echo "laion2b_s34b_b79k" ;;
    ViT-B-16) echo "laion2b_s34b_b88k" ;;
    ViT-L-14) echo "laion2b_s32b_b82k" ;;
    ViT-H-14) echo "laion2b_s32b_b79k" ;;
    *) echo "laion2b_s34b_b79k" ;;
  esac
}

for MODEL in $MODELS; do
  PT="$(pretrained_for "$MODEL")"
  # memory profile: big models -> fewer samples, last-11 layers, fp16
  case "$MODEL" in
    ViT-L-14|ViT-H-14) SPC="10"; LAST="11"; DTYPE="fp16"; NLL="11" ;;
    *)                 SPC="";   LAST="";   DTYPE="fp32"; NLL="11" ;;
  esac
  MEM_ARGS=""
  [ -n "$SPC" ]  && MEM_ARGS="$MEM_ARGS --samples_per_class $SPC --tot_samples_per_class 50"
  [ -n "$LAST" ] && MEM_ARGS="$MEM_ARGS --last_layers_only $LAST --save_dtype $DTYPE"

  for SEED in $SEEDS; do
    echo "================= $MODEL seed $SEED (pt=$PT) ================="
    RDIR="$RES/$MODEL"; mkdir -p "$RDIR"

    # 1) vision activations
    $PY -m utils.scripts.compute_activation_values --model "$MODEL" --pretrained "$PT" \
        --dataset imagenet --data_path "$DATA_PATH" --seed "$SEED" --device "$DEVICE" \
        --output_dir "$OUT" $MEM_ARGS

    # 2) text-tower activations
    $PY -m utils.scripts.compute_activation_values_text --model "$MODEL" --pretrained "$PT" \
        --text_descriptions "$TEXTSET" --seed "$SEED" --device "$DEVICE" --output_dir "$OUT"

    # 3) text-set embeddings + zero-shot class embeddings
    $PY -m utils.scripts.compute_text_embeddings --model "$MODEL" --pretrained "$PT" \
        --data_path "utils/text_descriptions/${TEXTSET}.txt" --device "$DEVICE" --output_dir "$OUT"
    $PY -m utils.scripts.compute_classes_embeddings --model "$MODEL" --pretrained "$PT" \
        --dataset imagenet --device "$DEVICE" --output_dir "$OUT"

    # 4) text explanations, both towers (last 11 layers)
    $PY -m utils.scripts.compute_text_explanations --model "$MODEL" \
        --dataset imagenet --text_descriptions "$TEXTSET" --algorithm "$ALGO" \
        --num_of_last_layers "$NLL" --seed "$SEED" --device "$DEVICE" \
        --input_dir "$OUT" --output_dir "$OUT"
    $PY -m utils.scripts.compute_text_explanations_text --model "$MODEL" \
        --dataset imagenet_descriptions_personal --text_descriptions "$TEXTSET" --algorithm "$ALGO" \
        --num_of_last_layers "$NLL" --seed "$SEED" --device "$DEVICE" \
        --input_dir "$OUT" --output_dir "$OUT"

    # 5) verify the decomposition reconstructs the true embedding (cos ~ 1.0) on both towers
    $PY -m utils.scripts.verify_decomposition_activations --model "$MODEL" --pretrained "$PT" \
        --seed "$SEED" --device "$DEVICE" --output_dir "$OUT" --image_dataset imagenet \
        --data_path "$DATA_PATH" > "$RDIR/verify_decomposition_seed_${SEED}.json" || \
        echo "WARN: verify_decomposition failed for $MODEL seed $SEED"

    # 6) ablation orderings (both towers, attn + mlp)
    for TOWER in vision text; do
      for KIND in attn mlp; do
        $PY -m utils.scripts.ablation_orderings --model "$MODEL" --seed "$SEED" \
            --input_dir "$OUT" --tower "$TOWER" --kind "$KIND" --trials 5 \
            --out "$RDIR/ablation_orderings_${TOWER}_${KIND}_seed_${SEED}.json"
      done
    done

    # 7) shared-pairs dual-tower analysis
    VIS_EXPL="imagenet_completeness_${TEXTSET}_${MODEL}_algo_${ALGO}_seed_${SEED}.jsonl"
    TXT_EXPL="imagenet_descriptions_personal_completeness_text_${TEXTSET}_${MODEL}_algo_${ALGO}_seed_${SEED}.jsonl"
    $PY -m utils.scripts.shared_pairs_analysis --model "$MODEL" --seed "$SEED" \
        --input_dir "$OUT" --vision_expl "$VIS_EXPL" --text_expl "$TXT_EXPL" \
        --pair_k 50 --cutoff 0.99 --subspace_k 8 --trials 5 \
        --out "$RDIR/shared_pairs_seed_${SEED}.json"
  done
done
echo "run_all_archs.sh complete."
