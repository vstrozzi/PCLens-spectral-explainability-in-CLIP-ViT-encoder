"""
Text-encoder counterpart of utils/scripts/compute_text_explanations.py.

Decomposes each multi-head-attention head of the CLIP *text* encoder (over the
EOS-token contributions produced by compute_activation_values_text.py) into
principal components and labels each PC with the closest text descriptions from
a probe bank, via the shared `svd_data_approx` algorithm.

The PC directions (vh) live in the shared CLIP space, so the notebook also
characterizes each component "from the image side" by scoring vh against image
embeddings -- giving the bidirectional (image <-> text) interpretation.

Writes:
  {bank}_completeness_text_{textprobe}_{model}_algo_{algorithm}_seed_{seed}.jsonl

Adapted from https://github.com/yossigandelsman/clip_text_span. MIT License
Copyright (c) 2024 Yossi Gandelsman.
"""
import numpy as np
import os
import json
import tqdm
import argparse
from pathlib import Path

from utils.scripts.algorithms_text_explanations import svd_data_approx  # main algorithm
from utils.scripts.algorithms_text_explanations_prev import *  # other selectable algorithms
from utils.datasets_constants.imagenet_classes import imagenet_classes


def get_args_parser():
    parser = argparse.ArgumentParser("Text-encoder spectral decomposition", add_help=False)
    parser.add_argument("--model", default="ViT-B-32", type=str, metavar="MODEL")
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--input_dir", default="./output_dir")
    parser.add_argument("--dataset", default="imagenet_descriptions_personal", type=str,
                        help="Name of the decomposed text dataset (matches compute_activation_values_text).")
    parser.add_argument("--text_descriptions", default="top_1500_nouns_5_sentences_imagenet_clean", type=str,
                        help="Name of the text probe bank used to label the components.")
    parser.add_argument("--text_dir", default="./utils/text_descriptions", type=str)
    parser.add_argument("--num_of_last_layers", type=int, default=4,
                        help="How many of the last text-encoder layers to decompose.")
    parser.add_argument("--text_per_princ_comp", type=int, default=5,
                        help="Number of text examples to keep per principal component.")
    parser.add_argument("--max_text", type=int, default=80,
                        help="Maximum number of PCs / texts to use for the approximation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", default="svd_data_approx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe_modality", default="text", choices=["text", "image"],
                        help="Approximate/label the text-encoder components with a TEXT bank "
                             "(default) or with IMAGE embeddings (same shared-space "
                             "dimensionality). 'image' loads {probe_name}_embeddings/_labels.")
    parser.add_argument("--probe_name", default="imagenet", type=str,
                        help="Image dataset name used when --probe_modality image.")
    return parser


def main(args):
    with open(os.path.join(args.input_dir, f"{args.dataset}_attn_text_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        attns = np.load(f)  # [N, l, h, d]
    with open(os.path.join(args.input_dir, f"{args.dataset}_mlp_text_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        mlps = np.load(f)  # [N, l + 1, d]

    assert attns.ndim == 4, (
        f"Expected summed per-head activations [N, l, h, d], got {attns.shape}. "
        "Re-run compute_activation_values_text without --spatial."
    )
    print(f"Number of text layers: {attns.shape[1]}, heads: {attns.shape[2]}")

    # Mean-ablate every head outside the last `num_of_last_layers` layers.
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers):
        for head in range(attns.shape[2]):
            attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)

    # Load the probe set (CLIP embeddings + labels) used to approximate/label the components.
    # Both modalities live in the same shared CLIP space, so svd_data_approx / text_span are
    # unchanged -- only the probe matrix and its labels differ.
    if args.probe_modality == "image":
        with open(os.path.join(args.input_dir, f"{args.probe_name}_embeddings_{args.model}_seed_{args.seed}.npy"), "rb") as f:
            text_features = np.load(f)  # [M, d] image embeddings
        labs = np.load(os.path.join(args.input_dir, f"{args.probe_name}_labels_{args.model}_seed_{args.seed}.npy"))
        lines = [imagenet_classes[int(l)] if int(l) < len(imagenet_classes) else f"{args.probe_name}_{i}"
                 for i, l in enumerate(labs)]
        probe_tag = f"_imgprobe_{args.probe_name}"
    else:
        with open(os.path.join(args.input_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb") as f:
            text_features = np.load(f)  # [M, d] text embeddings
        with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
            lines = [i.replace("\n", "") for i in f.readlines()]
        probe_tag = ""
    print(f"Probe modality: {args.probe_modality} ({text_features.shape[0]} probes, dim {text_features.shape[1]})")

    out_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_completeness_text_{args.text_descriptions}_{args.model}_algo_{args.algorithm}_seed_{args.seed}{probe_tag}.jsonl",
    )
    with open(out_path, "w") as jsonl_file:
        select_algo = globals()[args.algorithm]
        for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]):
            for head in range(attns.shape[2]):
                results, json_info = select_algo(
                    attns[:, i, head], text_features, lines, i, head,
                    args.text_per_princ_comp, args.device, iters=args.max_text,
                )
                attns[:, i, head] = results
                jsonl_file.write(json.dumps({"layer": i, "head": head, **json_info}) + "\n")

        # Final (mean-ablated-and-replaced) text embedding decomposition (head == -1).
        final_embedding = mlps.sum(axis=1) + attns.sum(axis=(1, 2))  # [N, d]
        _, json_info = select_algo(
            final_embedding, text_features, lines, -1, -1, args.text_per_princ_comp, args.device,
        )
        jsonl_file.write(json.dumps({"layer": -1, "head": -1, **json_info}))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
