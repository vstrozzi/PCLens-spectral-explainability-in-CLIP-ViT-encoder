"""
Text-encoder counterpart of utils/scripts/compute_text_explanations.py.

Decomposes each multi-head-attention head of the CLIP *text* encoder (over the
EOS-token contributions produced by compute_activation_values_text.py) into
principal components and labels each PC with the closest text descriptions from
a probe dataset, via the shared `svd_data_approx` algorithm.

The PC directions (vh) live in the shared CLIP space, so the notebook also
characterizes each component "from the image side" by scoring vh against image
embeddings -- giving the bidirectional (image <-> text) interpretation.

Writes:
  {dataset}_completeness_text_{textprobe}_{model}_algo_{algorithm}_seed_{seed}.jsonl

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
                        help="Name of the text probe dataset used to label the components.")
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
                        help="Approximate/label the text-encoder components with a TEXT dataset "
                             "(default) or with IMAGE embeddings (same shared-space "
                             "dimensionality). 'image' loads {probe_name}_embeddings/_labels.")
    parser.add_argument("--probe_name", default="imagenet", type=str,
                        help="Image dataset name used when --probe_modality image.")
    parser.add_argument("--components", default="attn", choices=["attn", "mlp", "all"],
                        help="Which text-encoder components to decompose in the last "
                             "num_of_last_layers: attention heads, MLP layers, or both.")
    parser.add_argument("--out_file", default=None, type=str,
                        help="Explicit output .jsonl path (default keeps the completeness_ naming).")
    parser.add_argument("--image_set", default="self", type=str,
                        help="Image dataset(s) whose embeddings label the (text-encoder) PCs -- "
                             "the bidirectional image<->text interpretation. 'self' (the decomposed "
                             "text rows), 'all', or a comma-separated list of image dataset names "
                             "present as {ds}_embeddings_{model}_seed.")
    return parser


def main(args):
    with open(os.path.join(args.input_dir, f"{args.dataset}_attn_text_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        attns = np.load(f)  # [N, l, h, d]
    with open(os.path.join(args.input_dir, f"{args.dataset}_mlp_text_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        mlps = np.load(f)  # [N, l + 1, d]
    text_labels = np.load(
        os.path.join(args.input_dir, f"{args.dataset}_labels_text_{args.model}_seed_{args.seed}.npy"))

    assert attns.ndim == 4, (
        f"Expected summed per-head activations [N, l, h, d], got {attns.shape}. "
        "Re-run compute_activation_values_text without --spatial."
    )
    print(f"Number of text layers: {attns.shape[1]}, heads: {attns.shape[2]}")

    # Probe images (CLIP embeddings + labels) used to score the reconstructed class prototypes.
    with open(os.path.join(args.input_dir, f"{args.probe_name}_embeddings_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        text_features_imagenet = np.load(f)  # [M, d] image embeddings
        labs = np.load(os.path.join(args.input_dir, f"{args.probe_name}_labels_{args.model}_seed_{args.seed}.npy"))
    num_classes = int(text_labels.max()) + 1

    def zero_shot(emb):
        """Image -> argmax over reconstructed per-class text prototypes (mean over each class's rows)."""
        proto = np.stack([emb[text_labels == c].mean(axis=0) for c in range(num_classes)])  # [C, d]
        return float(((text_features_imagenet @ proto.T).argmax(axis=1) == labs).mean() * 100.0)

    # Full (un-ablated, un-reconstructed) accuracy: the completeness upper bound / real-model number.
    full_accuracy = zero_shot(mlps.sum(axis=1) + attns.sum(axis=(1, 2)))

    # Probe set to LABEL the components (text sentences or image embeddings; same shared space).
    if args.probe_modality == "image":
        text_features = text_features_imagenet
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

    # Row-aligned example metadata (decomposed text samples): index + class name, for the poles.
    idx_meta = [{"index": int(r),
                 "class_name": (imagenet_classes[int(text_labels[r])] if int(text_labels[r]) < len(imagenet_classes)
                                else str(int(text_labels[r])))}
                for r in range(len(text_labels))]

    def load_image_pool(spec):
        """Candidate pool (embeddings + idx/class) that labels the text-encoder PCs. 'self' -> the
        decomposed text rows; else image datasets / 'all' from {ds}_embeddings_{model}_seed.npy."""
        if spec == "self":
            return None, idx_meta
        suffix = f"_embeddings_{args.model}_seed_{args.seed}.npy"
        if spec == "all":
            names = sorted(os.path.basename(p)[:-len(suffix)]
                           for p in __import__("glob").glob(os.path.join(args.input_dir, f"*{suffix}"))
                           if "_text_" not in os.path.basename(p))
        else:
            names = [s.strip() for s in spec.split(",") if s.strip()]
        C, meta = [], []
        for nm in names:
            ep = os.path.join(args.input_dir, f"{nm}{suffix}")
            if not os.path.exists(ep):
                print(f"  [image_set] skip '{nm}': no {os.path.basename(ep)}")
                continue
            emb = np.load(ep).astype(np.float32)
            jm = os.path.join(args.input_dir, f"{nm}_idx_to_class_seed_{args.seed}.json")
            m = json.load(open(jm)) if os.path.exists(jm) else [{"index": r, "class_name": nm} for r in range(len(emb))]
            for e in m:
                e.setdefault("dataset", nm)
            C.append(emb)
            meta.extend(m[:len(emb)])
        if not C:
            print("  [image_set] nothing loaded; falling back to 'self'")
            return None, idx_meta
        print(f"  [image_set] pool = {spec} ({sum(len(c) for c in C)} images)")
        return np.concatenate(C, axis=0), meta

    C_pool, meta_pool = load_image_pool(args.image_set)

    def image_label(json_info, data_slice):
        """Label a unit's PCs with the image pool and reconstruct from the top-1 pool item per PC
        (symmetric to the text handling). Writes 'image' poles; returns reconstruction [N, d]."""
        if "vh" not in json_info or "embeddings_sort" not in json_info:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)
        mean_d = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        Craw = np.asarray(data_slice, dtype=np.float32) if C_pool is None else C_pool
        meta = idx_meta if C_pool is None else meta_pool
        Cc = Craw - Craw.mean(axis=0, keepdims=True)
        coords = (Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8)) @ vh.T
        kk = args.text_per_princ_comp
        for pc, entry in enumerate(json_info["embeddings_sort"]):
            col = coords[:, pc]
            top, bot = np.argsort(-col)[:kk], np.argsort(col)[:kk]
            entry["image"] = (
                [{f"image_max_{j}": int(meta[r]["index"]), f"class_max_{j}": meta[r].get("class_name"),
                  f"corr_max_{j}": float(col[r])} for j, r in enumerate(top)]
                + [{f"image_min_{j}": int(meta[r]["index"]), f"class_min_{j}": meta[r].get("class_name"),
                    f"corr_min_{j}": float(col[r])} for j, r in enumerate(bot)])
        pm = coords[[int(np.argmax(coords[:, pc])) for pc in range(coords.shape[1])]]
        coeff = pm.T @ np.linalg.pinv(pm @ pm.T) @ pm
        Xd = np.asarray(data_slice, dtype=np.float32) - mean_d
        return ((Xd @ vh.T) @ coeff @ vh + mean_d).astype(np.float32)

    def pc_reconstruct(data_slice, json_info):
        if "vh" not in json_info:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)
        mean = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        X = np.asarray(data_slice, dtype=np.float32) - mean
        return (X @ (vh.T @ vh) + mean).astype(np.float32)

    do_attn = args.components in ("attn", "all")
    do_mlp = args.components in ("mlp", "all")
    k = args.num_of_last_layers
    start_attn = max(0, attns.shape[1] - k)   # k >= #layers -> all layers (no ablation)
    start_mlp = max(0, mlps.shape[1] - k)
    image_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)
    pc_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)

    out_path = args.out_file or os.path.join(
        args.output_dir,
        f"{args.dataset}_completeness_text_{args.text_descriptions}_{args.model}_algo_{args.algorithm}_seed_{args.seed}{probe_tag}.jsonl",
    )
    with open(out_path, "w") as jsonl_file:
        select_algo = globals()[args.algorithm]

        if do_attn:
            for i in tqdm.trange(start_attn):
                for head in range(attns.shape[2]):
                    attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)
            for i in tqdm.trange(start_attn, attns.shape[1]):
                for head in range(attns.shape[2]):
                    results, json_info = select_algo(
                        attns[:, i, head], text_features, lines, i, head,
                        args.text_per_princ_comp, args.device, iters=args.max_text)
                    img_recon = image_label(json_info, attns[:, i, head])
                    pc_recon = pc_reconstruct(attns[:, i, head], json_info)
                    if img_recon is not None:
                        image_delta += img_recon - results
                        pc_delta += pc_recon - results
                    attns[:, i, head] = results
                    jsonl_file.write(json.dumps(
                        {"component": "attn", "layer": i, "head": head, **json_info}) + "\n")

        if do_mlp:
            for i in tqdm.trange(start_mlp):
                mlps[:, i] = np.mean(mlps[:, i], axis=0, keepdims=True)
            for i in tqdm.trange(start_mlp, mlps.shape[1]):
                results, json_info = select_algo(
                    mlps[:, i], text_features, lines, i, -1,
                    args.text_per_princ_comp, args.device, iters=args.max_text)
                img_recon = image_label(json_info, mlps[:, i])
                pc_recon = pc_reconstruct(mlps[:, i], json_info)
                if img_recon is not None:
                    image_delta += img_recon - results
                    pc_delta += pc_recon - results
                mlps[:, i] = results
                jsonl_file.write(json.dumps(
                    {"component": "mlp", "layer": i, "head": None, **json_info}) + "\n")

        final_embedding = mlps.sum(axis=1) + attns.sum(axis=(1, 2))  # text-span recon
        image_embedding = final_embedding + image_delta
        pc_embedding = final_embedding + pc_delta
        _, json_info = select_algo(
            final_embedding, text_features, lines, -1, -1, args.text_per_princ_comp, args.device)
        text_accuracy = zero_shot(final_embedding)
        image_accuracy = zero_shot(image_embedding)
        pc_accuracy = zero_shot(pc_embedding)
        print(f"Accuracy -- model(full): {full_accuracy:.2f}  PC: {pc_accuracy:.2f}  "
              f"image: {image_accuracy:.2f}  text: {text_accuracy:.2f}")

        final_object = {
            "component": args.components, "layer": -1, "head": -1,
            "accuracy": text_accuracy, "text_accuracy": text_accuracy,
            "image_accuracy": image_accuracy, "pc_accuracy": pc_accuracy,
            "full_accuracy": full_accuracy, "n_pcs": len(json_info.get("s", [])),
            **json_info,
        }
        jsonl_file.write(json.dumps(final_object))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
