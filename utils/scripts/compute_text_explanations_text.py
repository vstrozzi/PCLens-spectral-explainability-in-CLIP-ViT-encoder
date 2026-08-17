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
                             "(default) or with the evaluation dataset's IMAGE embeddings (same "
                             "shared-space dimensionality); 'image' uses --eval_dataset's images.")
    parser.add_argument("--probe_name", default="imagenet", type=str,
                        help="Reserved image dataset name (labeling only; the eval set is "
                             "--eval_dataset). Kept for backward compatibility.")
    parser.add_argument("--eval_dataset", default=None, type=str,
                        help="Image dataset providing the target's OWN evaluation images+labels for "
                             "the zero-shot metric (NOT a fixed external probe). Default: strip a "
                             "trailing '_classnames' from --dataset (e.g. binary_waterbirds_classnames "
                             "-> binary_waterbirds); falls back to --dataset itself.")
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

    # The TARGET's OWN images (CLIP embeddings + labels) score the reconstructed class prototypes --
    # NOT a fixed external probe. For an X_classnames text dataset this is X's own image set, so the
    # class ids line up (prototype c <-> image label c) and the zero-shot number is meaningful.
    eval_ds = args.eval_dataset or (
        args.dataset[:-len("_classnames")] if args.dataset.endswith("_classnames") else args.dataset)
    with open(os.path.join(args.input_dir, f"{eval_ds}_embeddings_{args.model}_seed_{args.seed}.npy"), "rb") as f:
        eval_img_emb = np.load(f)  # [M, d] image embeddings of the target dataset
    eval_labels = np.load(os.path.join(args.input_dir, f"{eval_ds}_labels_{args.model}_seed_{args.seed}.npy"))
    num_classes = int(text_labels.max()) + 1
    print(f"Evaluating against target images: {eval_ds} ({len(eval_img_emb)} images, "
          f"{len(np.unique(eval_labels))} classes)")

    def zero_shot(emb):
        """(image->label, label->image) top-1 accuracies (%), both directions, scoring the target
        dataset's OWN images against the reconstructed per-class text prototypes (mean over each
        class's rows): image->label: each image's argmax over prototypes == its label;
        label->image: each prototype's top-scoring image belongs to that class."""
        proto = np.stack([emb[text_labels == c].mean(axis=0) for c in range(num_classes)])  # [C, d]
        sims = eval_img_emb @ proto.T                                    # [M_eval, C]
        i2l = float((sims.argmax(axis=1) == eval_labels).mean() * 100.0)
        present = set(np.unique(eval_labels).tolist())                   # eval classes that exist
        top_img = sims.argmax(axis=0)                                    # [C] best eval image per prototype
        matched = [eval_labels[top_img[c]] == c for c in range(num_classes) if c in present]
        l2i = float(np.mean(matched) * 100.0) if matched else float("nan")
        return i2l, l2i

    # Full (un-ablated, un-reconstructed) accuracy: the completeness upper bound / real-model number.
    full_i2l, full_l2i = zero_shot(mlps.sum(axis=1) + attns.sum(axis=(1, 2)))

    # Probe set to LABEL the components (text sentences or image embeddings; same shared space).
    if args.probe_modality == "image":
        text_features = eval_img_emb
        lines = [imagenet_classes[int(l)] if int(l) < len(imagenet_classes) else f"{eval_ds}_{i}"
                 for i, l in enumerate(eval_labels)]
        probe_tag = f"_imgprobe_{eval_ds}"
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
        # Stable orthogonal projector onto the selected items' span (see compute_text_explanations.py).
        sv, vt = np.linalg.svd(pm, full_matrices=False)[1:]
        kk = int((sv > sv.max() * 1e-6).sum()) if sv.size and sv.max() > 0 else 0
        coeff = vt[:kk].T @ vt[:kk]
        Xd = np.asarray(data_slice, dtype=np.float32) - mean_d
        return ((Xd @ vh.T) @ coeff @ vh + mean_d).astype(np.float32)

    def pool_reconstruct(json_info, data_slice, pool_emb):
        """Reconstruct a unit from the top-1 pool embedding per PC (no poles). Same math as
        image_label; used for the 'self' pools of the text tower."""
        if "vh" not in json_info or "embeddings_sort" not in json_info or pool_emb is None:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)
        mean_d = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        Cc = pool_emb - pool_emb.mean(axis=0, keepdims=True)
        coords = (Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8)) @ vh.T
        pm = coords[[int(np.argmax(coords[:, pc])) for pc in range(coords.shape[1])]]
        sv, vt = np.linalg.svd(pm, full_matrices=False)[1:]
        kk = int((sv > sv.max() * 1e-6).sum()) if sv.size and sv.max() > 0 else 0
        coeff = vt[:kk].T @ vt[:kk]
        Xd = np.asarray(data_slice, dtype=np.float32) - mean_d
        return ((Xd @ vh.T) @ coeff @ vh + mean_d).astype(np.float32)

    # 'self' pools = the target's OWN paired subset (X_classnames -> X): its images and its class names.
    image_self_pool = eval_img_emb.astype(np.float32)                          # paired image subset
    _clf = os.path.join(args.input_dir, f"{eval_ds}_classifier_{args.model}.npy")
    text_self_pool = np.ascontiguousarray(np.load(_clf).T).astype(np.float32) if os.path.exists(_clf) else None

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
    image_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)      # image_ref
    text_self_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)  # own class names
    image_self_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32) # own paired images
    pc_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)
    # Kept-PC count (rank at 99% variance) of each decomposed head/MLP unit, for mean/std reporting.
    comp_ranks = []

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
                    comp_ranks.append(len(json_info.get("s", [])))
                    X = attns[:, i, head]
                    img_recon = image_label(json_info, X)
                    pc_recon = pc_reconstruct(X, json_info)
                    ts = pool_reconstruct(json_info, X, text_self_pool)
                    isf = pool_reconstruct(json_info, X, image_self_pool)
                    if img_recon is not None:
                        image_delta += img_recon - results
                        pc_delta += pc_recon - results
                        image_self_delta += isf - results
                        if ts is not None:
                            text_self_delta += ts - results
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
                comp_ranks.append(len(json_info.get("s", [])))
                X = mlps[:, i]
                img_recon = image_label(json_info, X)
                pc_recon = pc_reconstruct(X, json_info)
                ts = pool_reconstruct(json_info, X, text_self_pool)
                isf = pool_reconstruct(json_info, X, image_self_pool)
                if img_recon is not None:
                    image_delta += img_recon - results
                    pc_delta += pc_recon - results
                    image_self_delta += isf - results
                    if ts is not None:
                        text_self_delta += ts - results
                mlps[:, i] = results
                jsonl_file.write(json.dumps(
                    {"component": "mlp", "layer": i, "head": None, **json_info}) + "\n")

        final_embedding = mlps.sum(axis=1) + attns.sum(axis=(1, 2))  # text-span recon
        image_embedding = final_embedding + image_delta
        pc_embedding = final_embedding + pc_delta
        _, json_info = select_algo(
            final_embedding, text_features, lines, -1, -1, args.text_per_princ_comp, args.device)
        nan = float("nan")
        tr_i2l, tr_l2i = zero_shot(final_embedding)          # text_ref (probe text set)
        ir_i2l, ir_l2i = zero_shot(image_embedding)          # image_ref (image_set pool, e.g. imagenet)
        ts_i2l, ts_l2i = zero_shot(final_embedding + text_self_delta) if text_self_pool is not None else (nan, nan)
        is_i2l, is_l2i = zero_shot(final_embedding + image_self_delta)   # target's own paired images
        pc_i2l, pc_l2i = zero_shot(pc_embedding)
        print(f"Accuracy (image->label / label->image) -- "
              f"model(full): {full_i2l:.2f}/{full_l2i:.2f}  PC: {pc_i2l:.2f}/{pc_l2i:.2f}  "
              f"text_self: {ts_i2l:.2f}/{ts_l2i:.2f}  text_ref: {tr_i2l:.2f}/{tr_l2i:.2f}  "
              f"image_self: {is_i2l:.2f}/{is_l2i:.2f}  image_ref: {ir_i2l:.2f}/{ir_l2i:.2f}")

        final_object = {
            "component": args.components, "layer": -1, "head": -1,
            # Every *_accuracy is image->label; the matching *_l2i is label->image (both %).
            "accuracy": tr_i2l, "text_accuracy": tr_i2l,
            "text_ref_accuracy": tr_i2l, "text_ref_accuracy_l2i": tr_l2i,       # probe text set
            "text_self_accuracy": ts_i2l, "text_self_accuracy_l2i": ts_l2i,     # target's own class names
            "image_self_accuracy": is_i2l, "image_self_accuracy_l2i": is_l2i,   # target's own paired images
            "image_ref_accuracy": ir_i2l, "image_ref_accuracy_l2i": ir_l2i,     # image_set pool (imagenet)
            "image_accuracy": ir_i2l, "pc_accuracy": pc_i2l, "pc_accuracy_l2i": pc_l2i,
            "full_accuracy": full_i2l, "full_accuracy_l2i": full_l2i,
            "n_pcs": len(json_info.get("s", [])),
            "n_pcs_comp_mean": float(np.mean(comp_ranks)) if comp_ranks else float("nan"),
            "n_pcs_comp_std": float(np.std(comp_ranks)) if comp_ranks else float("nan"),
            "n_comps": len(comp_ranks),
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
