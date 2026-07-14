""" 
Adapted from https://github.com/yossigandelsman/clip_text_span. MIT License Copyright (c) 2024 Yossi Gandelsman
"""
import numpy as np
import torch
import os
import json
import tqdm
import argparse
from pathlib import Path
from torch.nn import functional as F
from utils.misc.misc import accuracy

from utils.scripts.algorithms_text_explanations import svd_data_approx # All the algorithms
from utils.scripts.algorithms_text_explanations_prev import * # All the algorithms
def get_args_parser():
    parser = argparse.ArgumentParser("Completeness part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,   
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--text_descriptions",
        default="image_descriptions_per_class",
        type=str,
        help="name of the evalauted text set",
    )
    parser.add_argument(
        "--text_dir",
        default="./utils/text_descriptions",
        type=str,
        help="The folder with the text files",
    )

    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet or waterbirds"
    )
    parser.add_argument(
        "--num_of_last_layers",
        type=int,
        default=4,
        help="How many attention layers to replace.",
    )

    parser.add_argument(
        "--text_per_princ_comp",
        type=int,
        default=5,
        help="The number of text examples per princ_comp.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed used for the dataset.",
    )

    parser.add_argument(
        "--max_text",
        type=int,
        default=80,
        help="The maximum number of text to use for the approximation.",
    )

    parser.add_argument("--algorithm", default="svd_data_approx", help="The algorithm to use")
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--components", default="attn", choices=["attn", "mlp", "all"],
                        help="Which residual-stream components to text-explain in the last "
                             "num_of_last_layers: attention heads, MLP layers, or both. Only the "
                             "selected components are mean-ablated (early) and text-reconstructed; "
                             "the others are summed in at full strength.")
    parser.add_argument("--out_file", default=None, type=str,
                        help="Explicit output .jsonl path. Default keeps the "
                             "{dataset}_completeness_..._seed_{seed}.jsonl naming.")
    parser.add_argument("--image_set", default="self", type=str,
                        help="Image dataset(s) whose embeddings label the PCs (image poles + "
                             "image-span reconstruction), mirroring --text_descriptions: 'self' "
                             "(the decomposed dataset's own images), 'all', or a comma-separated "
                             "list of image dataset names present as {ds}_embeddings_{model}_seed.")
    return parser


def main(args):
    """
    Evaluate a CLIP representation for a given dataset of text. This is needed to run text_span algorithm.
    """
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}_seed_{args.seed}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}_seed_{args.seed}.npy"), "rb"
    ) as f:
        mlps = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    
    labels = np.load(os.path.join(args.input_dir, f"{args.dataset}_labels_{args.model}_seed_{args.seed}.npy"))

    print(f"Number of layers: {attns.shape[1]}")

    # Zero-shot accuracy of a summed [N, d] embedding against the saved class classifier.
    classifier_t = torch.from_numpy(classifier).float().to(args.device)
    labels_t = torch.from_numpy(labels)

    def zero_shot(emb):
        proj = torch.from_numpy(np.ascontiguousarray(emb)).float().to(args.device) @ classifier_t
        return accuracy(proj.cpu(), labels_t)[0] * 100.0

    # Full (un-ablated, un-reconstructed) accuracy: the completeness upper bound.
    full_accuracy = zero_shot(mlps.sum(axis=1) + attns.sum(axis=(1, 2)))

    # Load text descriptions:
    with open(
        os.path.join(args.input_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb"
    ) as f:
        text_features = np.load(f)
    with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]

    # Row-aligned idx->class map (written by extract_activations) for the decomposed dataset,
    # used both as the default ('self') image pool and to fall back on.
    idx_map_path = os.path.join(args.input_dir, f"{args.dataset}_idx_to_class_seed_{args.seed}.json")
    if os.path.exists(idx_map_path):
        with open(idx_map_path) as f:
            idx_meta = json.load(f)  # list of {index, label, class_name}, one per activation row
    else:
        idx_meta = [{"index": int(r), "class_name": str(int(labels[r]))} for r in range(len(labels))]

    def load_image_pool(spec):
        """Candidate image pool (embeddings + idx/class metadata) that labels the PCs, mirroring the
        text set. 'self' -> the decomposed dataset's own activation rows; else named datasets / 'all'
        loaded from {ds}_embeddings_{model}_seed.npy (+ their idx->class maps). Returns (C or None, meta)."""
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
        """Label a unit's PCs with images from the pool and reconstruct it from the top-1 image per
        PC (symmetric to svd_data_approx's text handling). Writes 'image' poles into json_info;
        returns the image-span reconstruction [N, d] (or None)."""
        if "vh" not in json_info or "embeddings_sort" not in json_info:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)                          # [r, d]
        mean_d = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        Craw = np.asarray(data_slice, dtype=np.float32) if C_pool is None else C_pool
        meta = idx_meta if C_pool is None else meta_pool
        Cc = Craw - Craw.mean(axis=0, keepdims=True)
        coords = (Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8)) @ vh.T   # [M, r] in PC space
        kk = args.text_per_princ_comp
        for pc, entry in enumerate(json_info["embeddings_sort"]):
            col = coords[:, pc]
            top, bot = np.argsort(-col)[:kk], np.argsort(col)[:kk]
            entry["image"] = (
                [{f"image_max_{j}": int(meta[r]["index"]), f"class_max_{j}": meta[r].get("class_name"),
                  f"corr_max_{j}": float(col[r])} for j, r in enumerate(top)]
                + [{f"image_min_{j}": int(meta[r]["index"]), f"class_min_{j}": meta[r].get("class_name"),
                    f"corr_min_{j}": float(col[r])} for j, r in enumerate(bot)])
        # reconstruct the unit from the top-1 pool image per PC (least squares in PC space)
        pm = coords[[int(np.argmax(coords[:, pc])) for pc in range(coords.shape[1])]]  # [r, r]
        coeff = pm.T @ np.linalg.pinv(pm @ pm.T) @ pm
        Xd = np.asarray(data_slice, dtype=np.float32) - mean_d
        return ((Xd @ vh.T) @ coeff @ vh + mean_d).astype(np.float32)

    def pc_reconstruct(data_slice, json_info):
        """Pure rank-r PC reconstruction (project onto the kept PC subspace vh, no text/image
        example approximation) -- the completeness upper bound of the kept PCs. Returns [N, d]."""
        if "vh" not in json_info:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)                      # [r, d]
        mean = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        X = np.asarray(data_slice, dtype=np.float32) - mean                     # [N, d]
        return (X @ (vh.T @ vh) + mean).astype(np.float32)

    do_attn = args.components in ("attn", "all")
    do_mlp = args.components in ("mlp", "all")
    k = args.num_of_last_layers
    # First decomposed layer per stack; clamped so k >= #layers means "all layers" (no ablation).
    start_attn = max(0, attns.shape[1] - k)
    start_mlp = max(0, mlps.shape[1] - k)
    # Accumulate (alt recon - text recon) per decomposed unit so the image-span and PC-subspace
    # embeddings are emb_text + delta, without keeping extra full copies of the activations.
    image_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)
    pc_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)

    out_path = args.out_file or os.path.join(
        args.output_dir,
        f"{args.dataset}_completeness_{args.text_descriptions}_{args.model}_algo_{args.algorithm}_seed_{args.seed}.jsonl",
    )
    with open(out_path, "w") as jsonl_file:
        select_algo = globals()[args.algorithm]

        # --- attention heads ---
        if do_attn:
            for i in tqdm.trange(start_attn):                         # mean-ablate early heads
                for head in range(attns.shape[2]):
                    attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)
            for i in tqdm.trange(start_attn, attns.shape[1]):          # decompose last-k heads
                for head in range(attns.shape[2]):
                    results, json_info = select_algo(
                        attns[:, i, head], text_features, lines, i, head,
                        args.text_per_princ_comp, args.device, iters=args.max_text)
                    img_recon = image_label(json_info, attns[:, i, head])  # writes poles, returns recon
                    pc_recon = pc_reconstruct(attns[:, i, head], json_info)
                    if img_recon is not None:
                        image_delta += img_recon - results
                        pc_delta += pc_recon - results
                    attns[:, i, head] = results
                    jsonl_file.write(json.dumps(
                        {"component": "attn", "layer": i, "head": head, **json_info}) + "\n")

        # --- MLP layers (mlps is [N, l+1, d]; treat each layer as one unit, head = None) ---
        if do_mlp:
            for i in tqdm.trange(start_mlp):                          # mean-ablate early MLPs
                mlps[:, i] = np.mean(mlps[:, i], axis=0, keepdims=True)
            for i in tqdm.trange(start_mlp, mlps.shape[1]):            # decompose last-k MLPs
                results, json_info = select_algo(
                    mlps[:, i], text_features, lines, i, -1,
                    args.text_per_princ_comp, args.device, iters=args.max_text)
                img_recon = image_label(json_info, mlps[:, i])  # writes poles, returns recon
                pc_recon = pc_reconstruct(mlps[:, i], json_info)
                if img_recon is not None:
                    image_delta += img_recon - results
                    pc_delta += pc_recon - results
                mlps[:, i] = results
                jsonl_file.write(json.dumps(
                    {"component": "mlp", "layer": i, "head": None, **json_info}) + "\n")

        # Reassemble the (partly reconstructed) embedding and score every reconstruction.
        mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))   # text-span recon
        image_embedding = mean_ablated_and_replaced + image_delta               # image-span recon
        pc_embedding = mean_ablated_and_replaced + pc_delta                      # rank-r PC recon
        _, json_info = select_algo(
            mean_ablated_and_replaced, text_features, lines, -1, -1,
            args.text_per_princ_comp, args.device)
        text_accuracy = zero_shot(mean_ablated_and_replaced)
        image_accuracy = zero_shot(image_embedding)
        pc_accuracy = zero_shot(pc_embedding)
        print(f"Accuracy -- model(full): {full_accuracy:.2f}  PC: {pc_accuracy:.2f}  "
              f"image: {image_accuracy:.2f}  text: {text_accuracy:.2f}")

        json_object = {
            "component": args.components,
            "layer": -1,
            "head": -1,
            "accuracy": text_accuracy,          # text-span reconstruction (PCLens completeness)
            "text_accuracy": text_accuracy,     # top-1 text/PC span
            "image_accuracy": image_accuracy,   # top-1 image/PC span
            "pc_accuracy": pc_accuracy,         # rank-r PC subspace, no example approximation
            "full_accuracy": full_accuracy,     # real-model zero-shot (all components summed)
            "n_pcs": len(json_info.get("s", [])),
            **json_info,
        }
        jsonl_file.write(json.dumps(json_object))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
