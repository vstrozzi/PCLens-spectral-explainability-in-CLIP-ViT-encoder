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
    parser.add_argument("--image_set", default="imagenet", type=str,
                        help="Reference image pool for the image_ref column and the visualization "
                             "poles (default imagenet): 'self' (per-component activations), 'all', or "
                             "a comma-separated list of image dataset names present as "
                             "{ds}_embeddings_{model}_seed. The image_self column always uses the "
                             "decomposed dataset's own image embeddings, and text_self always uses "
                             "its class names, independent of this flag.")
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
    classifier_t = torch.from_numpy(classifier).float().to(args.device)   # [d, C]
    labels_t = torch.from_numpy(labels)
    _present = torch.unique(labels_t)                                     # classes with >=1 image

    def zero_shot(emb):
        """Return the (image->label, label->image) top-1 accuracies in %, so both directions are
        always comparable regardless of tower:
          image->label: for each image, argmax over class prototypes == its label (standard zero-shot);
          label->image: for each present class, the single top-scoring image belongs to that class."""
        proj = torch.from_numpy(np.ascontiguousarray(emb)).float().to(args.device) @ classifier_t  # [N, C]
        i2l = float(accuracy(proj.cpu(), labels_t)[0]) * 100.0
        top_img = proj.argmax(dim=0).cpu()                               # [C] best image per class
        l2i = 100.0 * torch.stack([(labels_t[top_img[c]] == c) for c in _present]).float().mean().item()
        return i2l, l2i

    # Full (un-ablated, un-reconstructed) accuracy: the completeness upper bound.
    full_i2l, full_l2i = zero_shot(mlps.sum(axis=1) + attns.sum(axis=(1, 2)))

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

    # ---- candidate pools for the four example-approximation columns --------------------------- #
    # text_ref : general text set (--text_descriptions); handled inside svd_data_approx (`results`).
    # text_self: the decomposed dataset's own class names -- already passed through the OpenAI
    #            template ensemble and averaged per class = the columns of the zero-shot classifier
    #            ({dataset}_classifier_{model}.npy), so no extra compute is needed.
    text_self_pool = np.ascontiguousarray(classifier.T).astype(np.float32)          # [C, d]
    # image_self: the decomposed dataset's own image embeddings (fallback: per-component activations).
    _self_emb = os.path.join(args.input_dir, f"{args.dataset}_embeddings_{args.model}_seed_{args.seed}.npy")
    image_self_pool = np.load(_self_emb).astype(np.float32) if os.path.exists(_self_emb) else None
    # image_ref : a general image dataset (default imagenet), selected by --image_set.
    image_ref_pool, image_ref_meta = load_image_pool(args.image_set)
    image_ref_ok = (args.image_set == "self") or (image_ref_pool is not None)

    def pool_reconstruct(json_info, data_slice, pool_emb, meta=None, pole_key=None):
        """Reconstruct a decomposed unit from the top-1 pool embedding per PC (least squares in the
        unit's PC subspace), symmetric to svd_data_approx's text handling. `pool_emb` is a fixed
        [M, d] candidate matrix (class names / own images / imagenet ...); None -> the unit's own
        activation rows are the pool. Each PC independently takes the argmax pool item (no
        de-duplication, so the same item may serve several PCs). When `pole_key` is set, also
        writes the top/bottom-k poles under json_info[pc][pole_key]. Returns [N, d] (or None)."""
        if "vh" not in json_info or "embeddings_sort" not in json_info:
            return None
        vh = np.asarray(json_info["vh"], dtype=np.float32)                          # [r, d]
        mean_d = np.asarray(json_info.get("mean_values_att", 0.0), dtype=np.float32)
        Craw = np.asarray(data_slice, dtype=np.float32) if pool_emb is None else pool_emb
        Cc = Craw - Craw.mean(axis=0, keepdims=True)                                # center by pool mean
        coords = (Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8)) @ vh.T   # [M, r] cos w/ PC axes
        if pole_key is not None:
            mm = meta if (pool_emb is not None and meta is not None) else idx_meta
            kk = args.text_per_princ_comp
            for pc, entry in enumerate(json_info["embeddings_sort"]):
                col = coords[:, pc]
                top, bot = np.argsort(-col)[:kk], np.argsort(col)[:kk]
                entry[pole_key] = (
                    [{f"image_max_{j}": int(mm[r]["index"]), f"class_max_{j}": mm[r].get("class_name"),
                      f"corr_max_{j}": float(col[r])} for j, r in enumerate(top)]
                    + [{f"image_min_{j}": int(mm[r]["index"]), f"class_min_{j}": mm[r].get("class_name"),
                        f"corr_min_{j}": float(col[r])} for j, r in enumerate(bot)])
        pm = coords[[int(np.argmax(coords[:, pc])) for pc in range(coords.shape[1])]]  # [r, r]
        # Orthogonal projector onto the span of the selected pool items (row space of pm), via SVD so
        # it stays stable and idempotent when the top-1 picks repeat and pm is strongly rank-deficient
        # (e.g. a 2-class-name pool). The old pm.T @ pinv(pm @ pm.T) @ pm squares the condition number
        # and blows up there.
        sv, vt = np.linalg.svd(pm, full_matrices=False)[1:]
        kk = int((sv > sv.max() * 1e-6).sum()) if sv.size and sv.max() > 0 else 0
        coeff = vt[:kk].T @ vt[:kk]
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
    # Accumulate (alt recon - text_ref recon) per decomposed unit so each variant embedding is
    # base(text_ref) + delta, without keeping extra full copies of the activations.
    text_self_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)   # class names
    image_self_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)  # own images
    image_ref_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)   # imagenet
    pc_delta = np.zeros((attns.shape[0], attns.shape[-1]), dtype=np.float32)
    # Kept-PC count (rank at 99% variance) of each decomposed head/MLP unit, for mean/std reporting.
    comp_ranks = []

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
                    comp_ranks.append(len(json_info.get("s", [])))
                    X = attns[:, i, head]
                    # image_ref (imagenet) writes the visualization poles ("image"); the class-name and
                    # own-image pools contribute accuracy only.
                    ir = pool_reconstruct(json_info, X, image_ref_pool, image_ref_meta,
                                          pole_key="image") if image_ref_ok else None
                    ts = pool_reconstruct(json_info, X, text_self_pool)
                    isf = pool_reconstruct(json_info, X, image_self_pool)
                    pc_recon = pc_reconstruct(X, json_info)
                    if ts is not None:
                        text_self_delta += ts - results
                        image_self_delta += isf - results
                        pc_delta += pc_recon - results
                        if ir is not None:
                            image_ref_delta += ir - results
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
                comp_ranks.append(len(json_info.get("s", [])))
                X = mlps[:, i]
                ir = pool_reconstruct(json_info, X, image_ref_pool, image_ref_meta,
                                      pole_key="image") if image_ref_ok else None
                ts = pool_reconstruct(json_info, X, text_self_pool)
                isf = pool_reconstruct(json_info, X, image_self_pool)
                pc_recon = pc_reconstruct(X, json_info)
                if ts is not None:
                    text_self_delta += ts - results
                    image_self_delta += isf - results
                    pc_delta += pc_recon - results
                    if ir is not None:
                        image_ref_delta += ir - results
                mlps[:, i] = results
                jsonl_file.write(json.dumps(
                    {"component": "mlp", "layer": i, "head": None, **json_info}) + "\n")

        # Reassemble the text_ref reconstruction (base) and score every example-approximation variant.
        base = mlps.sum(axis=1) + attns.sum(axis=(1, 2))                    # text_ref (general text set)
        pc_embedding = base + pc_delta                                       # rank-r PC recon
        _, json_info = select_algo(
            base, text_features, lines, -1, -1, args.text_per_princ_comp, args.device)
        nan = float("nan")
        tr_i2l, tr_l2i = zero_shot(base)                                        # general text set
        ts_i2l, ts_l2i = zero_shot(base + text_self_delta)                      # class-name pool
        is_i2l, is_l2i = zero_shot(base + image_self_delta)                     # own images
        ir_i2l, ir_l2i = zero_shot(base + image_ref_delta) if image_ref_ok else (nan, nan)  # imagenet
        pc_i2l, pc_l2i = zero_shot(pc_embedding)
        print(f"Accuracy (image->label / label->image) -- "
              f"model(full): {full_i2l:.2f}/{full_l2i:.2f}  PC: {pc_i2l:.2f}/{pc_l2i:.2f}  "
              f"text_self: {ts_i2l:.2f}/{ts_l2i:.2f}  text_ref: {tr_i2l:.2f}/{tr_l2i:.2f}  "
              f"image_self: {is_i2l:.2f}/{is_l2i:.2f}  image_ref: {ir_i2l:.2f}/{ir_l2i:.2f}")

        json_object = {
            "component": args.components,
            "layer": -1,
            "head": -1,
            # Every *_accuracy is image->label; the matching *_l2i is label->image (both %).
            "accuracy": tr_i2l,                     # back-compat: PCLens completeness == text_ref (i2l)
            "text_accuracy": tr_i2l,                # back-compat alias
            "text_ref_accuracy": tr_i2l, "text_ref_accuracy_l2i": tr_l2i,     # general text set
            "text_self_accuracy": ts_i2l, "text_self_accuracy_l2i": ts_l2i,  # class names (template mean)
            "image_self_accuracy": is_i2l, "image_self_accuracy_l2i": is_l2i,# dataset's own images
            "image_ref_accuracy": ir_i2l, "image_ref_accuracy_l2i": ir_l2i,  # general image set (imagenet)
            "image_accuracy": ir_i2l,               # back-compat alias (primary image pool)
            "pc_accuracy": pc_i2l, "pc_accuracy_l2i": pc_l2i,   # rank-r PC subspace (no example approx)
            "full_accuracy": full_i2l, "full_accuracy_l2i": full_l2i,  # real-model zero-shot
            "n_pcs": len(json_info.get("s", [])),   # rank of the single final-embedding decomposition
            "n_pcs_comp_mean": float(np.mean(comp_ranks)) if comp_ranks else float("nan"),  # mean kept-PC/unit
            "n_pcs_comp_std": float(np.std(comp_ranks)) if comp_ranks else float("nan"),    # std kept-PC/unit
            "n_comps": len(comp_ranks),             # number of decomposed heads+MLP units
            "image_ref_set": args.image_set,        # which pool fed image_ref
            **json_info,
        }
        jsonl_file.write(json.dumps(json_object))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
