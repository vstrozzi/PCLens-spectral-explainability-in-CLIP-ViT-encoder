"""
Compute the exact per-input layer x head decomposition of a CLIP ModifiedResNet
(RN50 / RN101) over a dataset and save it in the SAME on-disk layout the ViT
pipeline uses, so the existing analysis tooling (get_data, svd_data_approx,
reconstruct_*_mean_ablation_*, visualize_principal_component, ...) can be reused
unchanged with model = "RN50".

Saved arrays ({dataset}_{type}_{model}_seed_{seed}.npy):
    attn        [N, L+1, H, d]   c_{l,h}(I)   (l=0 stem .. L blocks; maps onto ViT [N,l,h,d])
    mlp         [N,   1,   d]     sum_h c_{P,h}(I) + out-proj bias (content-free positional term)
    labels      [N]
    embeddings  [N, d]            = attn.sum((1,2)) + mlp.sum(1) = visual(I)  (exact)
    poolattn    [N, H, K+1]       frozen class-token pooling weights a^h(I)

The identity  attn.sum((1,2)) + mlp.sum(1) == visual(I)  holds to numerical precision
(the runner asserts it on the first batch).

Adapted from utils/scripts/compute_activation_values.py (Gandelsman et al., MIT).
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, ImageNet

from utils.datasets.binary_waterbirds import BinaryWaterbirds
from utils.datasets.dataset_helpers import dataset_to_dataloader
from utils.models.factory import create_model_and_transforms
from utils.models.resnet_prs import decompose_resnet_image, verify_decomposition


def parse_int_or_none(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def get_args_parser():
    parser = argparse.ArgumentParser("ResNet PRS - exact layer x head decomposition", add_help=False)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--model", default="RN50", type=str, help="RN50 or RN101")
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--data_path", default="./datasets/", type=str)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--samples_per_class", default=5, type=parse_int_or_none)
    parser.add_argument("--tot_samples_per_class", default=50, type=parse_int_or_none)
    parser.add_argument("--max_nr_samples_before_writing", default=500, type=int)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="storage dtype for the component arrays")
    return parser


def main(args):
    # The decomposition disables TF32 internally; also disable it globally so any
    # incidental full-model forward matches the fp32 components.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, precision="fp32", cache_dir=args.cache_dir)
    model.to(args.device).eval()
    visual = model.visual
    assert hasattr(visual, "attnpool"), f"{args.model} is not a ModifiedResNet CLIP model"
    n_blocks = 1 + sum(len(getattr(visual, f"layer{i}")) for i in range(1, 5))
    print(f"{args.model}: L+1={n_blocks} block components, H={visual.attnpool.num_heads} heads, "
          f"d={visual.attnpool.c_proj.out_features}")

    if args.dataset == "imagenet":
        ds = ImageNet(root=args.data_path + "imagenet/", split="val", transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=args.data_path + "waterbird_complete95_forest2water2/",
                              split="test", transform=preprocess)
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(root=args.data_path, download=True, train=False, transform=preprocess)
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(root=args.data_path, download=True, train=False, transform=preprocess)
    else:
        ds = ImageFolder(root=args.data_path, transform=preprocess)

    dataloader = dataset_to_dataloader(
        ds, samples_per_class=args.samples_per_class,
        tot_samples_per_class=args.tot_samples_per_class,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    print(f"Dataset: {len(dataloader.dataset)} images in {len(dataloader)} batches.")

    store_dtype = np.float16 if args.dtype == "float16" else np.float32
    buffers = {k: [] for k in ["attn", "mlp", "labels", "embeddings", "poolattn"]}

    def tmpl(kind):
        return os.path.join(args.output_dir,
                            f"{args.dataset}_{kind}_{args.model}_seed_{args.seed}")

    chunk_index = 0
    total_seen = 0
    buffered = 0

    for final in ["attn", "mlp", "labels", "embeddings", "poolattn"]:
        f = tmpl(final) + ".npy"
        if os.path.exists(f):
            os.remove(f)

    def write_chunk(idx):
        for kind in buffers:
            with open(f"{tmpl(kind)}_chunk{idx}.npy", "wb") as f:
                np.save(f, np.concatenate(buffers[kind], axis=0))
            buffers[kind].clear()
        return 0  # reset buffered count

    for i, (image, labels) in enumerate(tqdm.tqdm(dataloader)):
        total_seen += image.shape[0]
        buffered += image.shape[0]
        image = image.to(args.device, non_blocking=True)
        out = decompose_resnet_image(visual, image, check=(i == 0))  # assert exactness on 1st batch
        c_lh = out["c_lh"]                       # [B, L+1, H, d]
        c_pos = out["c_pos"]                     # [B, d]
        emb = c_lh.sum(dim=(1, 2)) + c_pos       # [B, d] == visual(image)
        if i == 0:
            e = verify_decomposition(visual, image, verbose=True)
            print(f"  first-batch sum-recovery max abs err = {e:.2e}")

        buffers["attn"].append(c_lh.cpu().numpy().astype(store_dtype))
        buffers["mlp"].append(c_pos[:, None, :].cpu().numpy().astype(store_dtype))
        buffers["labels"].append(labels.cpu().numpy())
        buffers["embeddings"].append(emb.cpu().numpy().astype(store_dtype))
        buffers["poolattn"].append(out["attn"].cpu().numpy().astype(store_dtype))

        if buffered >= args.max_nr_samples_before_writing:
            buffered = write_chunk(chunk_index); chunk_index += 1

    if len(buffers["labels"]) > 0:
        write_chunk(chunk_index); chunk_index += 1

    # Merge chunks -> single memmap arrays, then delete chunks.
    print("\nConcatenating chunk files...")

    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    def merge(kind):
        chunk_files = sorted(glob.glob(f"{tmpl(kind)}_chunk*.npy"), key=natural_sort_key)
        shapes = [np.load(cf, mmap_mode="r").shape for cf in chunk_files]
        dtype = np.load(chunk_files[0], mmap_mode="r").dtype
        final_shape = (sum(s[0] for s in shapes),) + shapes[0][1:]
        out_arr = np.lib.format.open_memmap(f"{tmpl(kind)}.npy", mode="w+", dtype=dtype, shape=final_shape)
        off = 0
        for cf, s in zip(chunk_files, shapes):
            out_arr[off:off + s[0]] = np.load(cf, mmap_mode="r"); off += s[0]
        out_arr.flush(); del out_arr
        for cf in chunk_files:
            os.remove(cf)

    for kind in buffers:
        merge(kind)
    print("Done. Saved:", ", ".join(f"{tmpl(k)}.npy" for k in buffers))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
