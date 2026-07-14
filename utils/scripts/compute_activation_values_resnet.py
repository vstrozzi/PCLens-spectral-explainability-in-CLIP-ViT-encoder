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
    embeddings  [N, d]            = attn.sum((1,2)) + mlp.sum(1)
    poolattn    [N, H, K+1]       frozen class-token pooling weights a^h(I)

By default (--normalize True, matching the ViT/text pipeline) every component is divided
by ||visual(I)||, so the identity is
    attn.sum((1,2)) + mlp.sum(1) == visual(I)/||visual(I)|| == encode_image(I, normalize=True).
With --normalize False the components keep their raw scale and sum to the un-normalized
visual(I) exactly. Either way the runner asserts the raw decomposition is exact on the
first batch (verify_decomposition, before the optional normalization).

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument("--normalize", default=True, type=str2bool,
                        help="Divide every component by ||visual(image)|| so they sum to the "
                             "L2-normalized embedding, matching the ViT/text pipeline "
                             "(encode_image(normalize=True)). False keeps the exact raw decomposition "
                             "that sums to the un-normalized visual(image). Ignored when --vision_proj False.")
    parser.add_argument("--vision_proj", default=True, type=str2bool,
                        help="Apply the attnpool output projection W_o so components live in the "
                             "CLIP space (out_dim). False saves the pre-projection per-head value "
                             "stream (dim C//H, e.g. 64) with W_o factored out; the projection "
                             "(W_o, b_o) is written to {dataset}_out_proj_{model}_seed_{seed}.npz so "
                             "the embedding is recoverable. --normalize is ignored in this mode.")
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
    n_heads = visual.attnpool.num_heads
    out_dim = visual.attnpool.c_proj.out_features
    C = visual.attnpool.v_proj.in_features
    Wo, bo = visual.get_output_projection()  # attnpool output projection W_o [out_dim, C], b_o
    print(f"{args.model}: L+1={n_blocks} block components, H={n_heads} heads, "
          f"d={out_dim if args.vision_proj else C // n_heads}"
          f"{'' if args.vision_proj else ' (pre-projection; W_o factored out)'}")
    if not args.vision_proj:
        # Persist the factored-out projection so the embedding is recoverable without the model.
        np.savez(os.path.join(args.output_dir,
                              f"{args.dataset}_out_proj_{args.model}_seed_{args.seed}.npz"),
                 weight=Wo.detach().cpu().numpy(), bias=bo.detach().cpu().numpy())
        if args.normalize:
            print("  (--normalize ignored: no CLIP-space norm in pre-projection mode)")

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
        out = decompose_resnet_image(visual, image, check=(i == 0),  # assert exactness on 1st batch
                                     vision_proj=args.vision_proj)
        c_lh = out["c_lh"]                        # [B, L+1, H, d]  (d=out_dim proj, else dh)
        c_pos = out["c_pos"]                      # [B, d] (proj) or [B, H, dh] (pre-projection)
        if args.vision_proj:
            emb = c_lh.sum(dim=(1, 2)) + c_pos    # [B, out_dim] == visual(image)
        else:
            # Re-apply the factored-out projection to recover the embedding for saving.
            Wo_r = Wo.reshape(out_dim, n_heads, C // n_heads)      # [out_dim, H, dh]
            pooled = c_lh.sum(dim=1) + c_pos                       # [B, H, dh]
            emb = torch.einsum("bhd,ohd->bo", pooled, Wo_r) + bo[None, :]
        if i == 0:
            e = verify_decomposition(visual, image, verbose=True, vision_proj=args.vision_proj)
            print(f"  first-batch sum-recovery max abs err = {e:.2e}")
        if args.normalize and args.vision_proj:
            # Per-sample scalar: sum(components)=emb, so dividing each by ||emb|| makes them
            # sum to emb/||emb|| == encode_image(image, normalize=True).
            norm = emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, 1]
            c_lh = c_lh / norm[:, :, None, None]
            c_pos = c_pos / norm
            emb = emb / norm

        buffers["attn"].append(c_lh.cpu().numpy().astype(store_dtype))
        buffers["mlp"].append(c_pos[:, None].cpu().numpy().astype(store_dtype))
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
