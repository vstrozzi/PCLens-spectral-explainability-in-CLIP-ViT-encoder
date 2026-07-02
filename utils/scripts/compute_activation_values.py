"""
Adapted from https://github.com/yossigandelsman/clip_text_span.
MIT License Copyright (c) 2024 Yossi Gandelsman
"""
import numpy as np
import torch
import os
import glob
import argparse
from pathlib import Path
import re
import tqdm
from utils.models.factory import create_model_and_transforms
from utils.datasets.binary_waterbirds import BinaryWaterbirds
from utils.datasets.dataset_helpers import dataset_to_dataloader
from utils.models.prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

def parse_int_or_none(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream - Option B", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    parser.add_argument("--model", default="ViT-H-14", type=str, metavar="MODEL",
                        help="Name of model to use")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", type=str)
    # Dataset parameters
    parser.add_argument("--data_path", default="./datasets/", type=str, help="dataset path")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="imagenet, cub, waterbirds, CIFAR10, etc.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="cache directory for models weight")
    parser.add_argument("--samples_per_class", default=None, type=parse_int_or_none,
                        help="number of samples per class")

    parser.add_argument("--tot_samples_per_class", default=None, type=parse_int_or_none,
                        help="Total number of samples per class in the dataset")
    parser.add_argument("--max_nr_samples_before_writing", default=20, type=int,
                        help="How many samples to keep in RAM before saving them to chunk files")

    parser.add_argument("--quantization", help="Quantization size (choose 'fp16' or 'fp32')", default="fp32", type=str)
    parser.add_argument("--vision_proj", help="Project output down to text-img CLIP embedding space", default=True, type=str2bool)
    parser.add_argument("--full_output", help="Whether to output all the patch tokens and not only the CLS", default=False, type=str2bool)

    return parser


def main(args):
    """
    Calculates the projected residual stream (i.e. head activations) for a whole dataset 
    and saves them in separate chunk files. After finishing, it concatenates all chunks 
    into single .npy files (one per data type), and deletes the chunk files.
    """

    # Build & move model:
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, precision=args.quantization, cache_dir=args.cache_dir
    )
    model.to(args.device)
    model.eval()
    
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.device, spatial=False, vision_projection=args.vision_proj, full_output=args.full_output)
    # Dataset:
    if args.dataset == "imagenet":
        ds = ImageNet(root=args.data_path + "imagenet/", split="val", transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=args.data_path+"waterbird_complete95_forest2water2/", split="test", transform=preprocess)
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        ds = ImageFolder(root=args.data_path, transform=preprocess)

    # Depending
    dataloader = dataset_to_dataloader(
        ds,
        samples_per_class=args.samples_per_class,
        tot_samples_per_class=args.tot_samples_per_class,  # or whatever you prefer
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    num_total_images = len(dataloader) * args.batch_size
    print(f"We are using a dataset containing {num_total_images} images.")

    # We'll accumulate results here and flush them out every 
    # args.max_nr_samples_before_writing images (approx).
    attention_results = []
    mlp_results = []
    cls_to_cls_results = []
    labels_results = []

    # For chunk naming
    chunk_index = 0
    total_samples_seen = 0

    # We won't create final .npy until after the loop
    # Instead, we'll keep chunk-based filenames:
    # E.g. {dataset}_attn_{model}_seed_{seed}_chunk0.npy, chunk1.npy, ...
    chunk_attn_template = os.path.join(
    args.output_dir,
        f"{args.dataset}_attn{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy"
    )
    chunk_mlp_template = os.path.join(
        args.output_dir,
        f"{args.dataset}_mlp{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy"
    )
    chunk_cls_template = os.path.join(
        args.output_dir,
        f"{args.dataset}_cls_attn{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy"
    )
    chunk_labels_template = os.path.join(
        args.output_dir,
        f"{args.dataset}_labels{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy"
    )

    final_attn_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_attn{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}.npy"
    )
    final_mlp_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_mlp{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}.npy"
    )
    final_cls_attn_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_cls_attn{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}.npy"
    )
    final_labels_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_labels{'_full' if args.full_output else ''}_{args.model}_seed_{args.seed}.npy"
    )

    # Remove final files if they exist, just to be safe
    for ff in [final_attn_file, final_mlp_file, final_cls_attn_file, final_labels_file]:
        if os.path.exists(ff):
            os.remove(ff)

    def write_chunk_files(this_chunk_idx):
        #Save the arrays accumulated in attention_results, mlp_results, etc.
        #to chunk-based files, then clear them from memory.
        
        attn_filename   = chunk_attn_template.format(idx=this_chunk_idx)
        mlp_filename    = chunk_mlp_template.format(idx=this_chunk_idx)
        cls_filename    = chunk_cls_template.format(idx=this_chunk_idx)
        labels_filename = chunk_labels_template.format(idx=this_chunk_idx)

        with open(attn_filename, 'wb') as f:
            np.save(f, np.concatenate(attention_results, axis=0))
        with open(mlp_filename, 'wb') as f:
            np.save(f, np.concatenate(mlp_results, axis=0))
        with open(cls_filename, 'wb') as f:
            np.save(f, np.concatenate(cls_to_cls_results, axis=0))
        with open(labels_filename, 'wb') as f:
            np.save(f, np.concatenate(labels_results, axis=0))

        attention_results.clear()
        mlp_results.clear()
        cls_to_cls_results.clear()
        labels_results.clear()
        
    
    for i, (image, labels) in enumerate(tqdm.tqdm(dataloader)):

        batch_size_here = image.shape[0]
        total_samples_seen += batch_size_here

        with torch.no_grad():
            prs.reinit()
            # First, move the image to the GPU asynchronously.
            image = image.to(args.device, non_blocking=True)
            # Then, cast to lower precision on the GPU.
            if args.quantization == "fp16":
                image = image.to(dtype=torch.float16)
            representation = model.encode_image(image, attn_method= "head_no_spatial", normalize=False
            )
            
            attentions, mlps = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]
            mlps = mlps.detach().cpu().numpy()              # [b, l+1, d]

        # Accumulate in memory
        attention_results.append(attentions)  # reduce the spatial dimension
        mlp_results.append(mlps)
        cls_to_cls_results.append(attentions[:, :, 0]) # store the cls->cls attention, reduce heads
        labels_results.append(labels.cpu().numpy())

        # Check if we should dump to chunk files
        if total_samples_seen % args.max_nr_samples_before_writing == 0:
            write_chunk_files(chunk_index)
            chunk_index += 1

    # If there's anything left in memory after the loop, write one more chunk
    if len(attention_results) > 0:
        write_chunk_files(chunk_index)
        chunk_index += 1

    # AFTER THE LOOP: CONCATENATE CHUNK FILES
    print("\nConcatenating chunk files into final .npy arrays...")

    # Helper to load all chunk files of a certain type
    def natural_sort_key(s):
    # Split the string into a list of numeric and non-numeric parts.
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    def merge_chunks(template, final_file):
        # Merge chunk files into final_file without loading them all into RAM at once.
        chunk_files = sorted(glob.glob(template.format(idx='*')), key=natural_sort_key)
        print(chunk_files)

        shapes = []
        dtype = None
        for cf in chunk_files:
            arr = np.load(cf, mmap_mode="r")
            shapes.append(arr.shape)
            if dtype is None:
                dtype = arr.dtype

        total_rows = sum(s[0] for s in shapes)
        final_shape = (total_rows,) + shapes[0][1:]

        out = np.lib.format.open_memmap(final_file, mode="w+", dtype=dtype, shape=final_shape)
        offset = 0
        for cf, shape in zip(chunk_files, shapes):
            arr = np.load(cf, mmap_mode="r")
            out[offset:offset + shape[0]] = arr
            offset += shape[0]
        out.flush()
        del out

        return chunk_files

    # 1) Attn
    attn_chunk_files = merge_chunks(chunk_attn_template, final_attn_file)

    # 2) MLP
    mlp_chunk_files = merge_chunks(chunk_mlp_template, final_mlp_file)

    # 3) CLS->CLS attn
    cls_chunk_files = merge_chunks(chunk_cls_template, final_cls_attn_file)

    # 4) Labels
    label_chunk_files = merge_chunks(chunk_labels_template, final_labels_file)

    print("Final single-file arrays created:\n"
          f"  {final_attn_file}\n"
          f"  {final_mlp_file}\n"
          f"  {final_cls_attn_file}\n"
          f"  {final_labels_file}")

    
    # DELETE CHUNK FILES
    print("Deleting chunk files...")
    all_chunks = (
        attn_chunk_files +
        mlp_chunk_files +
        cls_chunk_files +
        label_chunk_files
    )
    for cf in all_chunks:
        os.remove(cf)
    print("Chunk files removed.")

    print("Done.")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
