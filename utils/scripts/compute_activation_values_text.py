"""
Text-encoder counterpart of utils/scripts/compute_activation_values.py.

Runs the CLIP *text* encoder over a dataset of text descriptions and saves the
per-head attention and per-layer MLP contributions to the EOS ("eot") token,
each projected into the shared CLIP space, implementing

    M_text(t) = P~_txt Z0^eot + sum_l sum_h sum_i c_{i,l,h}
              + sum_l P~_txt [MLP_l(LN_l(Z_l))]^eot .

Saves (same shapes/conventions as the vision pipeline, with a `_text` tag):
    {dataset}_attn_text_{model}_seed_{seed}.npy    [N, l, h, d]   (or [N, l, m, h, d] if --spatial)
    {dataset}_mlp_text_{model}_seed_{seed}.npy      [N, l + 1, d]
    {dataset}_labels_text_{model}_seed_{seed}.npy   [N]            (line index in the dataset)

The summed-per-head form ([N, l, h, d]) satisfies the exact identity
    M_text = attn.sum(axis=(1, 2)) + mlp.sum(axis=1)
so the downstream svd_data_approx / mean-ablation reconstruction code is reused
unchanged. --spatial keeps the per-source-token axis (large; intended for small
datasets / debugging only).

Adapted from https://github.com/yossigandelsman/clip_text_span. MIT License
Copyright (c) 2024 Yossi Gandelsman.
"""
import numpy as np
import torch
import os
import glob
import re
import argparse
from pathlib import Path
import tqdm
from utils.models.factory import create_model_and_transforms, get_tokenizer
from utils.models.prs_hook_text import hook_prs_logger_text


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
    parser = argparse.ArgumentParser("Project Residual Stream - Text encoder", add_help=False)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--model", default="ViT-B-32", type=str, metavar="MODEL",
                        help="Name of model to use")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", type=str)
    parser.add_argument("--text_dir", default="./utils/text_descriptions", type=str,
                        help="Folder holding the text dataset .txt files")
    parser.add_argument("--text_descriptions", default="top_1500_nouns_5_sentences_imagenet_clean",
                        type=str, help="Name (without .txt) of the text dataset to decompose")
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="cache directory for model weights")
    parser.add_argument("--max_nr_samples_before_writing", default=2000, type=int,
                        help="How many samples to keep in RAM before flushing to chunk files")
    parser.add_argument("--native_per_class", default=1, type=int,
                        help="Number of consecutive sentences per class present in the dataset "
                             "(imagenet_descriptions_personal has 10, in class order). "
                             "Default 1 -> the dataset has no class structure (every row a class).")
    parser.add_argument("--sentences_per_class", default=1, type=int,
                        help="How many sentences to keep per class: the LAST k of each native "
                             "block (the last one is the concise class-name-like summary). "
                             "So --sentences_per_class 1 keeps only the class-name sentence per "
                             "class; k == native_per_class keeps all of them. Label = class index.")
    parser.add_argument("--quantization", default="fp32", type=str, help="'fp16' or 'fp32'")
    parser.add_argument("--text_proj", default=True, type=str2bool,
                        help="Project component outputs into the shared CLIP space")
    parser.add_argument("--spatial", default=False, type=str2bool,
                        help="Keep the per-source-token axis (memory heavy: [N, l, m, h, d])")
    return parser


def main(args):
    # Build & move model:
    model, _, _ = create_model_and_transforms(
        args.model, pretrained=args.pretrained, precision=args.quantization, cache_dir=args.cache_dir
    )
    model.to(args.device)
    model.eval()
    tokenizer = get_tokenizer(args.model)

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", model.context_length)
    print("Number of text layers:", len(model.transformer.resblocks))

    attn_method = "head" if args.spatial else "head_no_spatial"
    prs = hook_prs_logger_text(model, args.device, spatial=args.spatial, text_projection=args.text_proj)

    # Load the text dataset:
    with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]

    # Keep the LAST `sentences_per_class` sentences of every native class block; the last sentence
    # of each block is the concise class-name-like summary, so k == 1 -> one class-name per class.
    native = args.native_per_class
    k = args.sentences_per_class
    assert k <= native, f"--sentences_per_class ({k}) must be <= --native_per_class ({native})"
    num_classes = len(lines) // native
    kept_lines, kept_labels = [], []
    for c in range(num_classes):
        end = (c + 1) * native
        for r in range(end - k, end):
            kept_lines.append(lines[r])
            kept_labels.append(c)
    lines = kept_lines
    labels_all = np.array(kept_labels)
    print(f"Decomposing {len(lines)} texts from '{args.text_descriptions}' "
          f"({num_classes} classes x {k} of {native} sentences/class).")

    attention_results = []
    mlp_results = []
    labels_results = []
    chunk_index = 0
    total_samples_seen = 0

    tag = "_text"
    chunk_attn_template = os.path.join(
        args.output_dir, f"{args.text_descriptions}_attn{tag}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy")
    chunk_mlp_template = os.path.join(
        args.output_dir, f"{args.text_descriptions}_mlp{tag}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy")
    chunk_labels_template = os.path.join(
        args.output_dir, f"{args.text_descriptions}_labels{tag}_{args.model}_seed_{args.seed}_chunk{{idx}}.npy")

    final_attn_file = os.path.join(
        args.output_dir, f"{args.text_descriptions}_attn{tag}_{args.model}_seed_{args.seed}.npy")
    final_mlp_file = os.path.join(
        args.output_dir, f"{args.text_descriptions}_mlp{tag}_{args.model}_seed_{args.seed}.npy")
    final_labels_file = os.path.join(
        args.output_dir, f"{args.text_descriptions}_labels{tag}_{args.model}_seed_{args.seed}.npy")

    for ff in [final_attn_file, final_mlp_file, final_labels_file]:
        if os.path.exists(ff):
            os.remove(ff)

    def write_chunk_files(this_chunk_idx):
        with open(chunk_attn_template.format(idx=this_chunk_idx), 'wb') as f:
            np.save(f, np.concatenate(attention_results, axis=0))
        with open(chunk_mlp_template.format(idx=this_chunk_idx), 'wb') as f:
            np.save(f, np.concatenate(mlp_results, axis=0))
        with open(chunk_labels_template.format(idx=this_chunk_idx), 'wb') as f:
            np.save(f, np.concatenate(labels_results, axis=0))
        attention_results.clear()
        mlp_results.clear()
        labels_results.clear()

    cast_dtype = model.transformer.get_cast_dtype()
    for start in tqdm.trange(0, len(lines), args.batch_size):
        batch = lines[start:start + args.batch_size]
        labels = labels_all[start:start + len(batch)]  # class index per kept row
        total_samples_seen += len(batch)

        with torch.no_grad():
            prs.reinit()
            text_tokens = tokenizer(batch).to(args.device)
            prs.eos_idx = text_tokens.argmax(dim=-1)  # [b] EOS position per sample
            # Initial residual Z0 = token + positional embedding (no hook point for it)
            z0 = model.token_embedding(text_tokens).to(cast_dtype)
            z0 = z0 + model.positional_embedding.to(cast_dtype)
            # The registered hooks populate prs.attentions / prs.mlps / post_ln during this call
            representation = model.encode_text(text_tokens, normalize=False, attn_method=attn_method)
            # Prepend the projected initial term as the first MLP slot (mirrors visual.ln_pre_post)
            prs.mlps = [prs._gather_eos(z0).detach().cpu()] + prs.mlps

            attentions, mlps = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # [N, l, (m,) h, d]
            mlps = mlps.detach().cpu().numpy()               # [N, l + 1, d]

        attention_results.append(attentions)
        mlp_results.append(mlps)
        labels_results.append(labels)

        if total_samples_seen % args.max_nr_samples_before_writing < args.batch_size:
            write_chunk_files(chunk_index)
            chunk_index += 1

    if len(attention_results) > 0:
        write_chunk_files(chunk_index)
        chunk_index += 1

    print("\nConcatenating chunk files into final .npy arrays...")

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def merge_chunks(template, final_file):
        # Merge chunk files into final_file without loading them all into RAM at once.
        chunk_files = sorted(glob.glob(template.format(idx='*')), key=natural_sort_key)

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

        return out.shape, chunk_files

    attn_shape, attn_chunk_files = merge_chunks(chunk_attn_template, final_attn_file)
    mlp_shape, mlp_chunk_files = merge_chunks(chunk_mlp_template, final_mlp_file)
    labels_shape, label_chunk_files = merge_chunks(chunk_labels_template, final_labels_file)

    print("Final single-file arrays created:\n"
          f"  {final_attn_file}  shape {attn_shape}\n"
          f"  {final_mlp_file}  shape {mlp_shape}\n"
          f"  {final_labels_file}  shape {labels_shape}")

    for cf in attn_chunk_files + mlp_chunk_files + label_chunk_files:
        os.remove(cf)
    print("Chunk files removed. Done.")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
