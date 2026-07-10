"""
Sanity-check the residual-stream decomposition of BOTH CLIP encoders.

The saved per-head attention and per-layer MLP contributions are built so that
their sum reproduces the L2-normalized encoder output exactly (the final
LayerNorm is affine given the logged per-token mean/std, and the visual/text
projection is linear -- see utils/models/prs_hook{,_text}.py). This script
verifies that identity end-to-end:

    sum(saved attn) + sum(saved mlp)  ==  encode_*(x, normalize=True)

for the first `n_samples` inputs, by re-running the real CLIP forward pass on
the same inputs (in the same dataloader / text-dataset order the activations were
computed in) and comparing. A large error means the saved activations are stale
or the LayerNorm-linearization split is wrong.

Run the activation scripts first (they produce the .npy this reads):
    python -m utils.scripts.compute_activation_values      --model ... --dataset ...
    python -m utils.scripts.compute_activation_values_text --model ... --text_descriptions ...
"""
import os
import argparse
import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder

from utils.models.factory import create_model_and_transforms, get_tokenizer
from utils.datasets.binary_waterbirds import BinaryWaterbirds
from utils.datasets.dataset_helpers import dataset_to_dataloader


def _compare(recon, truth):
    """Per-sample comparison of two [n, d] embedding stacks. Returns a dict of metrics."""
    recon = recon.float()
    truth = truth.float()
    diff = (recon - truth).norm(dim=-1)
    rel_l2 = (diff / truth.norm(dim=-1)).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(recon, truth, dim=-1).cpu().numpy()
    abs_err = (recon - truth).abs()
    return {
        "n": recon.shape[0],
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "max_rel_l2": float(rel_l2.max()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
    }


def _report(name, metrics, tol):
    """Print metrics and return True if within tolerance."""
    ok = metrics["max_abs_err"] < tol
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name}: n={metrics['n']}  "
        f"max_abs_err={metrics['max_abs_err']:.2e}  mean_abs_err={metrics['mean_abs_err']:.2e}  "
        f"max_rel_l2={metrics['max_rel_l2']:.2e}  min_cos={metrics['min_cosine']:.6f}  "
        f"mean_cos={metrics['mean_cosine']:.6f}  (tol={tol:.0e})"
    )
    return ok


def verify_image_encoder(model, preprocess, attns, mlps, dataset, data_path, device,
                         n_samples, seed, samples_per_class, tot_samples_per_class,
                         batch_size=8, tol=1e-3):
    """
    Compare sum(saved attn/mlp)[:n] against the fresh normalized vision output on the
    same first-n images. `attns`,`mlps` are the saved arrays ([N,l,h,d], [N,l+1,d]).
    """
    n = min(n_samples, attns.shape[0])
    recon = (np.asarray(attns[:n]).sum(axis=(1, 2)) + np.asarray(mlps[:n]).sum(axis=1))
    recon = torch.from_numpy(recon).to(device)

    if dataset == "imagenet":
        ds = ImageNet(root=data_path + "imagenet/", split="val", transform=preprocess)
    elif dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=data_path + "waterbird_complete95_forest2water2/", split="test", transform=preprocess)
    elif dataset == "CIFAR100":
        ds = CIFAR100(root=data_path, download=True, train=False, transform=preprocess)
    elif dataset == "CIFAR10":
        ds = CIFAR10(root=data_path, download=True, train=False, transform=preprocess)
    else:
        ds = ImageFolder(root=data_path, transform=preprocess)

    # Same order the activations were saved in (compute_activation_values uses shuffle=False).
    dataloader = dataset_to_dataloader(
        ds, samples_per_class=samples_per_class, tot_samples_per_class=tot_samples_per_class,
        batch_size=batch_size, shuffle=False, num_workers=0, seed=seed,
    )

    truth = []
    seen = 0
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to(device)
            truth.append(model.encode_image(image, normalize=True).float().cpu())
            seen += image.shape[0]
            if seen >= n:
                break
    truth = torch.cat(truth, dim=0)[:n].to(device)
    return _compare(recon, truth), tol


def verify_text_encoder(model, tokenizer, attns, mlps, text_dir, text_descriptions, device,
                        n_samples, native_per_class, sentences_per_class, batch_size=64, tol=1e-3):
    """
    Compare sum(saved attn/mlp)[:n] against the fresh normalized text output on the
    same first-n dataset sentences (same last-k-per-class subsampling as the compute script).
    """
    n = min(n_samples, attns.shape[0])
    recon = (np.asarray(attns[:n]).sum(axis=(1, 2)) + np.asarray(mlps[:n]).sum(axis=1))
    recon = torch.from_numpy(recon).to(device)

    with open(os.path.join(text_dir, f"{text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]
    # Mirror compute_activation_values_text: keep the LAST k sentences of every native block.
    num_classes = len(lines) // native_per_class
    kept = [lines[r] for c in range(num_classes)
            for r in range((c + 1) * native_per_class - sentences_per_class, (c + 1) * native_per_class)]
    kept = kept[:n]

    truth = []
    with torch.no_grad():
        for start in range(0, len(kept), batch_size):
            tokens = tokenizer(kept[start:start + batch_size]).to(device)
            truth.append(model.encode_text(tokens, normalize=True).float().cpu())
    truth = torch.cat(truth, dim=0)[:n].to(device)
    return _compare(recon, truth), tol


def _model_pretrained(model_name):
    return {
        "ViT-H-14": "laion2b_s32b_b79k",
        "ViT-L-14": "laion2b_s32b_b82k",
        "ViT-B-16": "laion2b_s34b_b88k",
        "ViT-B-32": "laion2b_s34b_b79k",
    }.get(model_name, "laion2b_s34b_b79k")


def get_args_parser():
    parser = argparse.ArgumentParser("Verify residual-stream decomposition of both encoders", add_help=False)
    parser.add_argument("--model", default="ViT-B-32", type=str)
    parser.add_argument("--pretrained", default=None, type=str, help="default: laion tag for the model")
    parser.add_argument("--quantization", default="fp32", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_samples", default=16, type=int, help="how many inputs to re-run and compare")
    parser.add_argument("--tol", default=1e-3, type=float, help="max abs error allowed for a PASS")
    # image side
    parser.add_argument("--image_dataset", default="imagenet", type=str)
    parser.add_argument("--data_path", default="./datasets/", type=str)
    parser.add_argument("--samples_per_class", default=None, type=lambda v: None if v in ("None", "none", "") else int(v))
    parser.add_argument("--tot_samples_per_class", default=None, type=lambda v: None if v in ("None", "none", "") else int(v))
    # text side
    parser.add_argument("--text_dir", default="./utils/text_descriptions", type=str)
    parser.add_argument("--text_descriptions", default="imagenet_descriptions_personal", type=str)
    parser.add_argument("--native_per_class", default=10, type=int)
    parser.add_argument("--sentences_per_class", default=1, type=int)
    return parser


def main(args):
    pretrained = args.pretrained or _model_pretrained(args.model)
    img_attn = os.path.join(args.output_dir, f"{args.image_dataset}_attn_{args.model}_seed_{args.seed}.npy")
    img_mlp = os.path.join(args.output_dir, f"{args.image_dataset}_mlp_{args.model}_seed_{args.seed}.npy")
    txt_attn = os.path.join(args.output_dir, f"{args.text_descriptions}_attn_text_{args.model}_seed_{args.seed}.npy")
    txt_mlp = os.path.join(args.output_dir, f"{args.text_descriptions}_mlp_text_{args.model}_seed_{args.seed}.npy")

    have_img = os.path.exists(img_attn) and os.path.exists(img_mlp)
    have_txt = os.path.exists(txt_attn) and os.path.exists(txt_mlp)

    if not have_img:
        print("[SKIP] vision activations missing. Compute them first:\n"
              f"  python -m utils.scripts.compute_activation_values --model {args.model} "
              f"--pretrained {pretrained} --dataset {args.image_dataset} --seed {args.seed}")
    if not have_txt:
        print("[SKIP] text activations missing. Compute them first:\n"
              f"  python -m utils.scripts.compute_activation_values_text --model {args.model} "
              f"--pretrained {pretrained} --text_descriptions {args.text_descriptions} --seed {args.seed}")
    if not (have_img or have_txt):
        return 1

    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=pretrained, precision=args.quantization, cache_dir=args.cache_dir)
    model.to(args.device)
    model.eval()

    all_ok = True
    if have_img:
        attns = np.load(img_attn, mmap_mode="r")
        mlps = np.load(img_mlp, mmap_mode="r")
        metrics, tol = verify_image_encoder(
            model, preprocess, attns, mlps, args.image_dataset, args.data_path, args.device,
            args.n_samples, args.seed, args.samples_per_class, args.tot_samples_per_class, tol=args.tol)
        all_ok &= _report("VISION", metrics, tol)
    if have_txt:
        tokenizer = get_tokenizer(args.model)
        attns = np.load(txt_attn, mmap_mode="r")
        mlps = np.load(txt_mlp, mmap_mode="r")
        metrics, tol = verify_text_encoder(
            model, tokenizer, attns, mlps, args.text_dir, args.text_descriptions, args.device,
            args.n_samples, args.native_per_class, args.sentences_per_class, tol=args.tol)
        all_ok &= _report("TEXT  ", metrics, tol)

    print("\nAll decomposition checks passed." if all_ok else "\nSome checks FAILED (see above).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    raise SystemExit(main(args))
