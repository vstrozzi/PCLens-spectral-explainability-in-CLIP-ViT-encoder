"""
Extracts the residual-stream decomposition (per-head attn
+ per-layer MLP components) of BOTH CLIP encoders for a chosen set of models over a
chosen set of datasets, then verifies the decomposition.

Reads the model / dataset / seed defaults from experiments_info.json, asks what to run,
and then -- for every (model, dataset):

  * image tower : utils.scripts.compute_activation_values           (ViT)
                  utils.scripts.compute_activation_values_resnet     (RN50 / RN101)
  * text tower  : utils.scripts.compute_activation_values_text
                  run over (a) each dataset's class names and (b) any extra
                  sentence sets from utils/text_descriptions the user picks.
  * verify      : utils.scripts.verify_decomposition_activations
                  re-runs 5 samples through the real CLIP forward and checks
                  sum(components) == encode_*(x). By default all three towers
                  (ViT, ResNet, text) reconstruct the L2-normalized output.

Output in
    {output_dir}/activations_and_datasets_idxs_{seed}/
together with, for every dataset subset, an idx->class map so the exact images /
classes that were decomposed can be recovered, and a parameters.jsonl describing
the whole run.

Run from the repo root:
    python -m utils.scripts.extract_activations

Flow:
  1. pick models (default: all in experiments_info.models_config);
  2. pick datasets (only those already downloaded under ./datasets/);
  3. per dataset: show min/avg/max samples-per-class, ask elements_per_class
     (default: experiments_info) -> a class-balanced subset;
  4. pick extra sentence sets for the text tower (default: none);
  5. pick the seed (default: experiments_info.seed);
  6. optionally tweak the compute_activation_values knobs (L2/projection,
     quantization, full_output, last_layers_only, save_dtype, batch sizes, ...);
  7. pick the CUDA device, or run one-model-per-GPU in parallel (default: off);
  8. run everything and verify.
"""
import json
import os
import queue
import subprocess
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INFO_JSON = ROOT / "experiments_info.json"
DATASETS_DIR = ROOT / "datasets"
TEXT_DIR = ROOT / "utils" / "text_descriptions"
CALTECH_DIR = DATASETS_DIR / "caltech101" / "101_ObjectCategories"

PY = sys.executable  # same interpreter (MT env) for the delegated `-m` calls

# experiments_info.json is the single hub: it lists the models (with their slug /
# pretrained tag / type) and the datasets (with their pipeline dataset arg, data
# path, presence folder and download size). Everything below reads from it.
_INFO = json.loads(INFO_JSON.read_text())
MODELS = {m["slug"]: m for m in _INFO["models_config"]}
DATASETS = {d["name"]: d for d in _INFO["datsets_config"]}
DATASET_ORDER = list(DATASETS)


# --------------------------------------------------------------------------- #
# model conventions (all sourced from experiments_info.models_config)
# --------------------------------------------------------------------------- #
def model_slug(name: str) -> str:
    """experiments_info model name ('ViT-B/32', 'RN50') -> model_configs slug ('ViT-B-32')."""
    return next(m["slug"] for m in MODELS.values() if m["name"] == name)


def is_resnet(slug: str) -> bool:
    return MODELS[slug]["type"] == "resnet"


def pretrained_for(slug: str) -> str:
    return MODELS[slug]["pretrained"]


# --------------------------------------------------------------------------- #
# dataset registry: presence + how to build the *un-transformed* dataset object
# (for label counts / subset indices). The metadata (dataset_arg, data_path,
# folder) lives in the hub; only the actual torchvision/VisionDataset
# constructors and class-name lists (which can't be JSON) live here. The build
# must match exactly what the compute scripts build, so the saved idx->class map
# lines up with the saved activations.
# --------------------------------------------------------------------------- #
def is_present(name: str) -> bool:
    """Downloaded? Uses the hub `folder`; also accepts the CIFAR tarball (torchvision extracts it)."""
    if (DATASETS_DIR / DATASETS[name]["folder"]).exists():
        return True
    return name.startswith("CIFAR") and (DATASETS_DIR / f"{name.lower()}-python.tar.gz").exists()


def build_dataset(name: str, fairface_label: str = "gender"):
    """The un-transformed dataset object, built the same way the compute scripts build it."""
    from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, ImageNet

    if name == "CIFAR-10":
        return CIFAR10(root="./datasets/", train=False, download=True)
    if name == "CIFAR-100":
        return CIFAR100(root="./datasets/", train=False, download=True)
    if name == "ImageNet":
        return ImageNet(root="./datasets/imagenet/", split="val")
    if name == "Caltech-101":
        return ImageFolder(root=str(CALTECH_DIR))
    if name == "waterbird":
        from utils.datasets.binary_waterbirds import BinaryWaterbirds
        return BinaryWaterbirds(root="./datasets/waterbird_complete95_forest2water2/", split="test")
    if name == "fairface":
        from utils.datasets.fairface import FairFace
        return FairFace(root="./datasets/fairface/", split="val", label=fairface_label)
    raise ValueError(name)


def class_names(name: str, fairface_label: str = "gender"):
    from utils.datasets_constants.cifar_10_classes import cifar_10_classes
    from utils.datasets_constants.cifar_100_classes import cifar_100_classes
    from utils.datasets_constants.imagenet_classes import imagenet_classes
    from utils.datasets_constants.caltech_classes import caltech_101_classes
    from utils.datasets_constants.waterbird_classes import waterbird_classes
    from utils.datasets_constants.fairface_classes import FAIRFACE_CLASSES

    if name == "fairface":
        return list(FAIRFACE_CLASSES[fairface_label])
    return {
        "CIFAR-10": cifar_10_classes,
        "CIFAR-100": cifar_100_classes,
        "ImageNet": imagenet_classes,
        "Caltech-101": caltech_101_classes,
        "waterbird": waterbird_classes,
    }[name]


def dataset_labels(ds):
    if hasattr(ds, "targets"):
        return [int(t) for t in ds.targets]
    return [int(s[1]) for s in ds.samples]


# --------------------------------------------------------------------------- #
# interactive helpers (mirroring download_datasets.py)
# --------------------------------------------------------------------------- #
def _ask(prompt: str, default: str) -> str:
    resp = input(f"{prompt} [{default}]: ").strip()
    return resp or default


def _ask_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    return _ask(prompt, d).lower() in ("y", "yes", "true", "t", "1")


def _ask_int(prompt: str, default):
    resp = _ask(prompt, "None" if default is None else str(default))
    if resp.lower() in ("none", ""):
        return None
    return int(resp)


def _match(user_names, known):
    """Comma-separated user input -> canonical names (case-insensitive)."""
    lut = {n.lower(): n for n in known}
    out = []
    for raw in user_names:
        key = raw.strip().lower()
        if key in lut:
            out.append(lut[key])
        elif key:
            print(f"  (ignoring unknown '{raw.strip()}')")
    return out


def _pick(prompt, known, default="all"):
    """Ask for 'all' / comma-separated subset of `known`; return list in `known` order."""
    sel = _ask(f"{prompt} (comma-separated or 'all')", default)
    if sel.lower() == "all":
        return list(known)
    if sel.lower() in ("none", ""):
        return []
    chosen = _match(sel.split(","), known)
    return [n for n in known if n in chosen]


# --------------------------------------------------------------------------- #
# command builders (delegate to the existing pipeline scripts)
# --------------------------------------------------------------------------- #
def cmd_image(slug, name, device, seed, spc, tot, out, opt):
    d = DATASETS[name]
    if is_resnet(slug):
        return [PY, "-m", "utils.scripts.compute_activation_values_resnet",
                "--model", slug, "--pretrained", pretrained_for(slug),
                "--dataset", d["dataset_arg"], "--data_path", d["data_path"],
                "--seed", str(seed), "--device", device, "--output_dir", str(out),
                "--samples_per_class", str(spc), "--tot_samples_per_class", str(tot),
                "--batch_size", str(opt.batch_size),
                "--vision_proj", str(opt.vision_proj), "--normalize", str(opt.vision_proj),
                "--dtype", "float16" if opt.save_dtype == "fp16" else "float32",
                "--max_nr_samples_before_writing", str(opt.max_write)]
    c = [PY, "-m", "utils.scripts.compute_activation_values",
         "--model", slug, "--pretrained", pretrained_for(slug),
         "--dataset", d["dataset_arg"], "--data_path", d["data_path"],
         "--seed", str(seed), "--device", device, "--output_dir", str(out),
         "--samples_per_class", str(spc), "--tot_samples_per_class", str(tot),
         "--num_workers", str(opt.num_workers), "--batch_size", str(opt.batch_size),
         "--quantization", opt.quantization, "--vision_proj", str(opt.vision_proj),
         "--full_output", str(opt.full_output), "--save_dtype", opt.save_dtype,
         "--max_nr_samples_before_writing", str(opt.max_write)]
    if opt.last_layers_only is not None:
        c += ["--last_layers_only", str(opt.last_layers_only)]
    if name == "fairface":
        c += ["--fairface_label", opt.fairface_label]
    return c


def cmd_text(slug, text_dir, text_name, device, seed, out, opt, native=1, sentences=1):
    return [PY, "-m", "utils.scripts.compute_activation_values_text",
            "--model", slug, "--pretrained", pretrained_for(slug),
            "--text_dir", str(text_dir), "--text_descriptions", text_name,
            "--seed", str(seed), "--device", device, "--output_dir", str(out),
            "--batch_size", str(opt.text_batch_size), "--quantization", opt.quantization,
            "--text_proj", str(opt.vision_proj),
            "--native_per_class", str(native), "--sentences_per_class", str(sentences)]


def cmd_verify(slug, device, seed, out, image_name, text_dir, text_name, opt,
               spc=None, tot=None, native=1, sentences=1, data_path="./datasets/"):
    c = [PY, "-m", "utils.scripts.verify_decomposition_activations",
         "--model", slug, "--pretrained", pretrained_for(slug),
         "--seed", str(seed), "--device", device, "--output_dir", str(out),
         "--n_samples", "5", "--tol", str(opt.tol),
         "--image_dataset", image_name, "--data_path", data_path,
         "--image_normalize", str(opt.vision_proj),
         "--text_dir", str(text_dir), "--text_descriptions", text_name,
         "--native_per_class", str(native), "--sentences_per_class", str(sentences)]
    if spc is not None:
        c += ["--samples_per_class", str(spc), "--tot_samples_per_class", str(tot)]
    if image_name == "fairface":
        c += ["--fairface_label", opt.fairface_label]
    return c


# --------------------------------------------------------------------------- #
# per-model runner (delegated subprocesses are sequential within a model)
# --------------------------------------------------------------------------- #
def run_model(slug, device, cfg):
    tag = f"[{slug} @ {device}]"
    print(f"\n{'=' * 70}\n{tag} starting\n{'=' * 70}")

    def sh(cmd):
        print(f"{tag} >> {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"{tag} !! command exited with code {rc}")
        return rc

    out, seed, opt = cfg.out, cfg.seed, cfg.opt
    # 1) activations
    for name in cfg.datasets:
        if is_resnet(slug) and name == "fairface":
            print(f"{tag} skipping fairface (unsupported by the ResNet activation script).")
            continue
        d = DATASETS[name]
        spc, tot = cfg.subset[name]["spc"], cfg.subset[name]["tot"]
        sh(cmd_image(slug, name, device, seed, spc, tot, out, opt))
        # text tower over this dataset's class names (one class per line)
        sh(cmd_text(slug, out, f"{d['dataset_arg']}_classnames", device, seed, out, opt))
    for tset in cfg.text_sets:
        sh(cmd_text(slug, TEXT_DIR, tset, device, seed, out, opt))

    # 2) verify (5 re-run samples; both towers reconstruct the normalized output by default)
    if not opt.vision_proj:
        print(f"\n{tag} skipping verify: un-projected components do not reconstruct encode_*(x).")
        print(f"{tag} done.")
        return
    print(f"\n{tag} verifying decomposition ...")
    for name in cfg.datasets:
        if is_resnet(slug) and name == "fairface":
            continue
        d = DATASETS[name]
        spc, tot = cfg.subset[name]["spc"], cfg.subset[name]["tot"]
        sh(cmd_verify(slug, device, seed, out, d["dataset_arg"], out, f"{d['dataset_arg']}_classnames",
                      opt, spc=spc, tot=tot, data_path=d["data_path"]))
    for tset in cfg.text_sets:
        # text-only: a non-existent image_dataset makes verify SKIP the image side.
        sh(cmd_verify(slug, device, seed, out, "__none__", TEXT_DIR, tset, opt))
    print(f"{tag} done.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    all_models = [m["name"] for m in _INFO["models_config"]]
    ds_defaults = DATASETS  # name -> full hub entry (carries element_per_class)
    default_seed = _INFO.get("seed", 0)

    # -- models --
    print("Available models:", ", ".join(all_models))
    models = _pick("Which models to extract activations for?", all_models)
    if not models:
        print("No models selected. Exiting.")
        return
    slugs = [model_slug(m) for m in models]

    # -- datasets (only downloaded ones) --
    available = [n for n in DATASET_ORDER if is_present(n)]
    if not available:
        print("No datasets found under ./datasets/. Run utils.datasets.download_datasets first.")
        return
    print("\nDownloaded datasets:", ", ".join(available))
    datasets = _pick("Which datasets?", available)
    if not datasets:
        print("No datasets selected. Exiting.")
        return

    # FairFace attribute (needed before we can count its classes) --
    fairface_label = "gender"
    if "fairface" in datasets:
        fairface_label = _ask("FairFace label attribute (gender/race/age)", "gender")

    # -- per-dataset subset sizes --
    print("\nBuilding datasets to report per-class availability (this can take a moment) ...")
    subset = {}
    ds_objs = {}
    for name in datasets:
        ds = build_dataset(name, fairface_label)
        ds_objs[name] = ds
        labels = dataset_labels(ds)
        counts = Counter(labels)
        n_classes = len(ds.classes)
        c = np.array([counts.get(i, 0) for i in range(n_classes)])
        tot = len(ds) // n_classes  # contiguous block size used by dataset_subset
        default_epc = ds_defaults.get(name, {}).get("element_per_class", min(int(c.min()), tot))
        print(f"\n{name}: {n_classes} classes, {len(ds)} images  |  per class -> "
              f"min {int(c.min())}, avg {c.mean():.1f}, max {int(c.max())}  "
              f"(subset block size = {tot})")
        epc = _ask_int(f"  elements_per_class for {name}", default_epc)
        if epc is not None and epc > tot:
            print(f"  (clamping {epc} -> {tot}, the max the balanced sampler can draw)")
            epc = tot
        subset[name] = {"spc": epc, "tot": tot, "n_classes": n_classes,
                        "min": int(c.min()), "avg": float(c.mean()), "max": int(c.max())}

    # -- extra sentence sets for the text tower --
    text_files = sorted(p.stem for p in TEXT_DIR.glob("*.txt"))
    print("\nText-tower sentence sets available in utils/text_descriptions:")
    print("  " + ", ".join(text_files))
    print("(the text tower is ALWAYS also decomposed over each dataset's class names)")
    text_sets = _pick("Also decompose these sentence sets?", text_files, default="none")

    # -- seed --
    seed = _ask_int("\nSeed", default_seed)

    # -- compute_activation_values knobs --
    opt = types.SimpleNamespace(
        batch_size=8, text_batch_size=256, num_workers=4, quantization="fp32",
        vision_proj=True, full_output=False, last_layers_only=None, save_dtype="fp32",
        max_write=200, tol=1e-3, fairface_label=fairface_label,
    )
    # Normalize/project the final output into the shared CLIP space (L2). Default yes;
    # this is the setting under which the verify step reconstructs the true embedding.
    opt.vision_proj = _ask_bool("\nNormalize the final output (project to shared CLIP space, L2)?", True)
    if _ask_bool("Customize the remaining compute options (quantization, layers, batch sizes, ...)?", False):
        opt.quantization = _ask("  Model quantization (fp16/fp32)", opt.quantization)
        opt.save_dtype = _ask("  Saved-array dtype (fp16/fp32)", opt.save_dtype)
        opt.full_output = _ask_bool("  Keep all patch tokens (full_output)", opt.full_output)
        opt.last_layers_only = _ask_int("  Save only last k layers (None = all)", opt.last_layers_only)
        opt.batch_size = _ask_int("  Image batch size", opt.batch_size)
        opt.text_batch_size = _ask_int("  Text batch size", opt.text_batch_size)
        opt.num_workers = _ask_int("  DataLoader workers", opt.num_workers)
        opt.max_write = _ask_int("  Samples kept in RAM before flushing", opt.max_write)
    if opt.last_layers_only is not None:
        print("  NOTE: last_layers_only truncates the residual stream, so the verify step "
              "will not reconstruct the full output (expect it to FAIL by design).")

    # -- device / parallelism --
    import torch
    if not torch.cuda.is_available():
        print("\nCUDA is not available. This pipeline requires a CUDA device. Exiting.")
        return
    n_gpu = torch.cuda.device_count()
    names = ", ".join(f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(n_gpu))
    print(f"\nCUDA devices ({n_gpu}): {names}")
    parallel = n_gpu > 1 and _ask_bool("Run models in parallel, one per GPU?", False)
    if parallel:
        devices = [f"cuda:{i}" for i in range(n_gpu)]
    else:
        dev = _ask("Which CUDA device?", "cuda:0")
        devices = [dev]

    # -- output dir + idx->class maps + parameters.jsonl --
    out = ROOT / "output_dir" / f"activations_and_datasets_idxs_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    from utils.datasets.dataset_helpers import dataset_subset
    print("\nWriting subset idx->class maps + class-name files ...")
    for name in datasets:
        d = DATASETS[name]
        ds = ds_objs[name]
        labels = dataset_labels(ds)
        names_list = class_names(name, fairface_label)
        # class-name text file for the text-tower decomposition (index == label)
        (out / f"{d['dataset_arg']}_classnames.txt").write_text(
            "\n".join(str(x) for x in names_list) + "\n")
        # deterministic balanced subset -- identical to what the compute scripts draw
        spc, tot = subset[name]["spc"], subset[name]["tot"]
        if spc is None:
            idx = list(range(len(ds)))
        else:
            idx = dataset_subset(ds, samples_per_class=spc, tot_samples_per_class=tot, seed=seed).indices
        pairs = np.array([[int(i), labels[i]] for i in idx], dtype=np.int64)  # [N, 2] (index, label)
        np.save(out / f"{d['dataset_arg']}_idx_to_class_seed_{seed}.npy", pairs)
        (out / f"{d['dataset_arg']}_idx_to_class_seed_{seed}.json").write_text(json.dumps(
            [{"index": int(i), "label": int(l), "class_name": str(names_list[l])} for i, l in pairs],
            indent=2))
        print(f"  {name}: {len(pairs)} images -> {d['dataset_arg']}_idx_to_class_seed_{seed}.{{npy,json}}")

    params = [{"type": "run", "models": models, "model_slugs": slugs,
               "pretrained": {s: pretrained_for(s) for s in slugs},
               "datasets": datasets, "text_sets": text_sets, "seed": seed,
               "devices": devices, "parallel": parallel, "output_dir": str(out),
               "fairface_label": fairface_label, "options": vars(opt)}]
    for name in datasets:
        params.append({"type": "dataset", "name": name, "dataset_arg": DATASETS[name]["dataset_arg"], **subset[name]})
    with open(out / "parameters.jsonl", "w") as f:
        for rec in params:
            f.write(json.dumps(rec) + "\n")

    cfg = types.SimpleNamespace(datasets=datasets, text_sets=text_sets, subset=subset,
                                seed=seed, out=out, opt=opt)

    print(f"\nPlan: {len(slugs)} model(s) x {len(datasets)} dataset(s) + "
          f"{len(text_sets)} extra text set(s) -> {out}")
    print(f"Models: {', '.join(slugs)}   Datasets: {', '.join(datasets)}   Seed: {seed}")
    if not _ask_bool("Proceed?", True):
        print("Aborted (subset maps + parameters.jsonl were still written).")
        return

    # -- run --
    if parallel:
        dq = queue.Queue()
        for d in devices:
            dq.put(d)

        def worker(slug):
            device = dq.get()
            try:
                run_model(slug, device, cfg)
            finally:
                dq.put(device)

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            list(pool.map(worker, slugs))
    else:
        for slug in slugs:
            run_model(slug, devices[0], cfg)

    print(f"\nAll done. Outputs in {out}")


if __name__ == "__main__":
    main()
