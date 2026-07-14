"""
Interactive driver that runs a spectral-explanation algorithm (PCLens =
``svd_data_approx``, or any other selectable one) over the activations produced by
``extract_activations`` -- the analysis counterpart of that gathering step.

It never re-implements the algorithm: it discovers what activations exist, asks
what to analyse, ensures the small prerequisites (class classifier / candidate-text
embeddings) via the existing scripts, then delegates every run to

  * image tower : utils.scripts.compute_text_explanations
  * text  tower : utils.scripts.compute_text_explanations_text

and finally collects one zero-shot accuracy table for the whole batch.

Run from the repo root:
    python -m utils.scripts.run_explanations

Flow (everything is listed dynamically from output_dir):
  1. pick the seed (from the activation files found);
  2. pick models / datasets / towers (modality: image vs text) that exist;
  3. pick components (attn / mlp / all) and the algorithm;
  4. pick the candidate-text set used to label PCs, and n_explanations
     (PCs per unit: an integer cap, or 'auto' = up to 99% variance);
  5. pick num_of_last_layers, texts-per-PC, device / parallel;
  6. run, then write the accuracy table.

Output layout (consistent):
    output_dir/current_analyzed_dir/{algorithm}_{n_explanations}/
        {dataset}_{model}_{component}_{modality}.jsonl   # one file per run
        zero_shot_accuracy.txt                           # one table for the batch

Each .jsonl has one line per decomposed unit (attn head or mlp layer) with its PCs,
each PC labelled by its top/bottom TEXTS and IMAGES (index + class, for visualization),
and a final layer=-1 line carrying the reconstruction accuracies:
    full  (real model, all components summed) >= pc (rank-r PC subspace)
        >= image (top-image-span) ~ text (top-text-span, the PCLens completeness).
"""
import json
import os
import queue
import re
import subprocess
import sys
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFO_JSON = ROOT / "experiments_info.json"
OUTPUT_DIR = ROOT / "output_dir"
TEXT_DIR = ROOT / "utils" / "text_descriptions"
PY = sys.executable

_INFO = json.loads(INFO_JSON.read_text())
PRETRAINED = {m["slug"]: m["pretrained"] for m in _INFO["models_config"]}

# datasets whose zero-shot class classifier compute_classes_embeddings can build
CLASSIFIER_DATASETS = {"imagenet", "CIFAR10", "CIFAR100", "binary_waterbirds", "waterbirds", "cub", "fairface"}

_ACT_RE = re.compile(r"^(?P<ds>.+)_attn(?P<text>_text)?_(?P<model>[A-Za-z0-9\-]+)_seed_(?P<seed>\d+)\.npy$")


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover():
    """Scan output_dir (recursively) for activation files -> nested availability + input dirs.

    Returns (avail, indir): avail[seed][tower] = {model: set(datasets)};
    indir[seed] = directory the seed's activations live in.
    """
    avail = defaultdict(lambda: {"image": defaultdict(set), "text": defaultdict(set)})
    indir = {}
    for path in OUTPUT_DIR.rglob("*_seed_*.npy"):
        m = _ACT_RE.match(path.name)
        if not m:
            continue
        seed = int(m["seed"])
        tower = "text" if m["text"] else "image"
        ds, model = m["ds"], m["model"]
        # need the matching mlp + labels to be usable
        tag = "_text" if tower == "text" else ""
        needed = [f"{ds}_mlp{tag}_{model}_seed_{seed}.npy", f"{ds}_labels{tag}_{model}_seed_{seed}.npy"]
        if not all((path.parent / n).exists() for n in needed):
            continue
        avail[seed][tower][model].add(ds)
        indir.setdefault(seed, path.parent)
    return avail, indir


# --------------------------------------------------------------------------- #
# interactive helpers (same style as extract_activations / download_datasets)
# --------------------------------------------------------------------------- #
def _ask(prompt, default):
    # Non-interactive (no TTY, e.g. under SLURM) or EOF -> take the default. CLI flags set the
    # defaults (see get_args_parser), so a batch run is fully driven by args.
    if not sys.stdin.isatty():
        print(f"{prompt} [{default}]: {default}")
        return default
    try:
        resp = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return resp or default


def _ask_bool(prompt, default):
    return _ask(prompt, "y" if default else "n").lower() in ("y", "yes", "true", "t", "1")


def _pick(prompt, known, default="all"):
    known = list(known)
    sel = _ask(f"{prompt} (comma-separated or 'all')", default)
    if sel.lower() == "all":
        return known
    if sel.lower() in ("none", ""):
        return []
    lut = {k.lower(): k for k in known}
    chosen = [lut[t.strip().lower()] for t in sel.split(",") if t.strip().lower() in lut]
    return [k for k in known if k in chosen]


# --------------------------------------------------------------------------- #
# prerequisites (delegated to existing scripts)
# --------------------------------------------------------------------------- #
def sh(cmd, tag=""):
    print(f"{tag} >> {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"{tag} !! exited {rc}")
    return rc


def ensure_text_embeddings(model, textset, indir, device):
    """{textset}_{model}.npy (candidate concept labels) via compute_text_embeddings."""
    if (indir / f"{textset}_{model}.npy").exists():
        return True
    return sh([PY, "-m", "utils.scripts.compute_text_embeddings",
               "--model", model, "--pretrained", PRETRAINED.get(model, "openai"),
               "--data_path", str(TEXT_DIR / f"{textset}.txt"),
               "--device", device, "--output_dir", str(indir)]) == 0


def ensure_probe_embeddings(model, probe, indir, seed):
    """{probe}_embeddings_{model}_seed_{seed}.npy (image embeddings the text tower classifies)
    -- reassembled from the probe's saved image activations via compute_images_embedding."""
    if (indir / f"{probe}_embeddings_{model}_seed_{seed}.npy").exists():
        return True
    if not (indir / f"{probe}_attn_{model}_seed_{seed}.npy").exists():
        print(f"  [skip] no {probe} image activations to build {probe}_embeddings_{model} "
              f"(gather them with extract_activations, or pick a different --probe_name).")
        return False
    return sh([PY, "-m", "utils.scripts.compute_images_embedding",
               "--dataset", probe, "--model", model, "--seed", str(seed),
               "--output_dir", str(indir)]) == 0


def ensure_classifier(model, dataset, indir, device):
    """{dataset}_classifier_{model}.npy (zero-shot class matrix) via compute_classes_embeddings."""
    if (indir / f"{dataset}_classifier_{model}.npy").exists():
        return True
    if dataset not in CLASSIFIER_DATASETS:
        print(f"  [skip] no class-classifier builder for '{dataset}' (compute_classes_embeddings "
              f"supports {sorted(CLASSIFIER_DATASETS)}).")
        return False
    return sh([PY, "-m", "utils.scripts.compute_classes_embeddings",
               "--model", model, "--pretrained", PRETRAINED.get(model, "openai"),
               "--dataset", dataset, "--device", device, "--output_dir", str(indir)]) == 0


# --------------------------------------------------------------------------- #
# command builders
# --------------------------------------------------------------------------- #
def cmd_image(model, dataset, component, textset, indir, out_file, device, cfg):
    return [PY, "-m", "utils.scripts.compute_text_explanations",
            "--model", model, "--dataset", dataset, "--components", component,
            "--text_descriptions", textset, "--text_dir", str(TEXT_DIR),
            "--algorithm", cfg.algorithm, "--num_of_last_layers", str(cfg.num_of_last_layers),
            "--text_per_princ_comp", str(cfg.text_per_pc), "--max_text", str(cfg.max_text),
            "--seed", str(cfg.seed), "--device", device, "--image_set", cfg.image_set,
            "--input_dir", str(indir), "--output_dir", str(indir), "--out_file", str(out_file)]


def cmd_text(model, dataset, component, textset, indir, out_file, device, cfg):
    return [PY, "-m", "utils.scripts.compute_text_explanations_text",
            "--model", model, "--dataset", dataset, "--components", component,
            "--text_descriptions", textset, "--text_dir", str(TEXT_DIR),
            "--algorithm", cfg.algorithm, "--num_of_last_layers", str(cfg.num_of_last_layers),
            "--text_per_princ_comp", str(cfg.text_per_pc), "--max_text", str(cfg.max_text),
            "--seed", str(cfg.seed), "--device", device, "--probe_modality", "text",
            "--probe_name", cfg.probe_name, "--image_set", cfg.image_set,
            "--input_dir", str(indir), "--output_dir", str(indir), "--out_file", str(out_file)]


# --------------------------------------------------------------------------- #
# per-model runner
# --------------------------------------------------------------------------- #
def run_model(model, device, cfg):
    tag = f"[{model} @ {device}]"
    print(f"\n{'=' * 70}\n{tag} starting\n{'=' * 70}")
    ensure_text_embeddings(model, cfg.textset, cfg.indir, device)
    for tower in cfg.modalities:
        for dataset in cfg.datasets_by_tower[tower].get(model, []):
            out_file = cfg.run_dir / f"{dataset}_{model}_{cfg.component}_{tower}.jsonl"
            if tower == "image":
                if not ensure_classifier(model, dataset, cfg.indir, device):
                    continue
                sh(cmd_image(model, dataset, cfg.component, cfg.textset, cfg.indir, out_file, device, cfg), tag)
            else:
                if not ensure_probe_embeddings(model, cfg.probe_name, cfg.indir, cfg.seed):
                    continue
                sh(cmd_text(model, dataset, cfg.component, cfg.textset, cfg.indir, out_file, device, cfg), tag)
    print(f"{tag} done.")


# --------------------------------------------------------------------------- #
# accuracy table
# --------------------------------------------------------------------------- #
def write_accuracy_table(run_dir, cfg):
    rows = []
    for jf in sorted(run_dir.glob("*.jsonl")):
        try:
            last = jf.read_text().strip().splitlines()[-1]
            final = json.loads(last)
        except Exception:
            continue
        stem = jf.stem  # {dataset}_{model}_{component}_{modality}
        modality = stem.rsplit("_", 1)[-1]
        component = stem.rsplit("_", 2)[-2]
        model = stem.rsplit("_", 3)[-3]
        dataset = stem[: -(len(model) + len(component) + len(modality) + 3)]
        rows.append((dataset, model, component, modality,
                     final.get("n_pcs", ""), final.get("full_accuracy", float("nan")),
                     final.get("pc_accuracy", float("nan")), final.get("image_accuracy", float("nan")),
                     final.get("text_accuracy", final.get("accuracy", float("nan")))))
    rows.sort()
    hdr = ["dataset", "model", "component", "modality", "n_pcs",
           "model_acc", "pc_acc", "image_acc", "text_acc", "recovered_%"]
    w = [max(len(h), *(len(str(r[i])) for r in rows)) if rows else len(h) for i, h in enumerate(hdr[:5])]

    def fmt_num(x):
        return f"{x:6.2f}" if isinstance(x, (int, float)) and x == x else "   -  "

    lines = [f"algorithm={cfg.algorithm}  n_explanations={cfg.nexpl}  seed={cfg.seed}  "
             f"textset={cfg.textset}  num_of_last_layers={cfg.num_of_last_layers}", ""]
    lines.append("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr[:5])) + "  " + "  ".join(hdr[5:]))
    for r in rows:
        full, pc, img, txt = r[5], r[6], r[7], r[8]
        rec = f"{100 * txt / full:6.2f}" if isinstance(full, (int, float)) and full else "   -  "
        lines.append("  ".join(str(r[i]).ljust(w[i]) for i in range(5)) + "  "
                     + "  ".join([fmt_num(full), fmt_num(pc), fmt_num(img), fmt_num(txt), rec]))
    (run_dir / "zero_shot_accuracy.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {run_dir / 'zero_shot_accuracy.txt'}")


def run_transfer(models, modalities, datasets_by_tower, fit_by_tower, devices, cfg):
    """Transfer completeness: fit ONE basis per (tower, model) on a general dataset, evaluate on all
    targets. 1 thread per model (each on its own GPU when parallel). Writes one aggregated table."""
    from utils.scripts.algorithms_text_explanations_funcs import transfer_completeness
    import threading
    rows, lock = [], threading.Lock()

    def work(model, device):
        for tower in modalities:
            targets = sorted(datasets_by_tower[tower].get(model, []))
            if not targets:
                continue
            ensure_text_embeddings(model, cfg.textset, cfg.indir, device)
            if tower == "image":
                for ds in targets:
                    ensure_classifier(model, ds, cfg.indir, device)
            else:
                ensure_probe_embeddings(model, cfg.probe_name, cfg.indir, cfg.seed)
            try:
                r = transfer_completeness(
                    input_dir=str(cfg.indir), model=model, seed=cfg.seed, tower=tower,
                    fit_dataset=fit_by_tower[tower], eval_datasets=targets, components=cfg.component,
                    textset=cfg.textset, text_dir=str(TEXT_DIR), image_set=cfg.image_set,
                    probe_name=cfg.probe_name, num_of_last_layers=cfg.num_of_last_layers,
                    text_per_pc=cfg.text_per_pc, max_text=cfg.max_text, device=device)
            except FileNotFoundError as e:
                print(f"[transfer] skip {model}/{tower} (fit='{fit_by_tower[tower]}'): {e}")
                continue
            with lock:
                rows.extend(r)

    if len(devices) > 1:
        dq = queue.Queue()
        for d in devices:
            dq.put(d)

        def worker(model):
            dev = dq.get()
            try:
                work(model, dev)
            finally:
                dq.put(dev)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            list(pool.map(worker, models))
    else:
        for model in models:
            work(model, devices[0])

    hdr = ["fit_dataset", "dataset", "model", "tower", "components", "model_acc", "pc_acc", "image_acc", "text_acc"]
    lines = [f"TRANSFER completeness  algorithm={cfg.algorithm}  seed={cfg.seed}  textset={cfg.textset}  "
             f"image_set={cfg.image_set}", ""]
    lines.append("  ".join(hdr))
    for r in sorted(rows, key=lambda x: (x["tower"], x["dataset"], x["model"])):
        lines.append("  ".join([str(r["fit_dataset"]), str(r["dataset"]), str(r["model"]), r["tower"],
                                r["components"]] + [f"{r[k]:6.2f}" if r[k] is not None else "   -  "
                                for k in ("model_acc", "pc_acc", "image_acc", "text_acc")]))
    out = cfg.run_dir / "zero_shot_accuracy_transfer.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    avail, indir = discover()
    if not avail:
        print("No activations found under output_dir. Gather them first:\n"
              "  python -m utils.scripts.extract_activations")
        return

    seeds = sorted(avail)
    seed = int(_ask(f"\nSeed to analyse (found: {seeds})", str(seeds[0])))
    if seed not in avail:
        print("No activations for that seed. Exiting.")
        return
    by_tower = avail[seed]

    towers_present = [t for t in ("image", "text") if by_tower[t]]
    print(f"\nTowers available: {towers_present}")
    modalities = _pick("Which modalities (towers)?", towers_present)
    if not modalities:
        print("Nothing selected. Exiting.")
        return

    models = sorted({m for t in modalities for m in by_tower[t]})
    print(f"Models available: {models}")
    models = _pick("Which models?", models)

    datasets_present = sorted({d for t in modalities for m in by_tower[t] for d in by_tower[t][m]})
    print(f"Datasets available: {datasets_present}")
    datasets = set(_pick("Which datasets?", datasets_present))

    # keep only what actually exists, filtered by the user's picks
    datasets_by_tower = {t: {m: sorted(by_tower[t][m] & datasets) for m in models if m in by_tower[t]}
                         for t in modalities}

    mode = _ask("Analysis mode: 'self' (per-dataset basis), 'transfer' (fit ONE general basis, "
                "eval on all -- robust for few-label targets), or 'both'", "self")
    do_self, do_transfer = mode in ("self", "both"), mode in ("transfer", "both")

    component = _ask("Components (attn / mlp / all)", "all")
    algorithm = _ask("Algorithm", "svd_data_approx")

    textsets = sorted(p.stem for p in TEXT_DIR.glob("*.txt"))
    print(f"\nCandidate-text sets available:\n  {', '.join(textsets)}")
    textset = _ask("Which text set labels the PCs?", "top_1500_nouns_5_sentences_imagenet_clean")

    image_datasets = sorted({d for m in by_tower["image"] for d in by_tower["image"][m]}) if by_tower["image"] else []
    print(f"Image sets available to label the PCs: {['self', 'all'] + image_datasets}")
    image_set = _ask("Which image set labels the PCs? ('self', 'all', or dataset name(s))", "self")

    fit_by_tower = {}
    if do_transfer:
        for t in modalities:
            pool = sorted({d for m in models if m in by_tower[t] for d in by_tower[t][m]})
            default_fit = "imagenet" if (t == "image" and "imagenet" in pool) else (
                next((d for d in pool if "nouns" in d or "bias" in d), pool[0] if pool else ""))
            fit_by_tower[t] = _ask(f"  [transfer] general fit dataset for the {t} tower "
                                   f"(many samples; from {pool})", default_fit)

    nexpl_in = _ask("n_explanations = PCs per unit (integer, or 'auto' = up to 99% variance)", "auto")
    if nexpl_in.lower() in ("auto", "var99", ""):
        nexpl, max_text = "var99", 999
    else:
        nexpl, max_text = f"k{int(nexpl_in)}", int(nexpl_in)

    nll_in = _ask("num_of_last_layers to decompose ('all' or an integer)", "all")
    num_of_last_layers = 999 if nll_in.lower() in ("all", "") else int(nll_in)
    text_per_pc = int(_ask("texts/images per principal component", "5"))
    probe_name = _ask("probe image dataset for the text tower (needs its image embeddings)", "imagenet") \
        if "text" in modalities else "imagenet"

    import torch
    if not torch.cuda.is_available():
        print("\nCUDA not available; this requires a GPU. Exiting.")
        return
    n_gpu = torch.cuda.device_count()
    parallel = n_gpu > 1 and _ask_bool(f"\n{n_gpu} GPUs found. Run models in parallel, one per GPU?", False)
    devices = [f"cuda:{i}" for i in range(n_gpu)] if parallel else [_ask("Which CUDA device?", "cuda:0")]

    run_dir = OUTPUT_DIR / "current_analyzed_dir" / f"{algorithm}_{nexpl}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = types.SimpleNamespace(
        seed=seed, modalities=modalities, datasets_by_tower=datasets_by_tower, component=component,
        algorithm=algorithm, textset=textset, image_set=image_set, nexpl=nexpl, max_text=max_text,
        num_of_last_layers=num_of_last_layers, text_per_pc=text_per_pc, probe_name=probe_name,
        indir=indir[seed], run_dir=run_dir)

    n_runs = sum(len(datasets_by_tower[t].get(m, [])) for t in modalities for m in models)
    print(f"\nPlan: {len(models)} model(s) x {n_runs} (tower,dataset) run(s) -> {run_dir}")
    print(f"algorithm={algorithm}  component={component}  textset={textset}  n_explanations={nexpl}")
    if not _ask_bool("Proceed?", True):
        return

    if do_self:
        if parallel:
            dq = queue.Queue()
            for d in devices:
                dq.put(d)

            def worker(model):
                dev = dq.get()
                try:
                    run_model(model, dev, cfg)
                finally:
                    dq.put(dev)

            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(worker, models))
        else:
            for model in models:
                run_model(model, devices[0], cfg)
        print("\nCollecting zero-shot accuracy table ...")
        write_accuracy_table(run_dir, cfg)

    if do_transfer:
        print("\n=== Transfer completeness (fit one general basis, evaluate on all) ===")
        run_transfer(models, modalities, datasets_by_tower, fit_by_tower, devices, cfg)

    print(f"\nAll done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
