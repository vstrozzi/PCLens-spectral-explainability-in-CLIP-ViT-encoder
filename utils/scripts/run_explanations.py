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
CLASSIFIER_DATASETS = {"imagenet", "CIFAR10", "CIFAR100", "binary_waterbirds", "waterbirds", "cub", "fairface", "caltech"}

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


def ensure_classifier(model, dataset, indir, seed, device):
    """{dataset}_classifier_{model}.npy (zero-shot class matrix) via compute_classes_embeddings.
    For fairface the attribute (gender/race/age) is inferred from the extracted labels' class count
    so the classifier ALWAYS matches the labels; a stale classifier with the wrong #classes is
    rebuilt (this is what caused the gender-classifier-vs-race-labels mismatch)."""
    import numpy as np
    cf = indir / f"{dataset}_classifier_{model}.npy"
    extra = []
    if dataset == "fairface":
        lab = indir / f"fairface_labels_{model}_seed_{seed}.npy"
        if not lab.exists():
            print(f"  [skip] fairface: no {lab.name} to infer the attribute.")
            return False
        n = int(np.unique(np.load(lab)).size)
        attr = {2: "gender", 7: "race", 9: "age"}.get(n)
        if attr is None:
            print(f"  [skip] fairface labels have {n} classes (expected 2/7/9); cannot map attribute.")
            return False
        extra = ["--fairface_label", attr]
        if cf.exists():
            if int(np.load(cf).shape[1]) == n:
                return True
            print(f"  [fairface] classifier has {int(np.load(cf).shape[1])} classes but labels have "
                  f"{n} ({attr}); rebuilding to match.")
    elif cf.exists():
        return True
    if dataset not in CLASSIFIER_DATASETS:
        print(f"  [skip] no class-classifier builder for '{dataset}' (compute_classes_embeddings "
              f"supports {sorted(CLASSIFIER_DATASETS)}).")
        return False
    return sh([PY, "-m", "utils.scripts.compute_classes_embeddings",
               "--model", model, "--pretrained", PRETRAINED.get(model, "openai"),
               "--dataset", dataset, "--device", device, "--output_dir", str(indir)] + extra) == 0


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


def paired_eval_dataset(text_dataset):
    """Image dataset whose own images+labels evaluate a text-tower run (X_classnames -> X)."""
    return text_dataset[:-len("_classnames")] if text_dataset.endswith("_classnames") else text_dataset


def cmd_text(model, dataset, component, textset, indir, out_file, device, cfg):
    return [PY, "-m", "utils.scripts.compute_text_explanations_text",
            "--model", model, "--dataset", dataset, "--components", component,
            "--text_descriptions", textset, "--text_dir", str(TEXT_DIR),
            "--algorithm", cfg.algorithm, "--num_of_last_layers", str(cfg.num_of_last_layers),
            "--text_per_princ_comp", str(cfg.text_per_pc), "--max_text", str(cfg.max_text),
            "--seed", str(cfg.seed), "--device", device, "--probe_modality", "text",
            "--probe_name", cfg.probe_name, "--eval_dataset", paired_eval_dataset(dataset),
            "--image_set", cfg.image_set,
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
            if not cfg.rerun and out_file.exists():
                print(f"  [skip] {out_file.name} exists (rerun=false).")
                continue
            if tower == "image":
                if not ensure_classifier(model, dataset, cfg.indir, cfg.seed, device):
                    continue
                sh(cmd_image(model, dataset, cfg.component, cfg.textset, cfg.indir, out_file, device, cfg), tag)
            else:
                # Text tower is evaluated on the target's OWN images; datasets with no paired image
                # set (fit-only text sets like *_bias_clean) are not targets and are skipped.
                eval_ds = paired_eval_dataset(dataset)
                lab = cfg.indir / f"{eval_ds}_labels_{model}_seed_{cfg.seed}.npy"
                if not ensure_probe_embeddings(model, eval_ds, cfg.indir, cfg.seed) or not lab.exists():
                    print(f"  [skip] text tower '{dataset}': no paired image eval set '{eval_ds}' "
                          f"(needs {eval_ds}_embeddings + _labels).")
                    continue
                sh(cmd_text(model, dataset, cfg.component, cfg.textset, cfg.indir, out_file, device, cfg), tag)
    print(f"{tag} done.")


# --------------------------------------------------------------------------- #
# accuracy table
# --------------------------------------------------------------------------- #
def fmt_num(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "NA"


def fmt_pair(p):
    return f"{fmt_num(p[0])}/{fmt_num(p[1])}"


def write_accuracy_table(run_dir, cfg):
    rows = []
    for jf in sorted(run_dir.glob("*.jsonl")):
        if jf.name == "parameters.jsonl":            # run config, not a per-run result
            continue
        stem = jf.stem  # {dataset}_{model}_{component}_{modality}
        if stem.count("_") < 3:                       # not a per-run result file
            continue
        try:
            last = jf.read_text().strip().splitlines()[-1]
            final = json.loads(last)
        except Exception:
            continue
        modality = stem.rsplit("_", 1)[-1]
        component = stem.rsplit("_", 2)[-2]
        model = stem.rsplit("_", 3)[-3]
        dataset = stem[: -(len(model) + len(component) + len(modality) + 3)]
        # Each accuracy is a (image->label, label->image) pair, so both directions are comparable.
        def pair(base, *alts):
            i2l = final.get(base, float("nan"))
            for a in alts:
                if not (isinstance(i2l, (int, float)) and i2l == i2l):
                    i2l = final.get(a, float("nan"))
            return (i2l, final.get(base + "_l2i", float("nan")))
        rows.append((dataset, model, component, modality,
                     final.get("n_pcs", ""),
                     final.get("n_pcs_comp_mean", float("nan")),
                     final.get("n_pcs_comp_std", float("nan")),
                     pair("full_accuracy"), pair("pc_accuracy"),
                     pair("text_self_accuracy"),
                     pair("text_ref_accuracy", "text_accuracy", "accuracy"),
                     pair("image_self_accuracy"),
                     pair("image_ref_accuracy", "image_accuracy")))
    rows.sort()

    hdr = ["dataset", "model", "component", "modality", "n_pcs", "pc_comp_mean", "pc_comp_std",
           "model_acc", "pc_acc", "text_self", "text_ref", "image_self", "image_ref", "recovered_%"]
    legend = [
        "# PCLens completeness table -- SELF mode (the PC basis vh is fit on each TARGET's own activations).",
        f"# algorithm={cfg.algorithm}  n_explanations={cfg.nexpl}  seed={cfg.seed}  "
        f"textset={cfg.textset}  image_set={cfg.image_set}  num_of_last_layers={cfg.num_of_last_layers}",
        "#",
        "# Every *_acc / *_self / *_ref cell is  image->label / label->image  (top-1 zero-shot %, both directions):",
        "#   image->label : for each target image, argmax over the class prototypes == its label (standard zero-shot).",
        "#   label->image : for each class, the single top-scoring target image belongs to that class (retrieval).",
        "# Each *_self/_ref column APPROXIMATES the target's PCs with a candidate embedding pool (the 'explanator'),",
        "# rebuilds the embedding and re-runs zero-shot on the target. Columns:",
        "#   dataset      : target dataset (its activations are decomposed; zero-shot is measured on it).",
        "#   model        : CLIP model.",
        "#   component    : residual components decomposed (attn heads / mlp layers / all).",
        "#   modality     : tower explained -- image = image encoder, text = text encoder.",
        "#   n_pcs        : rank (kept PCs to 99% variance) of the single final-embedding decomposition.",
        "#   pc_comp_mean : mean kept-PC count across the decomposed head/MLP units.",
        "#   pc_comp_std  : std of that per-unit kept-PC count.",
        "#   model_acc    : real-model zero-shot, all components summed, NO approximation (upper bound).",
        "#   pc_acc       : rank-r PC-subspace reconstruction (no example approximation).",
        "#   text_self    : PCs approximated with the TARGET's own class-name text embeddings.",
        "#   text_ref     : PCs approximated with a general REFERENCE text set (=textset).",
        "#   image_self   : PCs approximated with the TARGET's own image embeddings.",
        "#   image_ref    : PCs approximated with a general REFERENCE image set (=image_set).",
        "#   recovered_%  : 100 * text_ref(image->label) / model_acc(image->label).",
        "#   NA           : not available for this run.",
    ]
    lines = legend + [",".join(hdr)]
    for r in rows:
        mean_c, std_c, full, pc, t_self, t_ref, i_self, i_ref = r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]
        rec = (f"{100 * t_ref[0] / full[0]:.2f}"
               if isinstance(full[0], (int, float)) and full[0] and t_ref[0] == t_ref[0] else "NA")
        lines.append(",".join([str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]),
                               fmt_num(mean_c), fmt_num(std_c), fmt_pair(full), fmt_pair(pc),
                               fmt_pair(t_self), fmt_pair(t_ref), fmt_pair(i_self), fmt_pair(i_ref), rec]))
    (run_dir / "zero_shot_accuracy.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {run_dir / 'zero_shot_accuracy.csv'}")


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
                    ensure_classifier(model, ds, cfg.indir, cfg.seed, device)
            else:
                # Text tower is scored on each target's OWN paired images (X_classnames -> X).
                for ds in targets:
                    ensure_probe_embeddings(model, paired_eval_dataset(ds), cfg.indir, cfg.seed)
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

    # Same CSV format/columns as the self-mode zero_shot_accuracy.csv, plus a leading fit_dataset.
    hdr = ["fit_dataset", "dataset", "model", "component", "modality", "n_pcs", "pc_comp_mean",
           "pc_comp_std", "model_acc", "pc_acc", "text_self", "text_ref", "image_self", "image_ref",
           "recovered_%"]
    legend = [
        "# PCLens completeness table -- TRANSFER mode (ONE PC basis vh is fit on the general "
        "fit_dataset and FROZEN, then evaluated on every target).",
        f"# algorithm={cfg.algorithm}  seed={cfg.seed}  textset={cfg.textset}  "
        f"image_set={cfg.image_set}  num_of_last_layers={cfg.num_of_last_layers}",
        "#",
        "# Every *_acc / *_self / *_ref cell is  image->label / label->image  (top-1 zero-shot %, both directions).",
        "# Columns match zero_shot_accuracy.csv, with a leading fit_dataset:",
        "#   fit_dataset  : general dataset the frozen PC basis was fit on (the transfer source).",
        "#   dataset      : target dataset (projected onto the frozen basis; zero-shot is measured on it).",
        "#   model        : CLIP model.",
        "#   component    : residual components decomposed (attn heads / mlp layers / all).",
        "#   modality     : tower explained -- image = image encoder, text = text encoder.",
        "#   n_pcs        : rank (99% variance) of the TARGET's own final-embedding decomposition.",
        "#   pc_comp_mean : mean kept-PC count across the frozen fit-basis units (constant across targets).",
        "#   pc_comp_std  : std of that per-unit kept-PC count.",
        "#   model_acc    : real-model zero-shot, all components summed, NO approximation (upper bound).",
        "#   pc_acc       : frozen rank-r PC-subspace reconstruction (no example approximation).",
        "#   text_self    : PCs approximated with the TARGET's own class-name text embeddings.",
        "#   text_ref     : PCs approximated with the general REFERENCE text set (=textset), frozen on fit.",
        "#   image_self   : PCs approximated with the TARGET's own image embeddings.",
        "#   image_ref    : PCs approximated with the general REFERENCE image set (=image_set), frozen on fit.",
        "#   recovered_%  : 100 * text_ref(image->label) / model_acc(image->label).",
        "#   NA           : not available for this run.",
    ]
    lines = legend + [",".join(hdr)]
    for r in sorted(rows, key=lambda x: (x["tower"], x["dataset"], x["model"])):
        full, t_ref = r["model_acc"], r["text_ref"]
        rec = (f"{100 * t_ref[0] / full[0]:.2f}"
               if isinstance(full[0], (int, float)) and full[0] and t_ref[0] == t_ref[0] else "NA")
        lines.append(",".join([
            str(r["fit_dataset"]), str(r["dataset"]), str(r["model"]), str(r["components"]),
            str(r["tower"]), str(r["n_pcs"]), fmt_num(r["pc_comp_mean"]), fmt_num(r["pc_comp_std"]),
            fmt_pair(r["model_acc"]), fmt_pair(r["pc_acc"]), fmt_pair(r["text_self"]),
            fmt_pair(r["text_ref"]), fmt_pair(r["image_self"]), fmt_pair(r["image_ref"]), rec]))
    out = cfg.run_dir / "zero_shot_accuracy_transfer.csv"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


def get_args_parser():
    import argparse
    p = argparse.ArgumentParser("run_explanations", add_help=True)
    # Each flag becomes the default of the matching prompt, so interactive runs pre-fill them and
    # non-interactive (SLURM) runs use them verbatim. Defaults == the interactive defaults.
    p.add_argument("--seed", default=None, help="default: the single/first seed found")
    p.add_argument("--modalities", default="all", help="'all' or comma list of image,text")
    p.add_argument("--models", default="all")
    p.add_argument("--datasets", default="all")
    p.add_argument("--mode", default="both", help="self | transfer | both")
    p.add_argument("--components", default="all", help="attn | mlp | all")
    p.add_argument("--algorithm", default="svd_data_approx")
    p.add_argument("--textset", default="top_1500_nouns_5_sentences_imagenet_clean")
    p.add_argument("--image_set", default="imagenet",
                   help="reference image pool for image_ref + poles (default imagenet); "
                        "image_self always uses the dataset's own images, text_self its class names")
    p.add_argument("--n_explanations", default="auto", help="integer or 'auto' (99% variance)")
    p.add_argument("--num_of_last_layers", default="all")
    p.add_argument("--text_per_princ_comp", default="5", help="top texts/images per PC")
    p.add_argument("--probe_name", default="imagenet")
    p.add_argument("--fit_image", default="imagenet", help="[transfer] general fit dataset, image tower")
    p.add_argument("--fit_text", default="", help="[transfer] general fit dataset, text tower (default: auto)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--parallel", action="store_true", help="one model per GPU across all visible GPUs")
    p.add_argument("--save_jsonl", default="true",
                   help="keep the per-run .jsonl explanations (true) or, if false, delete them after "
                        "building the tables so only zero_shot_accuracy.csv (+parameters.jsonl) remain")
    p.add_argument("--rerun", default="true",
                   help="recompute a (model,dataset,tower) run even if its .jsonl already exists (true); "
                        "false skips existing results and only fills missing ones + (re)builds the tables")
    return p


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = get_args_parser().parse_args()
    avail, indir = discover()
    if not avail:
        print("No activations found under output_dir. Gather them first:\n"
              "  python -m utils.scripts.extract_activations")
        return

    seeds = sorted(avail)
    seed = int(_ask(f"\nSeed to analyse (found: {seeds})", str(args.seed if args.seed is not None else seeds[0])))
    if seed not in avail:
        print("No activations for that seed. Exiting.")
        return
    by_tower = avail[seed]

    towers_present = [t for t in ("image", "text") if by_tower[t]]
    print(f"\nTowers available: {towers_present}")
    modalities = _pick("Which modalities (towers)?", towers_present, default=args.modalities)
    if not modalities:
        print("Nothing selected. Exiting.")
        return

    models = sorted({m for t in modalities for m in by_tower[t]})
    print(f"Models available: {models}")
    models = _pick("Which models?", models, default=args.models)

    datasets_present = sorted({d for t in modalities for m in by_tower[t] for d in by_tower[t][m]})
    print(f"Datasets available: {datasets_present}")
    datasets = set(_pick("Which datasets?", datasets_present, default=args.datasets))

    # keep only what actually exists, filtered by the user's picks
    datasets_by_tower = {t: {m: sorted(by_tower[t][m] & datasets) for m in models if m in by_tower[t]}
                         for t in modalities}

    mode = _ask("Analysis mode: 'self' (per-dataset basis), 'transfer' (fit ONE general basis, "
                "eval on all -- robust for few-label targets), or 'both'", args.mode)
    do_self, do_transfer = mode in ("self", "both"), mode in ("transfer", "both")

    component = _ask("Components (attn / mlp / all)", args.components)
    algorithm = _ask("Algorithm", args.algorithm)

    textsets = sorted(p.stem for p in TEXT_DIR.glob("*.txt"))
    print(f"\nCandidate-text sets available:\n  {', '.join(textsets)}")
    textset = _ask("Which text set labels the PCs?", args.textset)

    image_datasets = sorted({d for m in by_tower["image"] for d in by_tower["image"][m]}) if by_tower["image"] else []
    print(f"Image sets available to label the PCs: {['self', 'all'] + image_datasets}")
    image_set = _ask("Which image set labels the PCs? ('self', 'all', or dataset name(s))", args.image_set)

    fit_by_tower = {}
    if do_transfer:
        for t in modalities:
            pool = sorted({d for m in models if m in by_tower[t] for d in by_tower[t][m]})
            cli = args.fit_image if t == "image" else args.fit_text
            default_fit = cli or ("imagenet" if (t == "image" and "imagenet" in pool) else (
                next((d for d in pool if "nouns" in d or "bias" in d), pool[0] if pool else "")))
            fit_by_tower[t] = _ask(f"  [transfer] general fit dataset for the {t} tower "
                                   f"(many samples; from {pool})", default_fit)

    nexpl_in = _ask("n_explanations = PCs per unit (integer, or 'auto' = up to 99% variance)", args.n_explanations)
    if nexpl_in.lower() in ("auto", "var99", ""):
        nexpl, max_text = "var99", 999
    else:
        nexpl, max_text = f"k{int(nexpl_in)}", int(nexpl_in)

    nll_in = _ask("num_of_last_layers to decompose ('all' or an integer)", args.num_of_last_layers)
    num_of_last_layers = 999 if str(nll_in).lower() in ("all", "") else int(nll_in)
    text_per_pc = int(_ask("texts/images per principal component", args.text_per_princ_comp))
    probe_name = _ask("probe image dataset for the text tower (needs its image embeddings)", args.probe_name) \
        if "text" in modalities else args.probe_name

    import torch
    if not torch.cuda.is_available():
        print("\nCUDA not available; this requires a GPU. Exiting.")
        return
    n_gpu = torch.cuda.device_count()
    parallel = n_gpu > 1 and _ask_bool(f"\n{n_gpu} GPUs found. Run models in parallel, one per GPU?", args.parallel)
    devices = [f"cuda:{i}" for i in range(n_gpu)] if parallel else [_ask("Which CUDA device?", args.device)]
    save_jsonl = _ask_bool("Keep the per-run .jsonl explanations? ('n' = only the .csv tables)",
                           str(args.save_jsonl).lower() in ("true", "t", "yes", "y", "1"))
    rerun = _ask_bool("Recompute runs whose .jsonl already exists? ('n' = skip existing, fill missing only)",
                      str(args.rerun).lower() in ("true", "t", "yes", "y", "1"))

    run_dir = indir[seed] / f"{algorithm}_{nexpl}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = types.SimpleNamespace(
        seed=seed, modalities=modalities, datasets_by_tower=datasets_by_tower, component=component,
        algorithm=algorithm, textset=textset, image_set=image_set, nexpl=nexpl, max_text=max_text,
        num_of_last_layers=num_of_last_layers, text_per_pc=text_per_pc, probe_name=probe_name,
        indir=indir[seed], run_dir=run_dir, save_jsonl=save_jsonl, rerun=rerun)

    # Record the manually-specified parameters of this interactive run (like extract_activations.py).
    params = {
        "seed": seed, "modalities": modalities, "models": models,
        "datasets_by_tower": {t: {m: list(v) for m, v in datasets_by_tower[t].items()} for t in modalities},
        "mode": mode, "component": component, "algorithm": algorithm, "textset": textset,
        "image_set": image_set, "n_explanations": nexpl, "max_text": max_text,
        "num_of_last_layers": num_of_last_layers, "text_per_princ_comp": text_per_pc,
        "probe_name": probe_name, "fit_by_tower": fit_by_tower, "devices": devices,
        "parallel": parallel, "save_jsonl": save_jsonl, "rerun": rerun, "input_dir": str(indir[seed]),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "parameters.jsonl", "w") as pf:
        pf.write(json.dumps(params) + "\n")
    print(f"Wrote {run_dir / 'parameters.jsonl'}")

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

    # Tables are already written; drop the bulky per-run explanations if the user asked to.
    if not save_jsonl:
        removed = 0
        for jf in run_dir.glob("*.jsonl"):
            if jf.name == "parameters.jsonl":
                continue
            jf.unlink()
            removed += 1
        print(f"save_jsonl=false -> removed {removed} per-run .jsonl file(s); kept the .txt tables "
              f"and parameters.jsonl.")

    print(f"\nAll done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
