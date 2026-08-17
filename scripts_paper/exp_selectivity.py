"""Q3b - is the intervention SELECTIVE, or is it just damage that happens to help?

Worst-group accuracy going up is not evidence that a shortcut was removed. Mean-ablating most of a
network changes everything, and flattening a strong group raises the worst one. The claim needs a
double dissociation:

    a component whose removal destroys the SHORTCUT attribute while leaving the TASK intact.

Both are read out zero-shot from embeddings we already have, so this is cheap:

    task      "a photo of a landbird"      vs "a photo of a waterbird"
    shortcut  "a photo of a land background" vs "a photo of a water background"

Per component we record (delta task accuracy, delta shortcut accuracy) under single-component
ablation, and the same pair for the selected pool. Plotting one against the other is the figure:

    lower-right   removes the shortcut, keeps the task   <- what the method must find
    lower-left    removes both                            <- damage
    upper-*       removes neither                         <- irrelevant component

The shortcut readout needs its own ground truth, so this runs where the spurious attribute is
labelled - Waterbirds `place` - and the prompts are stated in the file rather than tuned, because
tuning the probe until the story works would make the result meaningless.

  python -m scripts_paper.exp_selectivity --models ViT-B-32 --device cpu
Writes output_dir/results_paper/6_2/selectivity_{comp,pool}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task, _mean_state, keep_only
from scripts_paper.exp_npairs import sample_cal
from scripts_paper.exp_waterbirds import TASK, VARIANTS, accuracies, groups, local_search
from scripts_paper.pclens_core import (ACT_DIR, MODEL_PRETRAINED, MODELS, RES_DIR, comp_delta,
                                       component_index, embed, logit_scale)

OUT = os.path.join(RES_DIR, "6_2")
# fixed, not tuned: tuning the probe until the story works would void the experiment
TASK_PROMPTS = ["a photo of a landbird", "a photo of a waterbird"]
SPUR_PROMPTS = ["a photo of a land background", "a photo of a water background"]


def _text_embed(model, prompts, device):
    """Encode the probe prompts with the frozen text tower (no decomposition needed: the probe is
    a fixed readout, not something the selector may modify)."""
    from utils.models.factory import create_model_and_transforms, get_tokenizer
    m, _, _ = create_model_and_transforms(model, pretrained=MODEL_PRETRAINED[model],
                                          precision="fp32")
    m.eval().to(device)
    tok = get_tokenizer(model)
    with torch.no_grad():
        e = m.encode_text(tok(prompts).to(device))
    del m
    return torch.nn.functional.normalize(e.float(), dim=1)


@torch.no_grad()
def _acc(Ev, T, tgt):
    L = torch.nn.functional.normalize(Ev, dim=1) @ T.T
    return float((L.argmax(1) == tgt).float().mean() * 100)


def run_model(model, device="cpu", seed=69, n_pairs=1, arm="informative", n_runs=3,
              max_steps=10 ** 6, seed_topk=32):
    av, mv, at, mt, _ = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    tau_inv = logit_scale(model)
    comps = ([("vision", c) for c in component_index(av, mv)] +
             [("text", c) for c in component_index(at, mt)])
    T_task = _text_embed(model, TASK_PROMPTS, device)
    T_spur = _text_embed(model, SPUR_PROMPTS, device)

    Ev0 = embed(av, mv)
    base_task = _acc(Ev0, T_task, y)
    base_spur = _acc(Ev0, T_spur, place)
    print(f"[{model}] full model: task(bird)={base_task:.2f}%  shortcut(background)="
          f"{base_spur:.2f}%  - both read out zero-shot with fixed prompts", flush=True)

    # ---- per-component single ablation: what does removing exactly this one do to each readout?
    acts = {"vision": (av, mv), "text": (at, mt)}
    mean = {t: (acts[t][0].mean(0), None if acts[t][1] is None else acts[t][1].mean(0))
            for t in acts}
    rows = []
    for c in component_index(av, mv):
        d = comp_delta(av, mv, c, *mean["vision"])
        Ev = Ev0 - d                      # mean-ablate this component only
        rows.append(dict(model=model, tower="vision", kind=c[0], layer=c[1], head=c[2],
                         task_acc=_acc(Ev, T_task, y), spur_acc=_acc(Ev, T_spur, place)))
    d = pd.DataFrame(rows)
    d["d_task"] = d.task_acc - base_task
    d["d_spur"] = d.spur_acc - base_spur
    d["base_task"] = base_task
    d["base_spur"] = base_spur
    # selectivity: drops the shortcut far more than the task
    d["selectivity"] = d.d_spur - d.d_task
    top = d.nsmallest(8, "selectivity")
    print(f"[{model}] most selective single components (shortcut falls, task held):", flush=True)
    for r in top.itertuples():
        nm = f"MLP {r.layer}" if r.kind == "mlp" else f"L{r.layer}H{r.head}"
        print(f"    {nm:10s} dtask={r.d_task:+6.2f}  dshortcut={r.d_spur:+6.2f}", flush=True)

    # ---- the selected pools, same two readouts
    prows = []
    for rep in range(n_runs):
        gen = torch.Generator().manual_seed(seed + 977 * rep)
        cal = sample_cal(y, place, n_pairs, arm, gen, device)
        if cal is None:
            continue
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, _ = local_search(av, mv, at, mt, tau_inv, cal, y[cal], max_steps=max_steps,
                                   seed_topk=seed_topk, verbose=False, **cfg)
            Ev, Et = keep_only(av, mv, at, mt, pool)
            wg = accuracies(Ev[held], Et, y[held], place[held])
            prows.append(dict(model=model, variant=variant, rep=rep, size=len(pool),
                              task_acc=_acc(Ev[held], T_task, y[held]),
                              spur_acc=_acc(Ev[held], T_spur, place[held]),
                              base_task=base_task, base_spur=base_spur,
                              worst_group=wg["worst_group"], total=wg["total"]))
            p = prows[-1]
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):4d}  task {base_task:.1f}->"
                  f"{p['task_acc']:.1f}  shortcut {base_spur:.1f}->{p['spur_acc']:.1f}  "
                  f"worst-group={p['worst_group']:.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, f"selectivity_comp_{model}.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(OUT, f"selectivity_pool_{model}.csv"), index=False)
    print(f"[{model}] wrote selectivity_{{comp,pool}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("Q3b: selectivity", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_pairs", default=1, type=int)
    p.add_argument("--arm", default="informative", choices=["informative", "random"])
    p.add_argument("--n_runs", default=3, type=int)
    p.add_argument("--max_steps", default=10 ** 6, type=int)
    p.add_argument("--seed_topk", default=32, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_pairs, a.arm, a.n_runs, a.max_steps, a.seed_topk)
