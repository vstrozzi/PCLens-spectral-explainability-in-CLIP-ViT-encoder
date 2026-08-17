"""The dehumanisation / crime-label audit at PC granularity - so the selected units can be NAMED.

Component-level selection says "drop vision layer 9 head 3", which is not an explanation. At PC
granularity the unit is (component, principal component), and a PC is a direction that can be
labelled with the texts and images at each of its poles. The question a bias audit actually needs
answered - what is the model using when it assigns a crime or non-human label to a face - therefore
becomes answerable: run the search, then read off the directions it dropped.

Same loss as the component-level audit (zero-shot cross-entropy against the 12 prompts), same four
selectors, same held-out protocol.

  python -m scripts_paper.exp_bias_pc --models ViT-B-32 --device cpu
  python -m scripts_paper.explain_pcs --task bias --model ViT-B-32 --variant backward --regime bias
Writes output_dir/results_paper/6_2/bias_pc_{eval,trace,moves}_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_bias_audit import audit, load_audit
from scripts_paper.exp_waterbirds import VARIANTS, _ce
from scripts_paper.exp_waterbirds_pc import build_units, local_search_pc
from scripts_paper.pclens_core import MODELS, RES_DIR, component_index, embed, logit_scale

OUT = os.path.join(RES_DIR, "6_2")


@torch.no_grad()
def replay_audit(units, alpha, P, scaf, trace, init, y, sub, held):
    tix = torch.arange(scaf["text"].shape[0], device=scaf["text"].device)

    def d(u):
        t, c, j = u
        idx = slice(None) if t == "vision" else tix
        return alpha[(t, c)][idx, j].unsqueeze(1) * P[(t, c)][j]

    Ev, Et = scaf["vision"].clone(), scaf["text"].clone()
    if init == "full":
        for u in units:
            if u[0] == "vision":
                Ev = Ev + d(u)
            else:
                Et = Et + d(u)
    out = []
    for r in trace.itertuples():
        for u, s in r.applied:
            if u[0] == "vision":
                Ev = Ev + s * d(u)
            else:
                Et = Et + s * d(u)
        out.append(dict(step=r.step, **audit(Ev, Et, y, sub, held)))
    return pd.DataFrame(out)


def run_model(model, device="cpu", seed=69, n_cal=16, k=8, max_steps=10 ** 6, reps=2,
              n_random=3):
    av, mv, at, mt, y, sub = load_audit(model, device, seed)
    tau_inv = logit_scale(model)
    units, alpha, P, scaf = build_units(model, av, mv, at, mt, k, device, seed)
    n_comp = len(component_index(av, mv)) + len(component_index(at, mt))
    full = audit(embed(av, mv), embed(at, mt), y, sub)
    print(f"[{model}] {len(units)} PC units from {n_comp} components (K<={k})\n"
          f"    FULL MODEL faces->human={full['face_human']:.2f}%  "
          f"non-human={full['face_nonhuman']:.2f}%  crime={full['face_crime']:.2f}%  "
          f"primates->non-human={full['primate_nonhuman']:.2f}%", flush=True)

    rows = [dict(model=model, variant="full model", rep=-1, size=len(units), **full)]
    traces, moves = [], []
    for rep in range(reps):
        g = torch.Generator().manual_seed(seed + rep)
        cal = []
        for c in (0, 8):
            w = (y == c).nonzero(as_tuple=True)[0]
            cal += w[torch.randperm(w.numel(), generator=g)[:n_cal].to(w.device)].tolist()
        cal = torch.tensor(sorted(cal), device=device)
        held = torch.ones(av.shape[0], dtype=torch.bool, device=device)
        held[cal] = False
        for variant, cfg in VARIANTS.items():
            pool, tr = local_search_pc(units, alpha, P, scaf, tau_inv, cal, y[cal],
                                       at.shape[0], max_steps=max_steps, **cfg)
            tj = replay_audit(units, alpha, P, scaf, tr, cfg["init"], y, sub, held).merge(
                tr[["step", "loss", "n_kept"]], on="step")
            base = dict(model=model, variant=variant, rep=rep, regime="bias")
            rows.append(dict(method="selected", size=len(pool), **base,
                             **tj.iloc[-1].drop("step").to_dict()))
            traces.append(tj.assign(**base))
            for r in tr.itertuples():
                for (t, c, j), sign in r.applied:
                    moves.append(dict(**base, step=r.step, tower=t, kind=c[0], layer=c[1],
                                      head=c[2], pc=j, action="add" if sign > 0 else "drop",
                                      loss=r.loss))
            print(f"  [rep{rep} {variant:14s}] kept={len(pool):5d}/{len(units)}  "
                  f"faces->non-human={tj.iloc[-1].face_nonhuman:.2f}%  "
                  f"crime={tj.iloc[-1].face_crime:.2f}%  "
                  f"disparity={tj.iloc[-1].disparity:.2f}  "
                  f"primates->non-human={tj.iloc[-1].primate_nonhuman:.2f}%", flush=True)
            rng = np.random.default_rng(seed + rep)
            tix = torch.arange(at.shape[0], device=device)
            for r in range(n_random):
                sel = [units[i] for i in rng.choice(len(units), size=len(pool), replace=False)]
                Ev, Et = scaf["vision"].clone(), scaf["text"].clone()
                for t, c, j in sel:
                    idx = slice(None) if t == "vision" else tix
                    dd = alpha[(t, c)][idx, j].unsqueeze(1) * P[(t, c)][j]
                    if t == "vision":
                        Ev = Ev + dd
                    else:
                        Et = Et + dd
                rows.append(dict(model=model, variant=variant, regime="bias",
                                 method="random units", rep=r, size=len(pool),
                                 **audit(Ev, Et, y, sub, held)))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"bias_pc_eval_{model}.csv"), index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        os.path.join(OUT, f"bias_pc_trace_{model}.csv"), index=False)
    pd.DataFrame(moves).to_csv(os.path.join(OUT, f"bias_pc_moves_{model}.csv"), index=False)
    print(f"[{model}] wrote bias_pc_{{eval,trace,moves}}", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("bias audit at PC granularity", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", default=8, type=int)
    p.add_argument("--n_cal", default=16, type=int)
    p.add_argument("--max_steps", default=10 ** 6, type=int,
                   help="cap; default = run to convergence")
    p.add_argument("--reps", default=2, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.k, a.max_steps, a.reps)
