"""Per-group Waterbirds numbers for every selector, rebuilt from the pools already saved.

`exp_waterbirds` records only total / worst-class / worst-group; a spurious-correlation result is
not readable from those alone, because a method can lift the worst group by flattening a strong one.
This re-evaluates each saved pool and writes EVERY cell:

    landbird-on-land, landbird-on-water, waterbird-on-land, waterbird-on-water

The two minority cells (bird on the background it is NOT correlated with) are where the shortcut
shows. No search is re-run - the pools are read back from waterbirds_pool_{model}.csv - so this is
cheap and cannot disagree with the numbers already reported.

  python -m scripts_paper.exp_waterbirds_table --device cpu
Writes output_dir/results_paper/6_2/waterbirds_table_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.exp_greedy_pool import _load_task, keep_only
from scripts_paper.exp_waterbirds import TASK, groups
from scripts_paper.pclens_core import MODELS, RES_DIR, component_index, embed

OUT = os.path.join(RES_DIR, "6_2")
CELL = {(0, 0): "landbird_land", (0, 1): "landbird_water",
        (1, 0): "waterbird_land", (1, 1): "waterbird_water"}


@torch.no_grad()
def all_cells(Ev, Et, y, place):
    """Total, per-class, per-group and the two summary minima, in one dict."""
    L = torch.nn.functional.normalize(Ev, dim=1) @ torch.nn.functional.normalize(Et, dim=1).T
    ok = (L.argmax(1) == y).float()
    out = {"total": float(ok.mean() * 100)}
    for c in (0, 1):
        m = y == c
        out[f"class_{c}"] = float(ok[m].mean() * 100) if m.any() else float("nan")
    for (c, p), name in CELL.items():
        m = (y == c) & (place == p)
        out[name] = float(ok[m].mean() * 100) if m.any() else float("nan")
    out["worst_class"] = min(out["class_0"], out["class_1"])
    out["worst_group"] = min(out[n] for n in CELL.values() if not np.isnan(out[n]))
    return out


def _cal_mask(y, place, regime, rep, n_cal, seed, device):
    """Reproduce the calibration draw of the original run so we score the same held-out set."""
    strata = y if regime == "class" else (2 * y + place)
    g = torch.Generator().manual_seed(int(seed) + int(rep))   # rep arrives as numpy int64
    cal = []
    for s in strata.unique():
        w = (strata == s).nonzero(as_tuple=True)[0]
        cal += w[torch.randperm(w.numel(), generator=g)[:n_cal].to(w.device)].tolist()
    held = torch.ones(y.shape[0], dtype=torch.bool, device=device)
    held[torch.tensor(cal, device=device)] = False
    return held


def run_model(model, device="cpu", seed=69, n_cal=8, n_random=5):
    pool_p = os.path.join(OUT, f"waterbirds_pool_{model}.csv")
    if not os.path.exists(pool_p):
        print(f"[skip] missing {pool_p}")
        return
    av, mv, at, mt, _ = _load_task(model, TASK, device, seed)
    y, place = groups(seed)
    y, place = y.to(device), place.to(device)
    allc = ([("vision", c) for c in component_index(av, mv)] +
            [("text", c) for c in component_index(at, mt)])

    rows = [dict(model=model, regime="-", variant="-", method="full model", rep=-1,
                 size=len(allc), **all_cells(embed(av, mv), embed(at, mt), y, place))]
    pools = pd.read_csv(pool_p)
    for (regime, variant, rep), g in pools.groupby(["regime", "variant", "rep"]):
        sel = [(r.tower, (r.kind, int(r.layer), int(r.head))) for r in g.itertuples()]
        held = _cal_mask(y, place, regime, rep, n_cal, seed, device)
        Ev, Et = keep_only(av, mv, at, mt, sel)
        rows.append(dict(model=model, regime=regime, variant=variant, method="selected", rep=rep,
                         size=len(sel), **all_cells(Ev[held], Et, y[held], place[held])))
        rng = np.random.default_rng(int(seed) + int(rep))
        for r in range(n_random):
            rsel = [allc[i] for i in rng.choice(len(allc), size=len(sel), replace=False)]
            Ev, Et = keep_only(av, mv, at, mt, rsel)
            rows.append(dict(model=model, regime=regime, variant=variant, method="random pool",
                             rep=r, size=len(sel),
                             **all_cells(Ev[held], Et, y[held], place[held])))
        print(f"  [{regime} {variant:14s} rep{rep}] size={len(sel):4d} "
              f"total={rows[-1 - n_random]['total']:.2f} "
              f"worst-group={rows[-1 - n_random]['worst_group']:.2f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, f"waterbirds_table_{model}.csv"), index=False)
    print(f"[{model}] wrote waterbirds_table ({len(d)} rows)", flush=True)


def get_args_parser():
    p = argparse.ArgumentParser("waterbirds per-group table", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_cal", default=8, type=int, help="must match the run that made the pools")
    p.add_argument("--n_random", default=5, type=int)
    p.add_argument("--seed", default=69, type=int)
    return p


if __name__ == "__main__":
    a = get_args_parser().parse_args()
    for m in a.models:
        run_model(m, a.device, a.seed, a.n_cal, a.n_random)
