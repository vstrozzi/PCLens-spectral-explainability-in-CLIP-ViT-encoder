"""Did the searches actually converge, or did they just run out of steps?

A greedy search that stops because it hit `max_steps` looks exactly like one that stopped because
no move improved the loss - same CSV, same monotone loss curve, plausible numbers. The difference
is that the first is a truncated search whose result depends on an arbitrary budget, and no
conclusion drawn from it is safe.

This has bitten this project repeatedly (6.4 k=64, the greedy pool at 96, PC-level at 300,
Waterbirds at 60 and again at 64/400), so it is a standing check rather than a per-script one.

The test, per model: a cap makes a large share of that model's runs stop at exactly the same step,
and that step is the largest one seen. Converged runs pile up at no particular value, so a few runs
coinciding by chance somewhere in the middle is not evidence of anything and is not flagged.

  python -m scripts_paper.audit_convergence
Exit status is 1 if anything looks truncated, so it can gate a results pass.
"""
import glob
import os
import sys
from collections import Counter

import pandas as pd

from scripts_paper.pclens_core import RES_DIR

# every family whose CSV carries a per-step trace
FAMILIES = ["waterbirds_trace", "waterbirds_pc_trace", "bias_audit_trace", "bias_pc_trace",
            "hardneg_trace_negation", "hardneg_trace_count", "hardneg_trace_order",
            "typographic_trace", "counting_trace", "insub_trace", "greedy_pool_trace"]
KEYS = ["model", "regime", "variant", "rep", "task", "level", "subset"]


def audit(family, share=0.3, min_runs=3):
    fs = sorted(glob.glob(os.path.join(RES_DIR, "6_2", f"{family}_*.csv")))
    if not fs:
        return "missing", None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "step" not in d.columns:
        return "no-step-column", None
    keys = [k for k in KEYS if k in d.columns]
    g = d.groupby(keys).step.max().reset_index()
    suspect = []
    for model, gm in (g.groupby("model") if "model" in g else [("all", g)]):
        top = gm.step.max()
        hits = int((gm.step == top).sum())
        if hits >= min_runs and hits / len(gm) >= share and top > 1:
            suspect.append((model, top, hits, len(gm)))
    return g, suspect


def main():
    bad = False
    for fam in FAMILIES:
        g, suspect = audit(fam)
        if isinstance(g, str):
            print(f"[--] {fam:26s} {g}")
            continue
        if suspect:
            bad = True
            print(f"[!!] {fam:26s} {len(g):4d} runs - CAP suspected:")
            for model, top, hits, tot in suspect:
                print(f"       {model:10s} {hits:3d}/{tot:3d} runs stop at EXACTLY step {top}")
        else:
            print(f"[ok] {fam:26s} {len(g):4d} runs, max step {g.step.max()}, "
                  f"{g.step.nunique()} distinct stopping points")
    if bad:
        print("\nRe-run the flagged families with the step cap removed before using their numbers.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
