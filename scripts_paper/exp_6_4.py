"""Paper section 6.4 - Linear approximation of the top components.

Replaces the activation space of a set of target components by a rank-k linear approximation
and measures how much COCO retrieval survives. Four bases, all rank k, all mean-centred:

  pc      the component's own top-k PCs (the variance-optimal basis; upper bound)
  text    for each of the top-k PCs, the text embedding of D_T maximising |cos(p_j, e)|
  image   idem from the image embeddings of D_I
  both    idem from the concatenation of the two pools

The last three are exactly PCLens' labelling rule (|cos| because a PC is an axis, not a
direction), so the curve says how much of a component is carried by the *nameable* part of
its spectrum. Applied per tower and to both towers at once; non-target components stay exact.

Targets: `topB{N}` = top-N components by Metric B (section 6.2), `late{L}` = every head of the
last L layers.

  python -m scripts_paper.exp_6_4 --models ViT-B-32
Writes output_dir/results_paper/6_4/linear_approx_{model}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts_paper.pclens_core import (ACT_DIR, MODELS, RES_DIR, comp_vec, component_index,
                                       embed, load_tower, retrieval_metrics)

OUT = os.path.join(RES_DIR, "6_4")
SCORES = os.path.join(RES_DIR, "6_2")
TEXT_POOL = "top_1500_nouns_5_sentences_imagenet_clean"      # D_T embeddings
IMAGE_POOL = "imagenet_embeddings_{model}_seed_69.npy"       # D_I embeddings


def load_pools(model, device, dtype=torch.float32, center=True):
    """Unit-norm concept pools: D_T sentence embeddings and D_I image embeddings.

    `center` subtracts each pool's own mean before re-normalising: CLIP embeddings of one
    modality sit in a narrow cone, so their shared direction (the modality gap) would otherwise
    dominate both the |cos| selection and the span. This is the mean-centering PCLens already
    applies when reconstructing a head from embeddings [9]."""
    pools = {}
    pt = os.path.join(ACT_DIR, f"{TEXT_POOL}_{model}.npy")
    pi = os.path.join(ACT_DIR, IMAGE_POOL.format(model=model))
    for name, path in (("text", pt), ("image", pi)):
        if os.path.exists(path):
            E = torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
            E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if center:
                E = E - E.mean(0, keepdim=True)
            pools[name] = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if "text" in pools and "image" in pools:
        pools["both"] = torch.cat([pools["text"], pools["image"]], 0)
    return pools


@torch.no_grad()
def component_basis(X, kmax, pools, bases):
    """Per-component cache: mean, top-kmax PCs, and the pool vectors each PC selects.

    The greedy pool selection walks the PCs in order and never repeats, so the choice for a
    prefix of k PCs is just the first k entries of the kmax-run -> one pass serves every k."""
    mu = X.mean(0)
    _, _, Vh = torch.linalg.svd(X - mu, full_matrices=False)
    P = Vh[:kmax]
    out = {"mu": mu, "pc": P}
    for b in bases:
        if b == "pc" or b not in pools:
            continue
        # candidate window must exceed kmax, otherwise late PCs find every candidate taken
        w = min(4 * P.shape[0] + 64, pools[b].shape[0])
        sim = (P @ pools[b].T).abs()                              # [kmax, |pool|]
        top = torch.topk(sim, w, dim=1).indices.tolist()
        chosen, used = [], set()
        for j in range(P.shape[0]):
            pick = next((i for i in top[j] if i not in used), top[j][0])
            used.add(pick); chosen.append(pick)
        out[b] = pools[b][chosen]
    return out


@torch.no_grad()
def approx_component(X, k, basis, cache):
    """Rank-k approximation of X [N,d]: mean + projection of the centred data on a k-dim span."""
    mu, P = cache["mu"], cache[basis][:k]
    Q, _ = torch.linalg.qr(P.T)                # pool vectors are not orthogonal
    return mu + ((X - mu) @ Q) @ Q.T


def target_components(model, tower, spec, a, m):
    """Resolve a target spec to a component list."""
    if spec.startswith("late"):
        L = int(spec[4:])
        return [("attn", l, h) for l in range(max(0, a.shape[1] - L), a.shape[1])
                for h in range(a.shape[2])]
    if spec.startswith("topB"):
        N = int(spec[4:])
        p = os.path.join(SCORES, f"component_scores_{model}.csv")
        df = pd.read_csv(p)
        df = df[df.tower == tower].sort_values("metricB").head(N)     # ascending = important
        return [(r.kind, int(r.layer), int(r.head)) for r in df.itertuples()]
    raise ValueError(spec)


@torch.no_grad()
def approx_embedding(a, m, comps, k, basis, caches, base):
    """Embedding with `comps` replaced by their rank-k approximation, everything else exact.

    Additive (never copies the [N,L,H,d] tensor), so it scales to ViT-H-14."""
    out = base.clone()
    for c in comps:
        X = comp_vec(a, m, c)
        out += approx_component(X, k, basis, caches[c]) - X
    return out


@torch.no_grad()
def run_model(model, device="cuda:0", ks=(1, 2, 4, 8, 16, 32, 64, 128, 256),
              targets=("late4", "topB32"), bases=("pc", "text", "image", "both")):
    os.makedirs(OUT, exist_ok=True)
    av, mv = load_tower(model, "vision", device=device)
    at, mt = load_tower(model, "text", device=device)
    pools = load_pools(model, device)
    X0, Y0 = embed(av, mv), embed(at, mt)
    base = retrieval_metrics(X0, Y0)
    acts = {"vision": (av, mv), "text": (at, mt)}

    rows = [dict(model=model, tower="none", target="none", basis="exact", k=-1,
                 n_comp=0, cos_recon=1.0, **base)]
    for spec in targets:
        comps = {t: target_components(model, t, spec, *acts[t]) for t in ("vision", "text")}
        caches = {t: {c: component_basis(comp_vec(*acts[t], c), max(ks), pools, bases)
                      for c in comps[t]} for t in ("vision", "text")}
        for basis in bases:
            if basis != "pc" and basis not in pools:
                continue
            for k in ks:
                approx = {"vision": approx_embedding(av, mv, comps["vision"], k, basis,
                                                     caches["vision"], X0),
                          "text": approx_embedding(at, mt, comps["text"], k, basis,
                                                   caches["text"], Y0)}
                for tower in ("vision", "text", "joint"):
                    Xa = approx["vision"] if tower in ("vision", "joint") else X0
                    Ya = approx["text"] if tower in ("text", "joint") else Y0
                    ref, cur = (X0, Xa) if tower != "text" else (Y0, Ya)
                    cos = torch.nn.functional.cosine_similarity(ref, cur, dim=-1).mean().item()
                    rows.append(dict(model=model, tower=tower, target=spec, basis=basis, k=k,
                                     n_comp=len(comps["vision"]) + len(comps["text"]),
                                     cos_recon=cos, **retrieval_metrics(Xa, Ya)))
                print(f"[{model}] {spec} {basis} k={k}: joint R@1_i2t="
                      f"{rows[-1]['R@1_i2t']:.2f} (base {base['R@1_i2t']:.2f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"linear_approx_{model}.csv"), index=False)
    print(f"[{model}] wrote {os.path.join(OUT, f'linear_approx_{model}.csv')}")
    return df


def get_args_parser():
    p = argparse.ArgumentParser("6.4 linear approximation", add_help=False)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--targets", nargs="+", default=["late4", "topB32"])
    return p


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    for m in args.models:
        run_model(m, args.device, tuple(args.ks), tuple(args.targets))
