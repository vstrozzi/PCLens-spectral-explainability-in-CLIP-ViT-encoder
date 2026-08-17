"""Shared loading / retrieval / component algebra for the paper experiments (6.2-6.4).

Conventions (match the extraction scripts):
  attn  [N, L, H, d]   already divided by ||encoder output||  -> they sum, with mlp, to the
  mlp   [N, L+1, d]    L2-NORMALISED embedding of the sample.
So ||c_a(m_i)|| as stored IS the norm fraction w_a(m_i) of Eq. (5), and
  sum over all components == x_i / ||x_i||.

A *component* is addressed by (kind, layer, head) with kind in {"attn","mlp"} and head=-1 for MLP.
Everything is torch, float32, on `device`; arrays are loaded lazily with mmap and moved per-need.
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT_DIR = os.path.join(ROOT, "output_dir", "activations_and_datasets_idxs_69")
RES_DIR = os.path.join(ROOT, "output_dir", "results_paper")

MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "RN50", "RN101"]
VIT_MODELS = [m for m in MODELS if m.startswith("ViT")]
# the checkpoint each model was decomposed from - the ViTs are LAION, the ResNets are OpenAI, and
# that split matters for any result compared against the original CLIP paper
MODEL_PRETRAINED = {"ViT-B-32": "laion2b_s34b_b79k", "ViT-B-16": "laion2b_s34b_b88k",
                    "ViT-L-14": "laion2b_s32b_b82k", "ViT-H-14": "laion2b_s32b_b79k",
                    "RN50": "openai", "RN101": "openai"}


# --------------------------------------------------------------------------- loading
def _p(name):
    return os.path.join(ACT_DIR, name)


def load_tower(model, tower, dataset="coco", text_set="coco_karpathy_test", seed=69,
               device="cpu", dtype=torch.float32, act_dir=None):
    """Return (attn [N,L,H,d], mlp [N,L+1,d]) for one tower, as stored (already /||out||).

    tower="vision" reads the image run of `dataset`; tower="text" reads the text run of `text_set`.
    ResNet vision towers have the same [N,L+1,H,d] layout (block x pool-head).
    `act_dir` overrides the activation directory (used for the pre-projection variant, whose
    components live in the encoder's own width and do NOT sum to the shared-space embedding)."""
    d_ = (lambda n: os.path.join(act_dir, n)) if act_dir else _p
    if tower == "vision":
        a, m = d_(f"{dataset}_attn_{model}_seed_{seed}.npy"), d_(f"{dataset}_mlp_{model}_seed_{seed}.npy")
    else:
        a = d_(f"{text_set}_attn_text_{model}_seed_{seed}.npy")
        m = d_(f"{text_set}_mlp_text_{model}_seed_{seed}.npy")
    attn = torch.from_numpy(np.load(a)).to(device=device, dtype=dtype)
    mlp = torch.from_numpy(np.load(m)).to(device=device, dtype=dtype) if os.path.exists(m) else None
    return attn, mlp


def logit_scale(model):
    """Learned 1/tau of the checkpoint (dumped by extract_coco.sbatch); 100.0 if missing."""
    p = _p(f"logit_scale_{model}.json")
    return json.load(open(p))["logit_scale"] if os.path.exists(p) else 100.0


def component_index(attn, mlp):
    """Ordered list of components: all attn heads (layer-major) then all mlp slots."""
    comps = [("attn", l, h) for l in range(attn.shape[1]) for h in range(attn.shape[2])]
    if mlp is not None:
        comps += [("mlp", l, -1) for l in range(mlp.shape[1])]
    return comps


def comp_vec(attn, mlp, comp):
    kind, l, h = comp
    return attn[:, l, h] if kind == "attn" else mlp[:, l]


def embed(attn, mlp):
    """Sum of all components == the L2-normalised encoder output."""
    out = attn.sum(dim=(1, 2))
    if mlp is not None:
        out = out + mlp.sum(dim=1)
    return out


# --------------------------------------------------------------------------- ablation
def comp_delta(attn, mlp, comp, mean_attn, mean_mlp):
    """c_a(m_i) - mean_a  [N, d]: what mean-ablating this component removes from the sum."""
    kind, l, h = comp
    return (attn[:, l, h] - mean_attn[l, h]) if kind == "attn" else (mlp[:, l] - mean_mlp[l])


def all_mean_embedding(attn, mlp, mean_attn, mean_mlp):
    """The fully mean-ablated embedding: a single constant vector (sum of every component mean)."""
    out = mean_attn.sum(dim=(0, 1))
    if mlp is not None:
        out = out + mean_mlp.sum(dim=0)
    return out


def ablated_embedding(attn, mlp, ablate, mean_attn=None, mean_mlp=None, keep_only=False):
    """Embedding after mean-ablating (or, with keep_only, after keeping ONLY) `ablate`.

    Mean-ablation replaces a per-sample contribution by a fixed mean vector, removing the
    sample-specific information while preserving the bias. Built additively from the
    per-component deltas (never a copy of the [N,L,H,d] tensor), so it also scales to H-14.
    Returns the NON-normalised sum; retrieval renormalises."""
    ma = attn.mean(0) if mean_attn is None else mean_attn
    mm = (mlp.mean(0) if mean_mlp is None else mean_mlp) if mlp is not None else None
    ablate = list(ablate)
    if keep_only:
        out = all_mean_embedding(attn, mlp, ma, mm).expand(attn.shape[0], -1).clone()
        for c in ablate:
            out += comp_delta(attn, mlp, c, ma, mm)
        return out
    out = embed(attn, mlp)
    for c in ablate:
        out -= comp_delta(attn, mlp, c, ma, mm)
    return out


# --------------------------------------------------------------------------- retrieval
@torch.no_grad()
def retrieval_metrics(X, Y, ks=(1, 5, 10)):
    """1-to-1 retrieval on paired rows (X[i] <-> Y[i]). Returns R@k both directions + median rank.

    Cosine ranking with renormalised embeddings, i.e. exactly CLIP's inference rule."""
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Yn = Y / Y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    S = Xn @ Yn.T                                    # [N_img, N_txt]
    n = S.shape[0]
    gt = torch.arange(n, device=S.device)
    out = {}
    for name, M in (("i2t", S), ("t2i", S.T)):
        # 0-based rank of the ground truth, ties counted against us: a fully mean-ablated
        # (constant) embedding scores every candidate equally and must read as chance, not 100%.
        gt_score = M[gt, gt].unsqueeze(1)
        rank = (M > gt_score).sum(dim=1) + (M == gt_score).sum(dim=1) - 1
        for k in ks:
            out[f"R@{k}_{name}"] = (rank < k).float().mean().item() * 100
        out[f"medr_{name}"] = (rank.float().median().item() + 1)
        out[f"meanr_{name}"] = (rank.float().mean().item() + 1)
    out["rsum"] = sum(out[f"R@{k}_{d}"] for k in ks for d in ("i2t", "t2i"))
    return out


# --------------------------------------------------------------------------- metrics A / B
@torch.no_grad()
def pcs_of(vecs, var=0.99, kmax=50, center=True):
    """PCA of a component's activation matrix. Returns (U [k,d] unit dirs, rho [k] PVE fractions,
    sv2 [k] squared singular values, mean [d]). k = min(kmax, #PCs reaching `var`)."""
    mu = vecs.mean(0)
    Xc = vecs - mu if center else vecs
    # economy SVD on [N,d]; d is small (<=1024) so this is cheap
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    sv2 = S ** 2
    pve = sv2 / sv2.sum().clamp_min(1e-30)
    k = int(torch.searchsorted(torch.cumsum(pve, 0), torch.tensor(var, device=pve.device)).item()) + 1
    k = max(1, min(k, kmax, Vh.shape[0]))
    U = Vh[:k]
    U = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return U, pve[:k], sv2[:k], mu


@torch.no_grad()
def A_metric(Ua, ra, Ub, rb):
    """User's PVE-weighted PC-cosine with Hungarian matching (playground_exploration E6)."""
    from scipy.optimize import linear_sum_assignment
    W = (torch.sqrt(ra[:, None] * rb[None, :]) * (Ua @ Ub.T).abs()).cpu().numpy()
    ri, ci = linear_sum_assignment(-W)
    return float(W[ri, ci].sum())


@torch.no_grad()
def align_geo(Ua, sa, Ub, sb):
    """Lambda_geo = sum_i sum_j SV_i^2 SV_j^2 |cos(PC_i,PC_j)| (playground_exploration E6b)."""
    return float((sa[:, None] * sb[None, :] * (Ua @ Ub.T).abs()).sum())


@torch.no_grad()
def metric_B_component(c_a, Y, tau_inv, neg_idx):
    """Literal Eq. (19)-(20) per component, aggregated over partners.

    Because e^{l_ij - l_ii} = prod_{a,b} r_ab, the per-component factor is the product over b,
    which telescopes exactly to
        r_a(i,j) = exp( (1/tau) * w_a(i) * [ cos(c_a(i), y_j) - cos(c_a(i), y_i) ] ),
    with w_a(i) = ||c_a(i)|| (activations are stored pre-divided by ||x_i||).

    Returns E_{i, j~neg}[r_a] (<1 = discriminative), and E[log r_a] as a stable secondary score.
    `Y` are the OTHER tower's full (unnormalised-sum) embeddings; `neg_idx` [N, M] sampled negatives.
    """
    w = c_a.norm(dim=-1)                                   # [N]
    ch = c_a / c_a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Yn = Y / Y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_all = ch @ Yn.T                                    # [N, N]
    n = cos_all.shape[0]
    pos = cos_all[torch.arange(n, device=cos_all.device), torch.arange(n, device=cos_all.device)]
    neg = torch.gather(cos_all, 1, neg_idx)                # [N, M]
    log_r = tau_inv * w[:, None] * (neg - pos[:, None])
    return float(log_r.exp().mean()), float(log_r.mean())


def sample_negatives(n, m, device, seed=69):
    """[n, m] random negative indices j != i."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, n - 1, (n, m), generator=g)
    shift = (idx >= torch.arange(n)[:, None]).long()
    return (idx + shift).to(device)
