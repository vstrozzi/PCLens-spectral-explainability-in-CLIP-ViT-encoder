"""Circuit / interaction-matrix reshaping under the Procrustes W.
M(y,c) = A(c) B(y)^T  (text-comp x image-comp);  rotate-image -> M'(y,c)=A(c) W B(y)^T.
Sum(M)=cos(e_c^classname, f);  Sum(M')=cos(e_c, W f).
Caches arrays for plotting. Memory-frugal (chunked, float32), cgroup-safe.
"""
import os, json, gc
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(k, "3")
import numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "pa", os.path.join(os.path.dirname(__file__), "procrustes_align.py"))
pa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pa)
sys.path.insert(0, pa.ROOT)

OUT = os.path.dirname(__file__)
D, SEED = pa.D, pa.SEED

def load_components(model, ds):
    attn = np.load(f"{D}/{ds}_attn_{model}_seed_{SEED}.npy")
    mlp  = np.load(f"{D}/{ds}_mlp_{model}_seed_{SEED}.npy")
    N, L, H = attn.shape[0], attn.shape[1], attn.shape[2]
    B = np.concatenate([attn.reshape(N, L * H, 512), mlp], 1)     # [N,157,512] f32
    img_lab = [f"a{l}.{h}" for l in range(L) for h in range(H)] + [f"m{l}" for l in range(mlp.shape[1])]
    del attn, mlp; gc.collect()
    ta = np.load(f"{D}/{ds}_classnames_attn_text_{model}_seed_{SEED}.npy")
    tm = np.load(f"{D}/{ds}_classnames_mlp_text_{model}_seed_{SEED}.npy")
    C, Lt, Ht = ta.shape[0], ta.shape[1], ta.shape[2]
    A = np.concatenate([ta.reshape(C, Lt * Ht, 512), tm], 1)      # [C,109,512] f32
    txt_lab = [f"a{l}.{h}" for l in range(Lt) for h in range(Ht)] + [f"m{l}" for l in range(tm.shape[1])]
    del ta, tm; gc.collect()
    lab = np.load(f"{D}/{ds}_labels_{model}_seed_{SEED}.npy")
    return A, B, lab, img_lab, txt_lab

def compute(model="ViT-B-16", ds="imagenet"):
    A, B, lab, img_lab, txt_lab = load_components(model, ds)
    N, nI = B.shape[0], B.shape[1]; nT = A.shape[1]
    F = B.sum(1).astype(np.float64)
    clf = np.load(f"{D}/{ds}_classifier_{model}.npy").astype(np.float64)
    Wp = pa.procrustes(F, clf[:, lab].T).astype(np.float32)        # attract-to-true

    # ---- mean interaction matrices M̄, M̄' (chunked over samples) ----
    Mbar = np.zeros((nT, nI)); Mpar = np.zeros((nT, nI))
    for s in range(0, N, 500):
        Ac = A[lab[s:s+500]]                    # [c,nT,512]
        Bc = B[s:s+500]                         # [c,nI,512]
        Ar = Ac @ Wp                            # rows W^T a_i  -> gives M' when dotted with b_j
        Mbar += np.einsum('cid,cjd->ij', Ac, Bc)
        Mpar += np.einsum('cid,cjd->ij', Ar, Bc)
    Mbar /= N; Mpar /= N
    dM = Mpar - Mbar

    # ---- per-component marginals of the TRUE-class score (rotate-image) ----
    e_true = A[lab].sum(1)                       # [N,512]
    Pimg = (e_true @ Wp - e_true)                # (W^T e_true - e_true)
    dimg = np.einsum('nd,njd->nj', Pimg, B).mean(0)          # [nI]
    Wf = (F @ Wp.T).astype(np.float32)           # W f
    dtxt = np.einsum('nid,nd->ni', A[lab], (Wf - F.astype(np.float32))).mean(0)   # [nT]

    # ---- per-sample stream decomposition of the TRUE-class score (for 2D scatter) ----
    attn_sum = B[:, :144, :].sum(1)              # [N,512] attention-head stream
    mlp_sum  = B[:, 144:, :].sum(1)              # [N,512] MLP stream
    et = e_true.astype(np.float32)
    scat = dict(
        cos_before = (F.astype(np.float32) * et).sum(1),
        cos_after  = (Wf * et).sum(1),
        attn_before = (attn_sum * et).sum(1),
        attn_after  = ((attn_sum @ Wp.T) * et).sum(1),
        mlp_before  = (mlp_sum * et).sum(1),
        mlp_after   = ((mlp_sum @ Wp.T) * et).sum(1),
        lab = lab.astype(np.int64),
    )
    del attn_sum, mlp_sum; gc.collect()

    dcos = float(((Wf * e_true).sum(1) - (F * e_true).sum(1)).mean())
    print(f"[{model}/{ds}] Δcos={dcos:.4f}  Σdimg={dimg.sum():.4f}  Σdtxt={dtxt.sum():.4f}  "
          f"ΣΔM={dM.sum():.4f}  (should all match)")
    print("  MLP share of |dimg|:", float(np.abs(dimg[144:]).sum()/np.abs(dimg).sum()))

    # ---- a few sample interaction matrices (diverse classes) ----
    from utils.datasets_constants.imagenet_classes import imagenet_classes as CN
    rng = np.random.default_rng(0)
    picks = []
    for c in [1, 340, 933, 555]:                 # goldfish, zebra, cheeseburger, fire engine (imagenet idx)
        idx = np.where(lab == c)[0]
        if len(idx): picks.append(int(idx[0]))
    samples = {}
    for n in picks:
        Msamp = A[lab[n]] @ B[n].T                # [nT,nI]
        Msamp_p = (A[lab[n]] @ Wp) @ B[n].T
        samples[n] = dict(cls=int(lab[n]), name=CN[lab[n]],
                          M=Msamp.astype(np.float32), Mp=Msamp_p.astype(np.float32))

    np.savez(os.path.join(OUT, f"circuit_{model}_{ds}.npz"),
             Mbar=Mbar, Mpar=Mpar, dM=dM, dimg=dimg, dtxt=dtxt,
             img_lab=np.array(img_lab), txt_lab=np.array(txt_lab),
             sample_ids=np.array(picks),
             **{f"M_{n}": samples[n]["M"] for n in picks},
             **{f"Mp_{n}": samples[n]["Mp"] for n in picks},
             sample_names=np.array([samples[n]["name"] for n in picks]),
             nI=nI, nT=nT, dcos=dcos, **{f"scat_{k}": v for k, v in scat.items()})
    del B, A; gc.collect()
    return dcos

if __name__ == "__main__":
    for m in ["ViT-B-16", "ViT-B-32"]:
        compute(m, "imagenet")
