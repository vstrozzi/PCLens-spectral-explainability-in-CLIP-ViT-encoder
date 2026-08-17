"""
Orthogonal-Procrustes alignment W* = argmax_{WᵀW=I} Σ_i cossim(e_i, W f_i) = V Uᵀ
over CLIP image/text encoder outputs. Positive vs negative pairs, combos, transfer.

Data: output_dir/activations_and_datasets_idxs_69  (PRS components, already /‖out‖ so
component-sum = unit embedding; classifier cols are unit-norm template-mean text).
"""
import os, json
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(k, "2")
import numpy as np

ROOT = "/home/virgilio/contrastive_learning/PCLens-spectral-explainability-in-CLIP-ViT-encoder"
D = f"{ROOT}/output_dir/activations_and_datasets_idxs_69"
SEED = 69
MODELS = ["ViT-B-16", "ViT-B-32"]
DATASETS = ["imagenet", "CIFAR100"]

def load(model, ds):
    """Return f [N,512] (unit image emb = component sum), clf [512,C], labels [N]."""
    attn = np.load(f"{D}/{ds}_attn_{model}_seed_{SEED}.npy")     # [N,L,H,512]
    mlp  = np.load(f"{D}/{ds}_mlp_{model}_seed_{SEED}.npy")      # [N,L+1,512]
    N = attn.shape[0]
    f = attn.reshape(N, -1, 512).sum(1) + mlp.sum(1)            # unit-norm embedding
    del attn, mlp
    clf = np.load(f"{D}/{ds}_classifier_{model}.npy").astype(np.float64)   # [512,C]
    lab = np.load(f"{D}/{ds}_labels_{model}_seed_{SEED}.npy")
    return f.astype(np.float64), clf, lab

def procrustes(F, G, repel=False):
    """M = Σ f_i g_iᵀ = FᵀG = U S Vᵀ.
    repel=False -> W = V Uᵀ   : maximize Σ cossim(g_i, W f_i)  (attract, min L2)
    repel=True  -> W = −V Uᵀ  : minimize Σ cossim(g_i, W f_i)  (repel,  max L2)."""
    M = F.T @ G
    U, S, Vt = np.linalg.svd(M)
    W = Vt.T @ U.T
    return -W if repel else W

def zshot_acc(F, clf, lab, W=None):
    X = F if W is None else F @ W.T          # rows = W f_i
    s = X @ clf                               # (W f_i)·e_c
    return float((s.argmax(1) == lab).mean())

def neg_targets(clf, lab, rng):
    """One random wrong class per sample -> target embedding [N,512]."""
    C = clf.shape[1]
    j = rng.integers(0, C, size=len(lab))
    same = j == lab
    while same.any():
        j[same] = rng.integers(0, C, size=int(same.sum()))
        same = j == lab
    return clf[:, j].T                        # [N,512]

def fit_pair_matrices(F, clf, lab, rng):
    """W_pos: attract each image to its TRUE class (max cossim).
    W_neg: repel each image from a random WRONG class (min cossim / max L2)."""
    G_pos = clf[:, lab].T                     # true-class embedding per sample
    G_neg = neg_targets(clf, lab, rng)        # random wrong-class embedding
    return procrustes(F, G_pos), procrustes(F, G_neg, repel=True)

def orthogonality_err(W):
    d = W.shape[0]
    return float(np.abs(W @ W.T - np.eye(d)).max())

def main():
    out = {}
    for model in MODELS:
        rng = np.random.default_rng(SEED)
        data = {ds: load(model, ds) for ds in DATASETS}
        res = {"baseline": {}, "orth_err": {}, "fit": {}}

        # baseline
        for ds, (F, clf, lab) in data.items():
            res["baseline"][ds] = zshot_acc(F, clf, lab)

        # fit W on each source dataset, eval everywhere
        Ws = {}
        for src in DATASETS:
            F, clf, lab = data[src]
            Wp, Wn = fit_pair_matrices(F, clf, lab, rng)
            Ws[src] = {"Wpos": Wp, "Wneg": Wn}
            res["orth_err"][src] = {"Wpos": orthogonality_err(Wp), "Wneg": orthogonality_err(Wn)}
            # sanity: mean cos(true class) before/after Wpos (attract -> up)
            G = clf[:, lab].T
            c0 = float((F * G).sum(1).mean())
            c1 = float(((F @ Wp.T) * G).sum(1).mean())
            res["orth_err"][src]["cos_true_before_after"] = [c0, c1]
            # sanity: mean cos to the sampled wrong class before/after Wneg (repel -> down)
            Gn = neg_targets(clf, lab, np.random.default_rng(SEED))
            n0 = float((F * Gn).sum(1).mean())
            n1 = float(((F @ Wn.T) * Gn).sum(1).mean())
            res["orth_err"][src]["cos_wrong_before_after"] = [n0, n1]
            # cross-covariance spectra: both are stable-rank ~1 (the modality-gap direction)
            for tag, Gt in [("M_pos", G), ("M_neg", Gn)]:
                s = np.linalg.svd(F.T @ Gt, compute_uv=False)
                res["orth_err"][src][f"{tag}_spectrum"] = {
                    "s1": float(s[0]), "s2": float(s[1]),
                    "stable_rank": float((s**2).sum() / s[0]**2)}

        for src in DATASETS:
            Wp, Wn = Ws[src]["Wpos"], Ws[src]["Wneg"]
            mats = {
                "Wpos": Wp,
                "Wneg": Wn,
                "Wpos@Wneg": Wp @ Wn,
                "Wneg@Wpos": Wn @ Wp,
                "Wpos@Wneg^T": Wp @ Wn.T,
            }
            res["fit"][src] = {}
            for evd in DATASETS:
                F, clf, lab = data[evd]
                cell = {name: zshot_acc(F, clf, lab, W) for name, W in mats.items()}
                res["fit"][src][evd] = cell

        # held-out split: fit W on train half (uses labels), eval on unseen test half
        res["split"] = {}
        for ds, (F, clf, lab) in data.items():
            n = len(lab); perm = np.random.default_rng(SEED).permutation(n)
            tr, te = perm[:n // 2], perm[n // 2:]
            Wp, Wn = fit_pair_matrices(F[tr], clf, lab[tr], np.random.default_rng(SEED))
            # translation baseline: shift images toward the text centroid (gap = translation?)
            mu = (clf.mean(1) - F[tr].mean(0))           # text-centroid minus image-centroid (train)
            def center_acc(idx):
                Xc = F[idx] + mu
                Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
                return float(((Xc @ clf).argmax(1) == lab[idx]).mean())
            res["split"][ds] = {
                "base_test": zshot_acc(F[te], clf, lab[te]),
                "Wpos_test": zshot_acc(F[te], clf, lab[te], Wp),
                "Wneg_test": zshot_acc(F[te], clf, lab[te], Wn),
                "shift_test": center_acc(te),
                "base_train": zshot_acc(F[tr], clf, lab[tr]),
                "Wpos_train": zshot_acc(F[tr], clf, lab[tr], Wp),
                "shift_train": center_acc(tr),
            }

        # image==text equivalence check (rotate classifier by W^T vs rotate image by W)
        F, clf, lab = data["imagenet"]
        Wp = Ws["imagenet"]["Wpos"]
        acc_img = zshot_acc(F, clf, lab, Wp)               # (W f)·e
        acc_txt = float(((F @ (Wp.T @ clf)).argmax(1) == lab).mean())  # f·(W^T e)
        res["img_eq_txt"] = {"rotate_image": acc_img, "rotate_text": acc_txt,
                             "abs_diff": abs(acc_img - acc_txt)}
        out[model] = res
        del data

    with open(os.path.join(os.path.dirname(__file__), "procrustes_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print_report(out)

def print_report(out):
    for model, res in out.items():
        print("=" * 78)
        print(f"MODEL {model}")
        print("  baseline zero-shot:", {k: round(v, 4) for k, v in res["baseline"].items()})
        for src, oe in res["orth_err"].items():
            cb, ca = oe["cos_true_before_after"]; nb, na = oe["cos_wrong_before_after"]
            print(f"  [{src}] orth-err Wpos={oe['Wpos']:.2e} Wneg={oe['Wneg']:.2e} | "
                  f"cos(true): {cb:.4f}->{ca:.4f} (Wpos attract) | "
                  f"cos(wrong): {nb:.4f}->{na:.4f} (Wneg repel)")
        print(f"  image==text check (imagenet Wpos): rotate_image={res['img_eq_txt']['rotate_image']:.4f} "
              f"rotate_text={res['img_eq_txt']['rotate_text']:.4f} |Δ|={res['img_eq_txt']['abs_diff']:.2e}")
        print("  held-out 50/50 split (fit on train half, eval on unseen test half):")
        for ds, s in res["split"].items():
            print(f"    {ds:9s} TEST  base={s['base_test']:.4f} Wpos={s['Wpos_test']:.4f}"
                  f"({s['Wpos_test']-s['base_test']:+.3f}) shift={s['shift_test']:.4f}"
                  f"({s['shift_test']-s['base_test']:+.3f}) Wneg={s['Wneg_test']:.4f} | "
                  f"TRAIN base={s['base_train']:.4f} Wpos={s['Wpos_train']:.4f}({s['Wpos_train']-s['base_train']:+.3f})"
                  f" shift={s['shift_train']:.4f}({s['shift_train']-s['base_train']:+.3f})")
        names = ["Wpos", "Wneg", "Wpos@Wneg", "Wneg@Wpos", "Wpos@Wneg^T"]
        for src in res["fit"]:
            print(f"  --- W fit on {src} ---")
            hdr = "    {:14s}".format("eval\\matrix") + "".join(f"{n:>13s}" for n in names)
            print(hdr)
            for evd in res["fit"][src]:
                base = res["baseline"][evd]
                row = "    {:14s}".format(evd)
                for n in names:
                    v = res["fit"][src][evd][n]
                    row += f"  {v:.4f}({v-base:+.3f})"
                print(row + f"   [base {base:.4f}]")

if __name__ == "__main__":
    main()
