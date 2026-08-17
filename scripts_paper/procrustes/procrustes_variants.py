"""Two follow-up variants:
 (A) HARDEST-negative repulsion  -> non-degenerate W_neg (not the global centroid).
 (B) per-SAMPLE image fit  vs  per-CLASS text fit  -> the only setup where deriving W
     on the image vs text encoder output genuinely differs (different anchor sets).
"""
import os, json
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(k, "2")
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location(
    "pa", os.path.join(os.path.dirname(__file__), "procrustes_align.py"))
pa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pa)
OUT, MODELS, DATASETS = os.path.dirname(__file__), pa.MODELS, pa.DATASETS

def hard_neg_targets(F, clf, lab):
    s = F @ clf
    s[np.arange(len(lab)), lab] = -1e9
    return clf[:, s.argmax(1)].T

def class_means(F, lab, C):
    mu = np.zeros((C, F.shape[1]))
    for c in range(C):
        m = lab == c
        if m.any(): mu[c] = F[m].mean(0)
    return mu

def acc(F, clf, lab, W=None, side="image"):
    if W is None: X, Cm = F, clf
    elif side == "image": X, Cm = F @ W.T, clf
    else:                 X, Cm = F, W @ clf
    return float(((X @ Cm).argmax(1) == lab).mean())

def variant_hard():
    print("\n" + "#"*70 + "\n# (A) HARDEST-negative repulsion  W_neg=-VUT on top-distractor pairs\n" + "#"*70)
    res = {}
    for model in MODELS:
        data = {ds: pa.load(model, ds) for ds in DATASETS}
        res[model] = {}; Ws = {}
        for src in DATASETS:
            F, clf, lab = data[src]
            Wp = pa.procrustes(F, clf[:, lab].T)
            Gh = hard_neg_targets(F, clf, lab)
            Wn = pa.procrustes(F, Gh, repel=True)
            s = np.linalg.svd(F.T @ Gh, compute_uv=False)
            Ws[src] = (Wp, Wn)
            res[model].setdefault("stable_rank_hardneg", {})[src] = float((s**2).sum()/s[0]**2)
        for src in DATASETS:
            Wp, Wn = Ws[src]
            mats = {"Wneg_hard": Wn, "Wpos@Wneg_hard": Wp@Wn, "Wneg_hard@Wpos": Wn@Wp,
                    "Wpos@Wneg_hardT": Wp@Wn.T}
            res[model].setdefault("fit", {})[src] = {}
            for evd in DATASETS:
                F, clf, lab = data[evd]; b = pa.zshot_acc(F, clf, lab)
                res[model]["fit"][src][evd] = {"base": b,
                    **{n: pa.zshot_acc(F, clf, lab, W) for n, W in mats.items()}}
        res[model]["split"] = {}
        for ds, (F, clf, lab) in data.items():
            n = len(lab); perm = np.random.default_rng(pa.SEED).permutation(n)
            tr, te = perm[:n//2], perm[n//2:]
            Wn = pa.procrustes(F[tr], hard_neg_targets(F[tr], clf, lab[tr]), repel=True)
            res[model]["split"][ds] = {"base_test": pa.zshot_acc(F[te], clf, lab[te]),
                                       "Wneg_hard_test": pa.zshot_acc(F[te], clf, lab[te], Wn)}
    for model, r in res.items():
        print(f"\n{model}  stable_rank(M_neg_hard): { {k: round(v,2) for k,v in r['stable_rank_hardneg'].items()} }")
        for src in DATASETS:
            for evd in DATASETS:
                c = r["fit"][src][evd]; b = c["base"]
                print(f"  fit {src:8s} eval {evd:8s} base {b:.3f} | " +
                      " ".join(f"{k}={c[k]:.3f}({c[k]-b:+.3f})" for k in c if k != "base"))
        for ds, s in r["split"].items():
            print(f"  held-out {ds:8s}: base {s['base_test']:.3f}  Wneg_hard {s['Wneg_hard_test']:.3f}"
                  f"({s['Wneg_hard_test']-s['base_test']:+.3f})")
    return res

def variant_perclass():
    print("\n" + "#"*70 + "\n# (B) per-SAMPLE image fit  vs  per-CLASS text fit  (genuinely different)\n" + "#"*70)
    res = {}
    for model in MODELS:
        data = {ds: pa.load(model, ds) for ds in DATASETS}
        res[model] = {}; W = {}
        for src in DATASETS:
            F, clf, lab = data[src]; C = clf.shape[1]
            Wimg = pa.procrustes(F, clf[:, lab].T)
            mu = class_means(F, lab, C); keep = np.linalg.norm(mu, axis=1) > 1e-6
            Wtxt = pa.procrustes(clf.T[keep], mu[keep])
            W[src] = (Wimg, Wtxt)
        for src in DATASETS:
            Wimg, Wtxt = W[src]; res[model][src] = {}
            for evd in DATASETS:
                F, clf, lab = data[evd]; b = pa.zshot_acc(F, clf, lab)
                res[model][src][evd] = {"base": b,
                    "Wimg_perSample": acc(F, clf, lab, Wimg, "image"),
                    "Wtxt_perClass":  acc(F, clf, lab, Wtxt, "text")}
        res[model]["split"] = {}
        for ds, (F, clf, lab) in data.items():
            C = clf.shape[1]; n = len(lab); perm = np.random.default_rng(pa.SEED).permutation(n)
            tr, te = perm[:n//2], perm[n//2:]
            Wimg = pa.procrustes(F[tr], clf[:, lab[tr]].T)
            mu = class_means(F[tr], lab[tr], C); keep = np.linalg.norm(mu, axis=1) > 1e-6
            Wtxt = pa.procrustes(clf.T[keep], mu[keep])
            res[model]["split"][ds] = {"base": pa.zshot_acc(F[te], clf, lab[te]),
                "Wimg": acc(F[te], clf, lab[te], Wimg, "image"),
                "Wtxt": acc(F[te], clf, lab[te], Wtxt, "text")}
    for model, r in res.items():
        print(f"\n{model}")
        for src in DATASETS:
            for evd in DATASETS:
                c = r[src][evd]; b = c["base"]
                print(f"  fit {src:8s} eval {evd:8s} base {b:.3f} | "
                      f"Wimg(perSample,rot-img)={c['Wimg_perSample']:.3f}({c['Wimg_perSample']-b:+.3f})  "
                      f"Wtxt(perClass,rot-txt)={c['Wtxt_perClass']:.3f}({c['Wtxt_perClass']-b:+.3f})")
        for ds, s in r["split"].items():
            print(f"  held-out {ds:8s}: base {s['base']:.3f}  Wimg {s['Wimg']:.3f}({s['Wimg']-s['base']:+.3f})"
                  f"  Wtxt {s['Wtxt']:.3f}({s['Wtxt']-s['base']:+.3f})")
    return res

if __name__ == "__main__":
    out = {"hard": variant_hard(), "perclass": variant_perclass()}
    json.dump(out, open(f"{OUT}/procrustes_variants.json", "w"), indent=2)
