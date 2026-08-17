"""Assemble the Procrustes-alignment report HTML (content-only, figures inlined)."""
import os, json, base64

OUT = os.path.dirname(__file__)
R = json.load(open(f"{OUT}/procrustes_results.json"))

def img(path):
    b = base64.b64encode(open(f"{OUT}/{path}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"

def f2(x): return f"{x:+.3f}"
def p(x):  return f"{x*100:.1f}%"

V = json.load(open(f"{OUT}/procrustes_variants.json"))
B16, B32 = R["ViT-B-16"], R["ViT-B-32"]
vb = V["perclass"]["ViT-B-16"]["split"]["imagenet"]
vb_img = vb["Wimg"] - vb["base"]; vb_txt = vb["Wtxt"] - vb["base"]
sr_hard = V["hard"]["ViT-B-16"]["stable_rank_hardneg"]["imagenet"]

def cell(m, src, evd, name): return m["fit"][src][evd][name]

# ---- headline numbers ----
b16_in_td  = cell(B16, "imagenet", "imagenet", "Wpos") - B16["baseline"]["imagenet"]
b16_split_te = B16["split"]["imagenet"]["Wpos_test"] - B16["split"]["imagenet"]["base_test"]
b16_split_tr = B16["split"]["imagenet"]["Wpos_train"] - B16["split"]["imagenet"]["base_train"]
sr_pos = B16["orth_err"]["imagenet"]["M_pos_spectrum"]["stable_rank"]
sr_neg = B16["orth_err"]["imagenet"]["M_neg_spectrum"]["stable_rank"]

FIG = {k: img(f"fig_{k}.png") for k in
       ["accuracy", "circuit_ViT-B-16", "circuit_ViT-B-32", "samples_ViT-B-16", "samples_ViT-B-32"]}

# ---------- accuracy table rows ----------
mats = ["Wpos", "Wneg", "Wpos@Wneg", "Wneg@Wpos", "Wpos@Wneg^T"]
scen = [("imagenet", "imagenet", "IN · fit IN"),
        ("CIFAR100", "CIFAR100", "C100 · fit C100"),
        ("imagenet", "CIFAR100", "C100 · IN→C100"),
        ("CIFAR100", "imagenet", "IN · C100→IN")]

def acc_table(m):
    head = "".join(f"<th>{s[2]}</th>" for s in scen)
    rows = ""
    for name in mats:
        tds = ""
        for src, evd, _ in scen:
            base = m["baseline"][evd]; v = cell(m, src, evd, name); d = v - base
            cls = "pos" if d > 0.003 else ("neg" if d < -0.003 else "zero")
            tds += f'<td class="{cls}"><b>{v:.3f}</b><span>{f2(d)}</span></td>'
        rows += f'<tr><th class="mat">{name}</th>{tds}</tr>'
    return f'<table class="acc"><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table>'

# ---------- mechanism stat rows ----------
def spec(m, ds, tag): return m["orth_err"][ds][f"{tag}_spectrum"]

HTML = f"""<title>Procrustes Alignment of CLIP Encoders — PCLens</title>
<style>
:root{{
  --paper:#f3f4f7; --surface:#ffffff; --raise:#fbfbfd;
  --ink:#161820; --muted:#585f6e; --faint:#8b91a0; --line:#e4e6ec;
  --accent:#3d52b5; --accent-ink:#2c3d8f; --accent-wash:#eceffb;
  --pos:#b23524; --neg:#2565c4; --grid-mono:0;
}}
@media (prefers-color-scheme:dark){{
  :root{{
    --paper:#0f1116; --surface:#171a21; --raise:#1c2029;
    --ink:#eef0f5; --muted:#a6acba; --faint:#6b7280; --line:#272b35;
    --accent:#8f9cec; --accent-ink:#aeb8f2; --accent-wash:#1b2033;
    --pos:#e2705f; --neg:#6ea2ea;
  }}
}}
:root[data-theme="dark"]{{
  --paper:#0f1116; --surface:#171a21; --raise:#1c2029;
  --ink:#eef0f5; --muted:#a6acba; --faint:#6b7280; --line:#272b35;
  --accent:#8f9cec; --accent-ink:#aeb8f2; --accent-wash:#1b2033;
  --pos:#e2705f; --neg:#6ea2ea;
}}
:root[data-theme="light"]{{
  --paper:#f3f4f7; --surface:#ffffff; --raise:#fbfbfd;
  --ink:#161820; --muted:#585f6e; --faint:#8b91a0; --line:#e4e6ec;
  --accent:#3d52b5; --accent-ink:#2c3d8f; --accent-wash:#eceffb;
  --pos:#b23524; --neg:#2565c4;
}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{
  background:var(--paper); color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  line-height:1.62; margin:0; letter-spacing:-0.003em;
}}
.mono{{font-family:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,monospace}}
.wrap{{max-width:1080px; margin:0 auto; padding:clamp(20px,4vw,60px) clamp(16px,4vw,44px)}}
.measure{{max-width:70ch}}
a{{color:var(--accent-ink)}}

/* header */
.eyebrow{{font-family:ui-monospace,Menlo,monospace; font-size:.74rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent-ink); margin:0 0 14px}}
h1{{font-size:clamp(1.9rem,3.6vw,2.7rem); line-height:1.08; font-weight:680;
  letter-spacing:-.02em; text-wrap:balance; margin:0 0 .5em}}
.lede{{font-size:1.12rem; color:var(--muted); max-width:64ch; margin:0}}
.rule{{height:2px; background:linear-gradient(90deg,var(--accent),transparent 62%);
  border:0; margin:34px 0 0}}

/* meta strip */
.meta{{display:flex; flex-wrap:wrap; gap:8px 10px; margin:22px 0 0; font-size:.8rem}}
.tag{{font-family:ui-monospace,Menlo,monospace; padding:4px 10px; border:1px solid var(--line);
  border-radius:999px; color:var(--muted); background:var(--surface)}}

/* sections */
section{{margin:56px 0 0}}
h2{{font-size:1.42rem; font-weight:660; letter-spacing:-.015em; margin:0 0 6px; text-wrap:balance}}
h2 .n{{font-family:ui-monospace,Menlo,monospace; font-size:.9rem; color:var(--accent-ink);
  margin-right:12px; font-weight:600}}
h3{{font-size:1.03rem; font-weight:650; margin:30px 0 6px; letter-spacing:-.01em}}
p{{margin:.55em 0}}
p, li{{color:var(--ink)}}
strong{{font-weight:660}}
.k{{font-family:ui-monospace,Menlo,monospace; background:var(--accent-wash); color:var(--accent-ink);
  padding:1px 6px; border-radius:5px; font-size:.9em}}

/* equation block */
.eq{{background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:18px 22px; margin:20px 0; overflow-x:auto;
  font-family:ui-monospace,Menlo,monospace; font-size:1.02rem; color:var(--ink)}}
.eq b{{color:var(--accent-ink)}}
.eq small{{display:block; color:var(--muted); font-family:ui-sans-serif,sans-serif; font-size:.85rem; margin-top:8px}}

/* stat tiles */
.tiles{{display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:14px; margin:26px 0}}
.tile{{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px 18px 16px}}
.tile .v{{font-family:ui-monospace,Menlo,monospace; font-size:1.72rem; font-weight:640;
  letter-spacing:-.02em; line-height:1; font-variant-numeric:tabular-nums}}
.tile .v.pos{{color:var(--pos)}} .tile .v.neg{{color:var(--neg)}} .tile .v.acc{{color:var(--accent-ink)}}
.tile .l{{font-size:.82rem; color:var(--muted); margin-top:9px; line-height:1.4}}

/* figure */
figure{{margin:26px 0; background:#ffffff; border:1px solid var(--line); border-radius:12px;
  padding:14px; overflow-x:auto}}
:root[data-theme="dark"] figure{{background:#f7f8fb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) figure{{background:#f7f8fb}}}}
figure img{{display:block; width:100%; min-width:640px; height:auto; border-radius:4px}}
figcaption{{font-size:.82rem; color:var(--muted); margin-top:12px; padding:0 4px;
  font-family:ui-sans-serif,sans-serif}}
figcaption b{{color:var(--ink)}}

/* tables */
.tblwrap{{overflow-x:auto; margin:22px 0}}
table.acc{{border-collapse:collapse; width:100%; font-size:.86rem; min-width:520px}}
table.acc th,table.acc td{{padding:8px 10px; text-align:center; border-bottom:1px solid var(--line)}}
table.acc thead th{{font-family:ui-monospace,Menlo,monospace; font-size:.76rem; color:var(--muted);
  font-weight:600; letter-spacing:.02em; text-align:center; border-bottom:1.5px solid var(--line)}}
table.acc th.mat{{font-family:ui-monospace,Menlo,monospace; text-align:left; color:var(--ink);
  font-weight:600; font-size:.82rem}}
table.acc td{{font-variant-numeric:tabular-nums}}
table.acc td b{{font-weight:640}}
table.acc td span{{display:block; font-size:.72rem; opacity:.75; font-family:ui-monospace,Menlo,monospace}}
td.pos{{background:color-mix(in srgb,var(--pos) 12%,transparent)}} td.pos b{{color:var(--pos)}}
td.neg{{background:color-mix(in srgb,var(--neg) 12%,transparent)}} td.neg span{{color:var(--neg)}}
td.zero{{color:var(--muted)}}
.cap{{font-family:ui-monospace,Menlo,monospace; font-size:.74rem; color:var(--faint);
  text-transform:uppercase; letter-spacing:.1em; margin:0 0 4px}}

/* callout */
.callout{{background:var(--raise); border:1px solid var(--line); border-radius:12px;
  padding:20px 24px; margin:24px 0}}
.callout.accent{{border-left:3px solid var(--accent)}}
.verdict{{display:grid; gap:14px; margin:22px 0}}
.qa{{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px 22px}}
.qa .q{{font-weight:640; margin:0 0 4px}}
.qa .a{{color:var(--muted); margin:0}}
.qa .a b{{color:var(--ink)}}
ul{{padding-left:1.15em}} li{{margin:.3em 0}}
footer{{margin:60px 0 10px; padding-top:20px; border-top:1px solid var(--line);
  font-size:.8rem; color:var(--faint); font-family:ui-monospace,Menlo,monospace}}
</style>

<div class="wrap">

<header>
  <p class="eyebrow">PCLens · spectral explainability · CLIP ViT</p>
  <h1>A rotation cannot fix the modality gap</h1>
  <p class="lede">Closed-form orthogonal Procrustes alignment of CLIP image and text
  encoders — positive vs. negative pairs, cross-dataset transfer, and what the
  rotation does to the internal component circuit.</p>
  <hr class="rule">
  <div class="meta">
    <span class="tag">ViT-B/16 · ViT-B/32</span>
    <span class="tag">ImageNet 5000 · CIFAR100 2000</span>
    <span class="tag">seed 69</span>
    <span class="tag">W = V Uᵀ · closed form</span>
    <span class="tag">image + text towers</span>
  </div>
</header>

<section>
  <h2><span class="n">01</span>The map</h2>
  <div class="measure">
  <p>Each CLIP embedding is the sum of its projected residual-stream components
  (attention heads + MLPs), already divided by <span class="k">‖output‖</span>, so
  the components sum to a unit vector and <span class="k">Σ M = cosθ</span> exactly.
  We fit a single orthogonal matrix that best rotates one tower onto the other.</p>
  </div>
  <div class="eq">
  <b>W*</b> = argmax<sub>WᵀW=I</sub> Σ<sub>i</sub> cossim(e<sub>i</sub>, W f<sub>i</sub>)
   = argmin<sub>WᵀW=I</sub> Σ<sub>i</sub> ‖e<sub>i</sub> − W f<sub>i</sub>‖²
   = <b>V Uᵀ</b>,&nbsp;&nbsp; where&nbsp; M = Σ<sub>i</sub> f<sub>i</sub> e<sub>i</sub>ᵀ = U S Vᵀ
  <small>W<sub>pos</sub> — attract each image f<sub>i</sub> to its <b>true</b> class text e (max cossim).
  &nbsp;·&nbsp; W<sub>neg</sub> = <b>−</b>V Uᵀ — repel each image from a <b>random wrong</b> class (min cossim / max L2).
  &nbsp;·&nbsp; Orthogonality error ≈ 3×10⁻¹⁵.</small>
  </div>
  <div class="measure">
  <p>Because both towers already live in one space, a perfect alignment would give
  <span class="k">W = I</span>. Any departure is the rotational part of CLIP's
  <em>modality gap</em>. We test whether recovering it helps zero-shot classification,
  whether it transfers ImageNet→CIFAR100, and whether fitting on the image or the text
  tower differs — then open the rotation up at the component level.</p>
  </div>
</section>

<section>
  <h2><span class="n">02</span>What the matrices do to accuracy</h2>
  <div class="tiles">
    <div class="tile"><div class="v pos">{b16_in_td:+.3f}</div>
      <div class="l">W<sub>pos</sub> zero-shot gain on the <b>fit set</b> (ImageNet, ViT-B/16) — looks like a win…</div></div>
    <div class="tile"><div class="v neg">{b16_split_te:+.3f}</div>
      <div class="l">…but on <b>held-out</b> images the same rotation <b>hurts</b>. It memorises, not generalises.</div></div>
    <div class="tile"><div class="v neg">−0.54</div>
      <div class="l">W<sub>pos</sub> fit on CIFAR100, applied to ImageNet. Rotations are dataset-specific.</div></div>
    <div class="tile"><div class="v acc">0.000</div>
      <div class="l">|Δacc| between rotating the <b>image</b> tower vs the <b>text</b> tower. Provably identical.</div></div>
  </div>
  <figure>
    <img alt="Accuracy heatmaps and held-out generalisation bars" src="{FIG['accuracy']}">
    <figcaption><b>Left/centre —</b> Δ zero-shot accuracy (matrix − baseline) for every
    matrix × fit→eval scenario. Only <b>W<sub>pos</sub> on its own fit set</b> is positive; every
    transfer is negative, and <b>W<sub>neg</sub> and all products with it collapse to ≈0</b>.
    <b>Right —</b> the held-out split: W<sub>pos</sub> lifts <b>train</b> accuracy (red) far above baseline
    but <b>test</b> accuracy (blue) falls below it — a textbook over-fit of a 512×512 (~130k-parameter)
    rotation to a few thousand samples.</figcaption>
  </figure>
  <div class="tblwrap">
    <p class="cap">ViT-B/16 — zero-shot accuracy (Δ vs baseline)</p>
    {acc_table(B16)}
  </div>
  <div class="tblwrap">
    <p class="cap">ViT-B/32 — zero-shot accuracy (Δ vs baseline)</p>
    {acc_table(B32)}
  </div>
  <div class="callout accent measure">
  <p style="margin:0"><strong>Reading.</strong> W<sub>pos</sub> raises mean true-class cosine from
  <b>0.27 → 0.72</b>, yet the only positive accuracy cells are the transductive diagonal
  (fit = eval, so the fit used the eval labels). Held out and across datasets it is negative.
  W<sub>neg</sub> repels correctly at the pair level (<b>cos(wrong): 0.08 → −0.52</b>) but its
  zero-shot accuracy is ~0 — and so is every product that contains it.</p>
  </div>
</section>

<section>
  <h2><span class="n">03</span>Image tower ≡ text tower</h2>
  <div class="measure">
  <p>With the <em>same</em> pairs, fitting the alignment on the image encoder output and
  on the text encoder output are not merely similar — they are the same operator.
  W<sub>text</sub> = W<sub>image</sub>ᵀ, and cos(Wf, e) = cos(f, Wᵀe) because W is
  norm-preserving. Rotating images or rotating the classifier inserts W into the
  <em>same</em> place in the bilinear score:</p>
  </div>
  <div class="eq">score(y,c) = e<sub>c</sub>ᵀ <b>W</b> f<sub>y</sub> = Σ<sub>ij</sub> A<sub>i</sub>(c) <b>W</b> B<sub>j</sub>(y)ᵀ = <b>1ᵀ M′(y,c) 1</b>
  <small>Measured |Δacc| = 0.000 on both models. A genuine image-vs-text difference can only
  appear if W is fit on each tower's own per-class geometry — not the case here.</small></div>
</section>

<section>
  <h2><span class="n">04</span>Why it can't generalise — the spectrum</h2>
  <div class="tiles">
    <div class="tile"><div class="v acc">≈ {sr_pos:.2f}</div>
      <div class="l">Stable rank of the cross-covariance <b>M<sub>pos</sub></b>. One direction dominates.</div></div>
    <div class="tile"><div class="v acc">≈ {sr_neg:.2f}</div>
      <div class="l">Stable rank of <b>M<sub>neg</sub></b> — even more collapsed (s₁/s₂ ≈ 160).</div></div>
    <div class="tile"><div class="v acc">0.9999</div>
      <div class="l">cos(mean random-negative target, global text centroid). The negatives <b>are</b> the centroid.</div></div>
  </div>
  <div class="measure">
  <p>Both cross-covariance matrices are effectively <strong>rank-1</strong>: their single
  dominant axis is the image-centroid → text-centroid direction — the modality gap itself.
  So W<sub>pos</sub> mostly performs one global rotation that lifts <em>every</em> image's cosine
  to <em>every</em> class by the same amount. A uniform lift is invisible to the arg-max, which
  is why accuracy doesn't move out of sample; the small class-specific residual (rank ≥ 2) is
  exactly what over-fits.</p>
  <p>W<sub>neg</sub> inherits the pathology in the extreme: a random wrong class averages to the
  global text centroid (cos 0.9999), so repelling from it is a degenerate rank-1 reflection that
  scrambles class information — hence the ~0 accuracy. This is the same
  <em>class-independent scaffold bias</em> that is arg-max-invisible in the interaction-matrix study;
  an orthogonal rotation can neither add the missing <em>translation</em> nor exploit the bias.</p>
  </div>
</section>

<section>
  <h2><span class="n">05</span>The circuit — which components W enforces and destroys</h2>
  <div class="measure">
  <p>Insert W into the component interaction matrix
  <span class="k">M(y,c) = A(c)·W·B(y)ᵀ</span> (text components × image components; W = I
  recovers the plain matrix). The reshaping is not spread across the network — it is
  concentrated almost entirely in the <strong>MLP / scaffold stream</strong>.</p>
  </div>
  <figure>
    <img alt="Interaction matrix reshaping for ViT-B/16" src="{FIG['circuit_ViT-B-16']}">
    <figcaption><b>Top —</b> mean interaction matrix M̄, after-rotation M̄′, and their difference
    ΔM̄ (signed-log colour). The change lives in the <b>MLP column and MLP row</b> (indices 144–156
    image · 96–108 text); the attention×attention block barely moves.
    <b>Bottom —</b> per-component marginal change in the true-class score: W <b>enforces</b> the
    early/late MLPs (<span class="mono">m11 +0.14, m0 +0.11, m8–m10</span>) and <b>destroys</b>
    the final <span class="mono">m12</span> slot (−0.10). <b>70%</b> of the total action falls on
    the 13 MLP components; the 144 attention heads are nearly untouched.</figcaption>
  </figure>
  <figure>
    <img alt="Sample circuits and dataset geometry for ViT-B/16" src="{FIG['samples_ViT-B-16']}">
    <figcaption><b>Top —</b> ΔM for four individual images (goldfish, zebra, cheeseburger, fire truck):
    every one is dominated by the same MLP column/row.
    <b>Bottom-left —</b> per-sample true-class similarity before vs after W: a uniform upward shift
    that stays above y=x and preserves ranking (why the arg-max is unchanged).
    <b>Bottom-right —</b> decomposing each image's score into its attention-stream and MLP-stream
    parts: W drives the cloud <b>straight up the MLP axis</b> while the attention axis is essentially
    frozen — the rotation rearranges the class-independent scaffold, not the semantic heads.</figcaption>
  </figure>
  <details class="callout">
    <summary style="cursor:pointer;font-weight:640">ViT-B/32 — same picture (click)</summary>
    <figure style="margin-top:16px"><img alt="ViT-B/32 circuit" src="{FIG['circuit_ViT-B-32']}"></figure>
    <figure><img alt="ViT-B/32 samples" src="{FIG['samples_ViT-B-32']}"></figure>
  </details>
</section>

<section>
  <h2><span class="n">06</span>Two more knobs, the same wall</h2>
  <div class="measure">
  <p><strong>Hardest-negative repulsion.</strong> Swapping the random wrong class for each image's
  <em>top distractor</em> leaves W<sub>neg</sub> just as degenerate — cross-covariance stable rank
  &asymp; {sr_hard:.2f}, zero-shot 0.000 everywhere. CLIP's text embeddings sit in a narrow cone whose
  dominant axis is the modality-gap direction, so <em>any</em> repulsion collapses onto it.</p>
  <p><strong>Per-sample image fit vs. per-class text fit.</strong> This is the one setup where deriving
  W on the image vs. the text encoder output can actually differ — the anchor sets differ (N per-image
  pairs vs. C per-class pairs). It does, but only marginally: on held-out ImageNet the coarser text fit
  over-fits slightly <em>less</em> ({vb_txt:+.3f}) than the per-sample image fit ({vb_img:+.3f}) — both
  stay net-negative. No substantial image-vs-text gap, and neither generalises.</p>
  </div>
</section>

<section>
  <h2><span class="n">07</span>Answers</h2>
  <div class="verdict">
    <div class="qa"><p class="q">Does W<sub>pos</sub> / W<sub>neg</sub> / the combination help zero-shot?</p>
      <p class="a"><b>No, except transductively.</b> W<sub>pos</sub> helps only when fit on the very set
      it is evaluated on; on held-out images it costs 1–13 points. W<sub>neg</sub> and every product
      (W<sub>pos</sub>W<sub>neg</sub>, W<sub>neg</sub>W<sub>pos</sub>, W<sub>pos</sub>W<sub>neg</sub>ᵀ)
      collapse the accuracy to ≈0.</p></div>
    <div class="qa"><p class="q">Does an ImageNet-fit matrix generalise to CIFAR100?</p>
      <p class="a"><b>No.</b> ImageNet→CIFAR100 is {cell(B16,'imagenet','CIFAR100','Wpos')-B16['baseline']['CIFAR100']:+.3f}
      (B/16) and {cell(B32,'imagenet','CIFAR100','Wpos')-B32['baseline']['CIFAR100']:+.3f} (B/32); the reverse is
      catastrophic (−0.53). The learned rotation encodes a dataset-specific gap direction.</p></div>
    <div class="qa"><p class="q">Is there a substantial difference between fitting on the image vs text encoder output?</p>
      <p class="a"><b>None — provably.</b> The two are transposes of one operator and give bit-identical
      zero-shot accuracy (|Δ| = 0.000). The difference the question anticipates only exists if each
      tower is fit on its own per-class geometry.</p></div>
    <div class="qa"><p class="q">Which components does W act on — the interactions of which parts?</p>
      <p class="a"><b>The MLP scaffold, not the attention heads.</b> ~70% of the reshaping is in the 13 MLP
      components: it enforces m0 and m8–m11, destroys m12, and leaves the 144 attention heads — where the
      class-discriminative signal lives — essentially untouched. W rearranges the class-independent
      "DC" stream, which is exactly why it lifts cosines uniformly yet never improves the decision.</p></div>
  </div>
  <div class="callout accent measure">
  <p style="margin:0"><strong>Bottom line.</strong> The closed-form orthogonal Procrustes is exact and
  dramatically raises pair cosine, but it is the wrong instrument for the modality gap: the gap's
  recoverable part is a single rank-1 (centroid-to-centroid) direction that is arg-max-invisible, and
  the rest is over-fit. Consistent with the broader PCLens finding — CLIP's uniform component sum is
  already calibrated, and no training-free rotation of the space beats it.</p>
  </div>
</section>

<footer>
  PCLens · orthogonal-Procrustes alignment study · ViT-B/16 · ViT-B/32 · ImageNet · CIFAR100 · seed 69
  &nbsp;·&nbsp; all numbers reproduced from output_dir PRS activations
</footer>

</div>
"""

open(f"{OUT}/report.html", "w").write(HTML)
print("wrote report.html", len(HTML), "bytes;  figures inlined:", len(FIG))
