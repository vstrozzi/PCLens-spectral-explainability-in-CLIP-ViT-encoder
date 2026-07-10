"""
Exact per-input decomposition of a CLIP ModifiedResNet image embedding into
layer x head components, following the derivation in the companion note
("Exact Per-Input Decomposition of CLIP-ResNet into Layer x Head Components").

    M_image(I) = sum_{l=0..L} sum_{h=1..H} c_{l,h}(I)  +  sum_{h=1..H} c_{P,h}(I)

l = 0 is the stem write, l = 1..L are the Bottleneck blocks (RN50: L=16, RN101: L=32),
h ranges over the AttentionPool2d heads (RN50/RN101: H=32).

The decomposition is EXACT per image: it freezes each post-residual ReLU as a 0/1
diagonal gate D_l(I), which makes the residual stream linear on this input, and freezes
the pooling softmax weights a^h(I), which makes attention pooling linear in the tokens.
Both are self-checking numerically (see `verify_decomposition`).

This module does not modify the model forward; it re-runs the (frozen) ModifiedResNet
with contribution tracking. It is generic over any ModifiedResNet (RN50, RN101, ...).

Adapted in spirit from the ViT PRS logger (utils/models/prs_hook.py).
"""
from contextlib import contextmanager
from typing import List, Tuple, Dict

import torch
from torch.nn import functional as F


@contextmanager
def _full_fp32():
    """Force true fp32 conv/matmul (disable TF32) inside the decomposition.

    On Ampere/Hopper GPUs PyTorch uses TF32 (10-bit mantissa) for conv/matmul by
    default, which is ~1e-3 relative and breaks the exactness of the linear
    contribution split (conv(a)+conv(b) != conv(a+b)).  We disable it locally and
    restore the previous state afterwards.
    """
    cudnn_prev = torch.backends.cudnn.allow_tf32
    matmul_prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_prev
        torch.backends.cuda.matmul.allow_tf32 = matmul_prev


# ----------------------------------------------------------------------------- #
# Feature-map decomposition:  Z(I) = sum_l M_{l->L}(I) g_l(I)
# ----------------------------------------------------------------------------- #

def _downsample_linear_const(downsample, x):
    """Split a ModifiedResNet shortcut (AvgPool -> Conv(no bias) -> BN) into its
    linear map applied to x and its input-independent constant.

    downsample(x) = linear(x) + const_broadcast, where BN is written as an affine map
        bn(y) = scale * y + shift,  scale = gamma/sqrt(var+eps),  shift = beta - scale*mean.
    Returns (linear(x), const[C])  with const broadcastable over spatial dims.
    """
    avgpool, conv, bn = list(downsample.children())  # keys "-1","0","1"
    y = conv(avgpool(x))  # linear in x (conv has bias=False, avgpool is linear)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # [C]
    shift = bn.bias - scale * bn.running_mean               # [C]
    linear = scale[None, :, None, None] * y
    return linear, shift


def _downsample_linear_only(downsample, c):
    """Linear part of the shortcut applied to a single contribution `c` (no BN shift)."""
    avgpool, conv, bn = list(downsample.children())
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return scale[None, :, None, None] * conv(avgpool(c))


@torch.no_grad()
def _bottleneck_capture(block, x):
    """Recompute one Bottleneck, exposing its pieces.

    Returns:
        x_next : ReLU(branch + identity)                 (real activation out of the block)
        branch : F_l(x) = bn3(conv3(avgpool(...)))       (the conv-branch payload g_l, pre-add)
        gate   : D_l = 1[branch + identity > 0]          (frozen post-residual ReLU)
        shift  : per-channel constant from the shortcut BN (0 if identity shortcut)
    """
    out = block.act1(block.bn1(block.conv1(x)))
    out = block.act2(block.bn2(block.conv2(out)))
    out = block.avgpool(out)
    branch = block.bn3(block.conv3(out))
    if block.downsample is not None:
        identity_lin, shift = _downsample_linear_const(block.downsample, x)
        identity = identity_lin + shift[None, :, None, None]
    else:
        identity = x
        shift = None
    pre = branch + identity
    gate = (pre > 0).to(pre.dtype)
    x_next = gate * pre  # == block.act3(pre), avoids in-place ReLU
    return x_next, branch, gate, shift


@torch.no_grad()
def decompose_feature_map(visual, image, check: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Decompose the final conv feature map Z(I) into per-block contributions.

    Returns:
        contribs : list of L+1 tensors [B, C, H', W'], contribs[l] = M_{l->L}(I) g_l(I).
                   contribs[0] is the stem write; contribs[1..L] the Bottleneck writes.
        Z        : the real final feature map [B, C, H', W'] (= sum_l contribs[l]).
    """
    with _full_fp32():
        x = visual.stem(image)      # g_0(I): the stem's write (opaque, nonlinear payload)
        contribs = [x]              # each entry is the running-propagated M_{k->·} g_k

        for layer in (visual.layer1, visual.layer2, visual.layer3, visual.layer4):
            for block in layer:
                x_next, branch, gate, shift = _bottleneck_capture(block, x)
                ds = block.downsample
                # Propagate existing contributions through this block's shortcut + gate.
                if ds is not None:
                    contribs = [gate * _downsample_linear_only(ds, c) for c in contribs]
                    new_write = gate * (branch + shift[None, :, None, None])  # BN shift attributed here
                else:
                    contribs = [gate * c for c in contribs]  # identity shortcut
                    new_write = gate * branch
                contribs.append(new_write)
                x = x_next

    if check:
        recon = torch.stack(contribs, 0).sum(0)
        err = (recon - x).abs().max().item()
        assert err < 1e-3 * (x.abs().max().item() + 1e-6), \
            f"feature-map decomposition mismatch: max abs err {err}"
    return contribs, x


# ----------------------------------------------------------------------------- #
# Attention-pool decomposition (linear given frozen softmax weights a^h)
# ----------------------------------------------------------------------------- #

def _tokens_from_map(fmap):
    """[B, C, H', W'] -> [K+1, B, C] with the mean/class token prepended (no positional add)."""
    B, C = fmap.shape[0], fmap.shape[1]
    t = fmap.reshape(B, C, -1).permute(2, 0, 1)          # [K, B, C]
    t = torch.cat([t.mean(dim=0, keepdim=True), t], dim=0)  # [K+1, B, C]
    return t


@torch.no_grad()
def attnpool_frozen_weights(attnpool, Z):
    """Frozen class-token attention weights a^h(I) computed on the real feature map Z.

    Returns attn [B, H, K+1] (softmax over the K+1 keys for query position 0).
    """
    H = attnpool.num_heads
    x = _tokens_from_map(Z) + attnpool.positional_embedding[:, None, :].to(Z.dtype)  # [K+1,B,C]
    L1, B, C = x.shape
    dh = C // H
    q0 = F.linear(x[0:1], attnpool.q_proj.weight, attnpool.q_proj.bias)  # [1,B,C]
    k = F.linear(x, attnpool.k_proj.weight, attnpool.k_proj.bias)        # [K+1,B,C]
    q0 = q0.reshape(1, B, H, dh)
    k = k.reshape(L1, B, H, dh)
    scaling = dh ** -0.5
    logits = torch.einsum("qbhd,lbhd->bhql", q0 * scaling, k)[:, :, 0, :]  # [B,H,K+1]
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def attnpool_split(attnpool, contribs, Z, check: bool = False):
    """Split AttentionPool2d(Z) into per-block-per-head + positional components.

    Given frozen weights a^h(I), pooling is linear in the tokens.  Splitting the value
    projection v_i = W_v z_i + (W_v P_i + b_v) sends the token content z_i to c_{l,h}
    (via the feature-map decomposition of z_i) and the positional/bias part to c_{P,h}.

    Returns:
        c_lh  : [B, L+1, H, out_dim]  the per-block per-head components c_{l,h}(I).
        c_pos : [B, out_dim]          sum_h c_{P,h}(I) + out_proj bias (content-free term).
        attn  : [B, H, K+1]           frozen class-token attention weights a^h(I).
    """
    H = attnpool.num_heads
    C = attnpool.v_proj.in_features
    out_dim = attnpool.c_proj.out_features
    dh = C // H
    B = Z.shape[0]

    with _full_fp32():
        attn = attnpool_frozen_weights(attnpool, Z)  # [B,H,K+1]
        Wo = attnpool.c_proj.weight.reshape(out_dim, H, dh)  # per-head output slabs

        def head_project(tokens):
            """tokens [K+1,B,C] with NO positional/bias -> per-head component [B,H,out_dim]."""
            v = F.linear(tokens, attnpool.v_proj.weight)          # value, no bias  [K+1,B,C]
            v = v.reshape(v.shape[0], B, H, dh)                    # [K+1,B,H,dh]
            head_out = torch.einsum("bhi,ibhd->bhd", attn, v)     # weighted sum over keys
            return torch.einsum("bhd,ohd->bho", head_out, Wo)     # [B,H,out_dim]

        # Content components: one per block contribution.
        c_lh = torch.stack([head_project(_tokens_from_map(c)) for c in contribs], dim=1)  # [B,L+1,H,out]

        # Positional + value-bias part (added to every token), then out-proj bias (once).
        P = attnpool.positional_embedding.to(Z.dtype)             # [K+1,C]
        v_pos = F.linear(P, attnpool.v_proj.weight, attnpool.v_proj.bias)  # [K+1,C], includes b_v
        v_pos = v_pos.reshape(P.shape[0], H, dh)                  # [K+1,H,dh] (batch-independent)
        head_pos = torch.einsum("bhi,ihd->bhd", attn, v_pos)     # [B,H,dh]
        c_pos = torch.einsum("bhd,ohd->bo", head_pos, Wo)        # [B,out_dim] (summed over heads)
        c_pos = c_pos + attnpool.c_proj.bias[None, :]            # single out-proj bias

    if check:
        with _full_fp32():
            real = attnpool(Z)                               # [B,out_dim]
        recon = c_lh.sum(dim=(1, 2)) + c_pos
        err = (recon - real).abs().max().item()
        assert err < 1e-3 * (real.abs().max().item() + 1e-6), \
            f"attn-pool decomposition mismatch: max abs err {err}"
    return c_lh, c_pos, attn


# ----------------------------------------------------------------------------- #
# Public entry point
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def decompose_resnet_image(visual, image, check: bool = False) -> Dict[str, torch.Tensor]:
    """Full exact decomposition of a CLIP ModifiedResNet image embedding.

    Args:
        visual : a ModifiedResNet (model.visual of a CLIP ResNet).
        image  : [B, 3, H, W] preprocessed batch on the same device as `visual`.
        check  : if True, assert exact recovery of visual(image) at every stage.

    Returns dict with:
        c_lh  : [B, L+1, H, out_dim]  components c_{l,h}(I)  (l=0 stem .. L).
        c_pos : [B, out_dim]          content-free positional+bias term sum_h c_{P,h}+b_o.
        attn  : [B, H, K+1]           frozen class-token attention weights.
    Their identity:  c_lh.sum((1,2)) + c_pos == visual(image).
    """
    contribs, Z = decompose_feature_map(visual, image, check=check)
    c_lh, c_pos, attn = attnpool_split(visual.attnpool, contribs, Z, check=check)
    return {"c_lh": c_lh, "c_pos": c_pos, "attn": attn}


@torch.no_grad()
def verify_decomposition(visual, image, verbose: bool = True) -> float:
    """Return max abs error between (sum of components) and visual(image)."""
    out = decompose_resnet_image(visual, image, check=False)
    recon = out["c_lh"].sum(dim=(1, 2)) + out["c_pos"]
    with _full_fp32():
        real = visual(image)
    err = (recon - real).abs().max().item()
    rel = err / real.abs().max().item()
    if verbose:
        print(f"[resnet_prs] components: {tuple(out['c_lh'].shape)}  "
              f"max|recon-real|={err:.3e}  rel={rel:.3e}")
    return err
