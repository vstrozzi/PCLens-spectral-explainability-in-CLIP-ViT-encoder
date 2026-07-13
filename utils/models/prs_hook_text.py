"""
Text-encoder counterpart of utils/models/prs_hook.py.

Decomposes the CLIP *text* encoder output at the EOS (end-of-text / "eot") token
into the sum of its multi-head-attention and MLP contributions, each projected
into the shared CLIP image-text space. Implements the decomposition

    M_text(t) = P~_txt Z0^eot
              + sum_l sum_h sum_{i=0..p_eot} c_{i,l,h}        (attention)
              + sum_l P~_txt [ MLP_l(LN_l(Z_l)) ]^eot         (MLP)

where P~_txt folds the final LayerNorm (ln_final) scaling and the text
projection (text_projection), and ^eot selects the per-sample EOS token row.

Differences vs. the vision PRSLogger:
  * gathers the EOS token at a *variable* per-sample position (text.argmax(-1))
    instead of the CLS token fixed at position 0;
  * the causal mask already contains the source positions 0..p_eot inside each
    head (attn_method="head_no_spatial"), so the stored per-head value is the
    summed contribution sum_i c_{i,l,h};

Adapted from https://github.com/yossigandelsman/clip_text_span. MIT License
Copyright (c) 2024 Yossi Gandelsman.
"""
import numpy as np
import torch


class PRSLoggerText(object):
    """
    Residual-stream decomposition of a CLIP-like text encoder at the EOS token.

    l = number of layers, n = sequence length, h = number of heads,
    d = attention/output dimension.

    Set `eos_idx` (a LongTensor of shape [b] with the per-sample EOS position)
    on the logger before each forward pass; the hooks gather that token.
    """

    def __init__(self, model, device, spatial: bool = False, text_projection: bool = True, full_output: bool = False):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.post_ln_std = None
        self.post_ln_mean = None
        self.model = model
        self.spatial = spatial  # keep the per-source-token axis (sum_i c_{i,l,h} un-collapsed)
        self.text_projection = text_projection  # apply the final text projection
        self.full_output = full_output  # keep all tokens instead of only the EOS token
        self.eos_idx = None  # [b] per-sample EOS position, set before forward

    def _gather_eos(self, ret):
        """Select the EOS token row along the sequence axis (dim=1)."""
        assert self.eos_idx is not None, "Set logger.eos_idx (per-sample EOS positions) before the forward pass."
        b = ret.shape[0]
        return ret[torch.arange(b, device=ret.device), self.eos_idx.to(ret.device)]

    @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial`"  # [b, n, h, d]
        orig_type = ret.dtype
        bias_term = self.model.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = self._gather_eos(ret).detach().cpu()  # [b, h, d] at EOS
        self.attentions.append(
            (return_value.to(dtype=torch.float32)
             + bias_term[np.newaxis, np.newaxis].cpu()
             / (return_value.shape[1])).to(dtype=orig_type)
        )  # [b, h, d]
        return ret

    @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        assert len(ret.shape) == 5, "Verify that you use method=`head` and not `head_no_spatial`"  # [b, n, m, h, d]
        assert self.spatial, "Verify that you use method=`head` and not `head_no_spatial`"
        orig_type = ret.dtype
        bias_term = self.model.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = self._gather_eos(ret).detach().cpu()  # [b, m, h, d] at the EOS query
        self.attentions.append(
            (return_value.to(dtype=torch.float32)
             + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
             / (return_value.shape[1] * return_value.shape[2])).to(dtype=orig_type)
        )  # [b, m, h, d]: contribution of each source token m through head h
        return ret

    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(self._gather_eos(ret).detach().cpu())  # [b, d] at EOS
        return ret

    @torch.no_grad()
    def log_initial(self, z0):
        """Record the projected initial residual Z0^eot as the first MLP slot."""
        self.mlps.append(self._gather_eos(z0).detach().cpu())  # [b, d]

    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = self._gather_eos(ret).detach().cpu()  # [b, 1]
        return ret

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = self._gather_eos(ret).detach().cpu()  # [b, 1]
        return ret

    def _normalize_mlps(self):
        orig_dtype = self.mlps.dtype
        # [b, l + 1, d]
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        mean_centered = (
            self.mlps.to(torch.float32)
            - self.post_ln_mean[:, :, np.newaxis].to(self.device, torch.float32) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.ln_final.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = (
            self.model.ln_final.bias.detach().to(self.device, torch.float32) / len_intermediates
        )
        post_ln = weighted_mean_by_std + bias_term
        if self.text_projection:
            return (post_ln @ self.model.text_projection.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    def _normalize_attentions_non_spatial(self):
        orig_dtype = self.attentions.dtype
        # [b, l, h, d]
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = self.attentions.shape[2]  # h
        mean_centered = self.attentions.to(torch.float32) - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis
        ].to(self.device, torch.float32) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.ln_final.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = self.model.ln_final.bias.detach().to(self.device, torch.float32) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        if self.text_projection:
            return (post_ln @ self.model.text_projection.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    def _normalize_attentions_spatial(self):
        orig_dtype = self.attentions.dtype
        # [b, l, m, h, d]
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = self.attentions.shape[2] * self.attentions.shape[3]  # m * h
        mean_centered = self.attentions.to(torch.float32) - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.ln_final.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = self.model.ln_final.bias.detach().to(self.device, torch.float32) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        if self.text_projection:
            return (post_ln @ self.model.text_projection.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    @torch.no_grad()
    def finalize(self, representation):
        """Apply the ln_final post-scaling, project, and normalize by the output norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(self.device)  # [b, l, (m,) h, d]
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        if self.full_output or not self.text_projection:
            return (self.attentions, self.mlps)
        norm = representation.norm(dim=-1).detach()
        projected_mlps = self._normalize_mlps()
        if self.spatial:
            projected_attentions = self._normalize_attentions_spatial()
            return (
                projected_attentions / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                projected_mlps / norm[:, np.newaxis, np.newaxis],
            )
        projected_attentions = self._normalize_attentions_non_spatial()
        return (
            projected_attentions / norm[:, np.newaxis, np.newaxis, np.newaxis],
            projected_mlps / norm[:, np.newaxis, np.newaxis],
        )

    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        self.eos_idx = None
        torch.cuda.empty_cache()


def hook_prs_logger_text(model, device, spatial: bool = False, text_projection: bool = True, full_output: bool = False):
    """
    Hook a text-encoder projected-residual-stream logger to the model.

    Registers on the text tower's own hook managers (the `textual` fork is unused
    by the CLIP constructor, so the transformer/ln_final carry standalone hook
    managers reachable through the module references).

    spatial=False -> encode with attn_method="head_no_spatial" (summed per-head
    EOS contribution, stored [b, l, h, d]); spatial=True -> encode with
    attn_method="head" (per-source-token contribution, stored [b, l, m, h, d]).
    """
    prs = PRSLoggerText(model, device, spatial=spatial, text_projection=text_projection, full_output=full_output)

    if spatial:
        model.transformer.hook.register(
            "resblocks.*.attn.out.post", prs.compute_attentions_spatial
        )
    else:
        model.transformer.hook.register(
            "resblocks.*.attn.out.post", prs.compute_attentions_non_spatial
        )
    model.transformer.hook.register(
        "resblocks.*.mlp.c_proj.post", prs.compute_mlps
    )
    model.ln_final.hook.register("mean", prs.log_post_ln_mean)
    model.ln_final.hook.register("sqrt_var", prs.log_post_ln_std)
    return prs
