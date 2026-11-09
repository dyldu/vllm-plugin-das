# SPDX-License-Identifier: Apache-2.0
"""Public OOT mHC subclasses. Native classes/flags are never rebound.

HIP routes to admitted AITER kernels or the explicit Torch reference, so a
TileLang import does not determine HCU dispatch. Native constructors are
inherited; delayed Pre still constructs its Post in the model config context.
"""
from __future__ import annotations

import torch
from vllm.model_executor.layers import mhc as native

from . import mhc_runtime as runtime


def _empty(residual):
    return residual.ndim >= 3 and residual.shape[-1] > 0 and residual.numel() == 0


def _post(x, residual, post, comb):
    if _empty(residual):
        return torch.empty_like(residual)
    if (runtime.policy.post_enabled and runtime.provider.mhc_post is not None
            and runtime.post_supported(x, residual, post, comb)):
        return torch.ops.vllm.hcu_mhc_hip_post(x, residual, post, comb)
    return native.mhc_kernels.mhc_post_torch(x, residual, post, comb)


class _CheckedConstruction:
    def __init__(self, *args, **kwargs):
        from .mhc_activation import check_health

        self._hcu_mhc_owned = runtime.v41_model()
        if self._hcu_mhc_owned:
            check_health()
        # Delegate the complete native constructor, including delayed Pre's
        # nested Post, without mirroring its argument table.
        super().__init__(*args, **kwargs)


class HcuMHCPreOp(_CheckedConstruction, native.MHCPreOp):
    def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
                    norm_weight=None, norm_eps=0.0):
        if not self._hcu_mhc_owned:
            return super().forward_hip(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
                norm_weight, norm_eps)
        if _empty(residual):
            return runtime.empty_pre(residual)[:3]
        if (runtime.policy.pre_enabled and runtime.provider.mhc_pre is not None
                and n_splits == 1
                and runtime.pre_supported(residual, fn, hc_scale, hc_base)):
            post, comb, layer_input = torch.ops.vllm.hcu_mhc_hip_pre(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        else:
            post, comb, layer_input = super().forward_native(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits)
        return post, comb, native._apply_mhc_norm(layer_input, norm_weight, norm_eps)


class HcuMHCPreDelayedOp(_CheckedConstruction, native.MHCPreDelayedOp):
    def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                    pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6,
                    sublayer_out=None, post_layer_mix=None, comb_res_mix=None):
        if not self._hcu_mhc_owned:
            return super().forward_hip(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix,
                x, norm_weight, norm_eps, sublayer_out, post_layer_mix, comb_res_mix)
        # Fused Post+collapse+project+Sinkhorn replaces separate Post then Pre.
        if (runtime.policy.pre_enabled
                and runtime.boundary_supported(
                    residual, fn, hc_scale, hc_base, pre_mix, x, sublayer_out,
                    post_layer_mix, comb_res_mix)):
            updated, post, comb, layer_input, next_pre = (
                torch.ops.vllm.hcu_mhc_boundary(
                    residual, sublayer_out, post_layer_mix, comb_res_mix, pre_mix,
                    fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat))
            return (updated, post, comb,
                    native._apply_mhc_norm(layer_input, norm_weight, norm_eps), next_pre)
        # Apply previous Post exactly once, even when Pre must use Torch.
        residual = self._maybe_post(residual, sublayer_out, post_layer_mix, comb_res_mix)
        if _empty(residual):
            return residual, *runtime.empty_pre(residual)
        if (runtime.policy.pre_enabled
                and runtime.provider.mhc_pre_gemm_sqrsum is not None
                and runtime.provider.mhc_pre_big_fuse is not None
                and runtime.pre_supported(residual, fn, hc_scale, hc_base, pre_mix, x)):
            post, comb, layer_input, next_pre = torch.ops.vllm.hcu_mhc_hip_delayed(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, x)
        else:
            post, comb, layer_input, next_pre = native.mhc_kernels.mhc_pre_delayed_torch(
                residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, x=x)
        return (residual, post, comb,
                native._apply_mhc_norm(layer_input, norm_weight, norm_eps), next_pre)


class HcuMHCPostOp(_CheckedConstruction, native.MHCPostOp):
    def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
        if not self._hcu_mhc_owned:
            return super().forward_hip(x, residual, post_layer_mix, comb_res_mix)
        return _post(x, residual, post_layer_mix, comb_res_mix)


class HcuMHCFusedPostPreOp(_CheckedConstruction, native.MHCFusedPostPreOp):
    def forward_hip(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                    hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1,
                    norm_weight=None, norm_eps=0.0):
        if not self._hcu_mhc_owned:
            return super().forward_hip(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps)
        # No fused vendor/TL kernel is admitted. Keep the native combined route
        # for Post5120, where the old GPU evidence is asymmetric.
        if _empty(residual):
            return torch.empty_like(residual), *runtime.empty_pre(residual)[:3]
        if residual.shape[-1] == 5120:
            result = super().forward_native(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, n_splits, tile_n)
            return (*result[:3], native._apply_mhc_norm(result[3], norm_weight, norm_eps))
        residual_cur = _post(x, residual, post_layer_mix, comb_res_mix)
        if (runtime.policy.pre_enabled and runtime.provider.mhc_pre is not None
                and n_splits == 1
                and runtime.pre_supported(residual_cur, fn, hc_scale, hc_base)):
            post, comb, layer_input = torch.ops.vllm.hcu_mhc_hip_pre(
                residual_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        else:
            post, comb, layer_input = native.mhc_kernels.mhc_pre_torch(
                residual_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        return (residual_cur, post, comb,
                native._apply_mhc_norm(layer_input, norm_weight, norm_eps))


PAIRS = (
    (native.MHCPreOp, HcuMHCPreOp),
    (native.MHCPreDelayedOp, HcuMHCPreDelayedOp),
    (native.MHCPostOp, HcuMHCPostOp),
    (native.MHCFusedPostPreOp, HcuMHCFusedPostPreOp),
)
