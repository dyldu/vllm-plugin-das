"""DSV4.1 delayed mHC boundary following SGLang's mhc_boundary_hip kernels.

The previous pre-mix collapses the BF16-rounded residual; the new pre-mix
belongs to the next sublayer. Projection partials use FP32 IEEE dot products.
The residual store preserves the installed AITER HIP Post BF16 truncation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mix_reduce_sinkhorn_row(
    row,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    tokens,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    """_hc_mix_reduce_sinkhorn_kernel for one row with a tree fixed by NUM_SLICES; scratch_ptr
    ([m, 32] fp32) round-trips the reduced mixes so the sinkhorn starts from a plain layout."""
    copy_indices = tl.arange(0, HC)
    row_indices = copy_indices[:, None]
    column_indices = copy_indices[None, :]

    s_idx = tl.arange(0, SLICES_PAD)
    s_ok = s_idx < NUM_SLICES
    off = (s_idx * tokens + row) * MIX
    a_pre = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + copy_indices[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_post = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + HC + copy_indices[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_comb = tl.sum(
        tl.load(
            part_mix_ptr
            + off[:, None, None]
            + 2 * HC
            + row_indices[None, :, :] * HC
            + column_indices[None, :, :],
            mask=s_ok[:, None, None],
            other=0.0,
        ),
        axis=0,
    )
    sq = tl.sum(tl.load(part_sq_ptr + s_idx * tokens + row, mask=s_ok, other=0.0), axis=0)
    scratch_row = scratch_ptr + row * 32
    tl.store(scratch_row + copy_indices, a_pre)
    tl.store(scratch_row + HC + copy_indices, a_post)
    tl.store(scratch_row + 2 * HC + row_indices * HC + column_indices, a_comb)
    tl.debug_barrier()
    a_pre = tl.load(scratch_row + copy_indices)
    a_post = tl.load(scratch_row + HC + copy_indices)
    a_comb = tl.load(scratch_row + 2 * HC + row_indices * HC + column_indices)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + copy_indices)) + PRE_EPS
    tl.store(pre_ptr + row * HC + copy_indices, pre)
    post = POST_MULT * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + copy_indices))
    tl.store(post_ptr + row * HC + copy_indices, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + row_indices * HC + column_indices)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + SINKHORN_EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + SINKHORN_EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + SINKHORN_EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + SINKHORN_EPS)
    tl.store(comb_ptr + row * HC * HC + row_indices * HC + column_indices, comb)

@triton.jit
def _hc_mix_reduce_sinkhorn_vec_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    tokens,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    """One program per row of _hc_mix_reduce_sinkhorn_row."""
    _hc_mix_reduce_sinkhorn_row(
        tl.program_id(0),
        part_mix_ptr,
        part_sq_ptr,
        scratch_ptr,
        scale_ptr,
        base_ptr,
        pre_ptr,
        post_ptr,
        comb_ptr,
        tokens,
        inv_k,
        rms_eps,
        MIX=MIX,
        HC=HC,
        NUM_SLICES=NUM_SLICES,
        SLICES_PAD=SLICES_PAD,
        ITERS=ITERS,
        PRE_EPS=PRE_EPS,
        SINKHORN_EPS=SINKHORN_EPS,
        POST_MULT=POST_MULT,
    )

@triton.jit
def _hc_boundary_partial_kernel(
    x_ptr,
    res_ptr,
    post_in_ptr,
    comb_in_ptr,
    pre_prev_ptr,
    w_ptr,
    res_out_ptr,
    y_ptr,
    part_mix_ptr,
    part_sq_ptr,
    tokens,
    hidden,
    x_stride_m,
    res_stride_m,
    w_stride_n,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
):
    """Grid (cdiv(tokens, BLOCK_M), hidden // BLOCK_K), one hidden slice of every copy per program. HAS_POST:
    res_out[k] = post[k]*x + sum_j comb[j,k]*res[j] in aiter::mhc_post's order, read back as bf16
    for the statistics; HAS_COMBINE: y = sum_k pre_prev[k] * copy_k in _hc_combine_kernel's order."""
    tl.static_assert(HC == 4, "the weight tiles are prefetched by name")
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = offs_m < tokens
    offs_h = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, MIX_PAD)
    mask_n = offs_n < MIX
    m2 = mask_m[:, None]
    w_off = offs_n[None, :] * w_stride_n + offs_h[:, None]

    w0 = tl.load(w_ptr + w_off + 0 * hidden, mask=mask_n[None, :], other=0.0)
    w1 = tl.load(w_ptr + w_off + 1 * hidden, mask=mask_n[None, :], other=0.0)
    w2 = tl.load(w_ptr + w_off + 2 * hidden, mask=mask_n[None, :], other=0.0)
    w3 = tl.load(w_ptr + w_off + 3 * hidden, mask=mask_n[None, :], other=0.0)
    if HAS_POST:
        xa = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_h[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        r_off = offs_m[:, None] * res_stride_m + offs_h[None, :]
        r0 = tl.load(res_ptr + r_off + 0 * hidden, mask=m2, other=0.0).to(tl.float32)
        r1 = tl.load(res_ptr + r_off + 1 * hidden, mask=m2, other=0.0).to(tl.float32)
        r2 = tl.load(res_ptr + r_off + 2 * hidden, mask=m2, other=0.0).to(tl.float32)
        r3 = tl.load(res_ptr + r_off + 3 * hidden, mask=m2, other=0.0).to(tl.float32)

        cj = tl.arange(0, HC)
        post_tile = tl.load(
            post_in_ptr + offs_m[:, None] * HC + cj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        cc = tl.arange(0, HC * HC)
        comb_tile = tl.load(
            comb_in_ptr + offs_m[:, None] * (HC * HC) + cc[None, :], mask=m2, other=0.0
        ).to(tl.float32)
    if HAS_COMBINE:
        pj = tl.arange(0, HC)
        pre_tile = tl.load(
            pre_prev_ptr + offs_m[:, None] * HC + pj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        collapsed = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for copy_index in tl.static_range(HC):
        if HAS_POST:
            pk = tl.sum(tl.where((cj == copy_index)[None, :], post_tile, 0.0), axis=1)
            c0 = tl.sum(tl.where((cc == 0 * HC + copy_index)[None, :], comb_tile, 0.0), axis=1)
            c1 = tl.sum(tl.where((cc == 1 * HC + copy_index)[None, :], comb_tile, 0.0), axis=1)
            c2 = tl.sum(tl.where((cc == 2 * HC + copy_index)[None, :], comb_tile, 0.0), axis=1)
            c3 = tl.sum(tl.where((cc == 3 * HC + copy_index)[None, :], comb_tile, 0.0), axis=1)
            mixed = xa * pk[:, None]
            mixed += r0 * c0[:, None]
            mixed += r1 * c1[:, None]
            mixed += r2 * c2[:, None]
            mixed += r3 * c3[:, None]
            rounded_bits = (mixed.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16)
            rounded = rounded_bits.to(res_out_ptr.dtype.element_ty, bitcast=True)
            tl.store(
                res_out_ptr + offs_m[:, None] * (HC * hidden) + copy_index * hidden + offs_h[None, :],
                rounded,
                mask=m2,
            )
            xt = rounded.to(tl.float32)
        else:
            xt = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + copy_index * hidden + offs_h[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if HAS_COMBINE:
            pp = tl.sum(tl.where((pj == copy_index)[None, :], pre_tile, 0.0), axis=1)
            collapsed += pp[:, None] * xt
        if copy_index == 0:
            w_tile = w0
        elif copy_index == 1:
            w_tile = w1
        elif copy_index == 2:
            w_tile = w2
        else:
            w_tile = w3
        acc += tl.dot(xt, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(xt * xt, axis=1)
    tl.store(
        part_mix_ptr + (pid_t * tokens + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=m2 & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_t * tokens + offs_m, sq, mask=mask_m)
    if HAS_COMBINE:
        tl.store(
            y_ptr + offs_m[:, None] * hidden + offs_h[None, :],
            collapsed.to(y_ptr.dtype.element_ty),
            mask=m2,
        )

def boundary(
    residual: torch.Tensor, sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
    pre_mix: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
    hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, copies, hidden = residual.shape
    slices = hidden // 64
    options = {"device": residual.device, "dtype": torch.float32}
    updated = torch.empty_like(residual)
    layer_input = torch.empty((tokens, hidden), device=residual.device,
                              dtype=residual.dtype)
    post = torch.empty((tokens, copies, 1), **options)
    comb = torch.empty((tokens, copies, copies), **options)
    next_pre = torch.empty((tokens, copies), **options)
    partial_mix = torch.empty((slices, tokens, 24), **options)
    partial_sq = torch.empty((slices, tokens), **options)
    scratch = torch.empty((tokens, 32), **options)
    if tokens:
        _hc_boundary_partial_kernel[(triton.cdiv(tokens, 16), slices)](
            sublayer_out, residual, post_layer_mix, comb_res_mix, pre_mix, fn,
            updated, layer_input, partial_mix, partial_sq, tokens, hidden,
            sublayer_out.stride(0), residual.stride(0), fn.stride(0),
            HC=copies, MIX=24, MIX_PAD=32, BLOCK_M=16, BLOCK_K=64,
            DOT_PRECISION="ieee", HAS_POST=True, HAS_COMBINE=True,
            num_warps=2, num_stages=1,
        )
        _hc_mix_reduce_sinkhorn_vec_kernel[(tokens,)](
            partial_mix, partial_sq, scratch, hc_scale, hc_base, next_pre,
            post, comb, tokens, 1.0 / (copies * hidden), rms_eps,
            MIX=24, HC=copies, NUM_SLICES=slices,
            SLICES_PAD=triton.next_power_of_2(slices), ITERS=sinkhorn_repeat,
            PRE_EPS=hc_pre_eps, SINKHORN_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value, num_warps=4,
        )
    return updated, post, comb, layer_input, next_pre


def boundary_fake(
    residual: torch.Tensor, sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor,
    pre_mix: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
    hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
    hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, copies, hidden = residual.shape
    options = {"device": residual.device, "dtype": torch.float32}
    return (
        torch.empty_like(residual), torch.empty((tokens, copies, 1), **options),
        torch.empty((tokens, copies, copies), **options),
        torch.empty((tokens, hidden), device=residual.device, dtype=residual.dtype),
        torch.empty((tokens, copies), **options),
    )
