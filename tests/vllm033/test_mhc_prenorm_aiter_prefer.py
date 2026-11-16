"""Contract: MHC fused post-pre prefers aiter prenorm over FP32 SB matmul."""
from pathlib import Path


def test_mhc_fused_post_pre_source_prefers_aiter_sqrsum():
    root = Path(__file__).resolve().parents[2]
    src = (root / "vllm_hcu/model_executor/layers/mhc.py").read_text()
    assert "from aiter.ops.mhc import mhc_pre_gemm_sqrsum" in src
    assert "tf32_hc_prenorm_gemm as tf32_hc_prenorm_gemm" in src
    assert "tf32_hc_pernorm_gemm" not in src  # typo must stay fixed
    # aiter branch must appear before the deepgemm/torch fallback call site
    # inside the fused post-pre large-token path.
    aiter_idx = src.find("from aiter.ops.mhc import mhc_pre_gemm_sqrsum")
    fallback_idx = src.find(
        "tf32_hc_prenorm_gemm(\n"
        "                residual_cur.view(num_tokens, hc_hidden_size),"
    )
    assert aiter_idx > 0
    assert fallback_idx > aiter_idx


def test_tf32_prenorm_patch_prefers_native_op():
    src = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/patch/worker/op_opt/patch_deep_gemm.py"
    ).read_text()
    assert "op.tf32_hc_prenorm_gemm" in src
    assert "Cijk_*SB_*" in src or "torch.matmul" in src
