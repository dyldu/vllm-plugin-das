# SPDX-License-Identifier: Apache-2.0
"""Mixed text/image routing and graph replay against the actual native router."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import bf16_mega_gate as mega


@pytest.mark.parametrize("padded", [False, True])
def test_vision_bias_replaces_text_bias(monkeypatch, padded):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    torch.manual_seed(17)
    m, h, e, sentinel = 48, 5120, 384, 128000
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(e, h, device="cuda").bfloat16() * 0.02)
    gate.e_score_correction_bias = torch.randn(e, device="cuda") * 0.03
    gate.tid2eid = None
    image_bias = torch.randn(e, device="cuda") * 3.0
    router = SimpleNamespace(scoring_func="sqrtsoftplus", top_k=6, renormalize=True,
                             routed_scaling_factor=1.5, use_grouped_topk=False,
                             bias_vl=image_bias, image_sentinel_lo=sentinel)
    ids = torch.tensor([0, sentinel - 1, sentinel, sentinel + 1, sentinel + 2,
                        sentinel + 3, sentinel + 4, sentinel + 5,
                        sentinel + 100, -1, 42, sentinel], device="cuda").repeat(4)
    padding = torch.arange(m, device="cuda") >= m - 3 if padded else None
    monkeypatch.setattr(mega, "_get_padding_mask", lambda _: padding)
    monkeypatch.setattr(fused_topk_bias_router, "_get_padding_mask", lambda _: padding)
    assert mega.can_use_bf16_mega_gate(x, gate, router, hash_indices_table=None, input_ids=ids)

    def fused():
        return mega.try_run_bf16_mega_gate(x, gate, router, indices_dtype=torch.int32,
                                         input_ids=ids)

    def compare(actual):
        assert actual is not None
        weights, experts, logits = actual
        stock_logits = torch.mm(x, gate.weight.T, out_dtype=torch.float32)
        expected_weights, expected_ids = fused_topk_bias(
            x, stock_logits, "sqrtsoftplus", gate.e_score_correction_bias, 6, True,
            indices_type=torch.int32, input_tokens=ids, routed_scaling_factor=1.5,
            bias_vl=image_bias, image_sentinel_lo=sentinel)
        torch.testing.assert_close(logits, stock_logits, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(experts, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)

    compare(fused())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused()
    # Change image membership after capture, including the upper exclusive bound.
    ids.add_(3)
    graph.replay()
    torch.cuda.synchronize()
    compare(actual)
