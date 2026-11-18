# SPDX-License-Identifier: Apache-2.0
"""Real serving-operator checks, including BF16 rounding and FP8 code/scales."""
import pytest
import torch

from hcu_vllm_das.moe.shared_act_quant import silu_clamp_bf16_quant


@pytest.mark.parametrize("m", [0, 1, 8, 48, 64, 256])
@pytest.mark.parametrize("magnitude", [0.0, 0.001, 0.25, 1.0, 8.0])
def test_matches_stock_activation_and_quant(m, magnitude):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm import _custom_ops as ops

    torch.cuda.set_device(0)
    torch.manual_seed(42)
    x = (torch.randn(m, 4608, device="cuda") * magnitude).bfloat16()
    if m and magnitude:
        x[:, :6] = torch.tensor([-20, -10, -0.01, 0, 10, 20], device="cuda")
        x[:, 2304:2310] = x[:, :6]
    actual, scale = silu_clamp_bf16_quant(x, 10.0)
    if not m:
        assert actual.shape == (0, 2304) and scale.shape == (0, 1)
        return
    y = torch.empty((m, 2304), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
    reference, ref_scale = ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
    torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)


def test_replay_reads_new_inputs():
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    torch.cuda.set_device(0)
    x = torch.randn(48, 4608, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_clamp_bf16_quant(x, 10.0)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q, s = silu_clamp_bf16_quant(x, 10.0)
    x.normal_()
    graph.replay()
    expected_q, expected_s = silu_clamp_bf16_quant(x, 10.0)
    torch.cuda.synchronize()
    torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)


def test_all_finite_bf16_gate_values_preserve_rounding():
    """Every BF16 gate value, crossed with signed zeros and clamp multipliers.

    A constant gate per row makes its rounding visible in the row scale, even
    when a quantized code would otherwise hide an activation discrepancy.
    """
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm import _custom_ops as ops

    torch.cuda.set_device(0)
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = torch.empty(values.numel(), 4608, device="cuda", dtype=torch.bfloat16)
    x[:, :2304] = values[:, None]
    multipliers = torch.tensor([-20, -10, -1, -0.0, 0.0, 1 / 512, 1, 10, 20],
                               device="cuda", dtype=torch.bfloat16)
    x[:, 2304:] = multipliers.repeat(256)
    actual, scale = silu_clamp_bf16_quant(x, 10.0)
    y = torch.empty((x.shape[0], 2304), device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
    expected, expected_scale = ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 48])
def test_fused_activation_is_consumed_by_real_linear(m, native_linear_config):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from types import SimpleNamespace
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
    from hcu_vllm_das.moe.shared_act_quant import QuantizedSharedActivation, enable_shared_prequant

    torch.manual_seed(23)
    k, n = 2304, 5120
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym, activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(n, k), input_dtype=torch.bfloat16, out_dtype=torch.bfloat16)
    kernel = TritonFP8ScaledMMLinearKernel(
        config, ("weight", "weight_scale", "input_scale", "input_scale_ub"))
    down = torch.nn.Module()
    down.weight = torch.nn.Parameter(torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).T,
                                     requires_grad=False)
    down.weight_scale = torch.full((n, 1), 0.02, device="cuda")
    down.scheme = SimpleNamespace(fp8_linear=kernel)
    x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    act = SiluAndMulWithClamp(10.0)
    expected = kernel.apply_weights(down, act(x))
    enable_shared_prequant(down)
    assert get_input_quant_key(down) == kFp8DynamicTokenSym
    produced = QuantizedSharedActivation(act, down)(x)
    assert isinstance(produced, QuantizedActivation)
    stock_q, stock_scale = kernel.quant_fp8(act(x))
    torch.testing.assert_close(produced.scale, stock_scale, rtol=0, atol=0)
    torch.testing.assert_close(produced.data.view(torch.uint8), stock_q.view(torch.uint8),
                               rtol=0, atol=0,
                               msg=lambda text: (f"quant={kernel.quant_fp8._forward_method.__name__}, "
                                                 f"aiter={kernel.quant_fp8.use_aiter}: " + text))
    actual = down.scheme.fp8_linear.apply_weights(down, produced)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def native_linear_config():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    with set_current_vllm_config(VllmConfig(
            compilation_config=CompilationConfig(mode=0, custom_ops=["all"]))):
        yield


def test_native_shared_mlp_advertises_quantized_consumer(monkeypatch,
                                                        native_linear_config):
    """Exercise the actual model constructor and quantization scheme together."""
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from types import SimpleNamespace
    from vllm.config import get_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    from hcu_adapter.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config
    from hcu_vllm_das.moe.shared_act_quant import (
        QuantizedSharedActivation, TokenFP8TritonLinearKernel,
        install_shared_activation,
    )

    # Only constructor metadata is needed; the GPU consumer test above checks
    # real weights and outputs. No process group or model-size allocation here.
    monkeypatch.setattr(parallel_state, "_TP",
                        SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(get_current_vllm_config(), "model_config",
                        SimpleNamespace(dtype=torch.bfloat16))
    config = HcuDSV41ChannelFP8Config.from_config(dict(
        quant_method="fp8", is_per_channel=True, activation_scheme="dynamic"))
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    with torch.device("meta"):
        shared = DeepseekV4MLP(
            5120, 2304, "silu", 10.0, config, reduce_results=False,
            is_sequence_parallel=True, prefix="model.layers.0.ffn.shared_experts")
    install_shared_activation(shared)
    assert isinstance(shared.act_fn, QuantizedSharedActivation)
    assert type(shared.down_proj.scheme.fp8_linear) is TokenFP8TritonLinearKernel
    assert get_input_quant_key(shared.down_proj) == kFp8DynamicTokenSym
    installed = shared.act_fn
    install_shared_activation(shared)
    assert shared.act_fn is installed
