"""CPU contracts for SGLang-align mid-dispatch shared experts (no CUDA)."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import shared_experts as runtime
from vllm_hcu.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm_hcu.platforms import envs as henvs


@pytest.fixture
def mid_dispatch_setup(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.append(("wait", self.name, other.name))

    main, auxiliary = Stream("main"), Stream("aux")
    monkeypatch.setattr(runtime, "current_stream", lambda: main)
    monkeypatch.setattr(runtime, "aux_stream", lambda: auxiliary)
    monkeypatch.setattr(
        runtime, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )
    monkeypatch.setattr(runtime.envs, "VLLM_DISABLE_SHARED_EXPERTS_STREAM", False)
    monkeypatch.setattr(runtime.envs, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 64)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP", True)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, stream: calls.append(("record", stream.name))
    )

    @contextmanager
    def stream_scope(stream):
        calls.append(("enter", stream.name))
        yield

    monkeypatch.setattr(torch.cuda, "stream", stream_scope)

    class Layer(torch.nn.Module):
        def forward(self, x):
            calls.append(("compute",))
            return x * 3 + 1

    def build(mk_overlap=True, safe=True, backend="deepep_low_latency"):
        config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                enable_eplb=False,
                all2all_backend=backend,
                use_fi_nvl_two_sided_kernels=False,
            )
        )
        return runtime.SharedExperts(
            Layer(), config, False, lambda: mk_overlap, lambda: safe
        )

    return build, calls


def test_mid_dispatch_beats_early_launch(mid_dispatch_setup):
    build, calls = mid_dispatch_setup
    shared = build()
    x = torch.ones(2, 4)
    assert (
        shared._determine_shared_experts_order(x)
        is runtime.SharedExpertsOrder.MID_DISPATCH_OVERLAPPED
    )
    shared.maybe_sync_shared_experts_stream(x)
    assert not calls  # no early aux launch when mid-dispatch wins
    shared(x, runtime.SharedExpertsOrder.MID_DISPATCH_OVERLAPPED)
    assert calls == [("compute",)]
    torch.testing.assert_close(shared.output, x * 3 + 1)


def test_join_aux_before_dispatch_recv(mid_dispatch_setup, monkeypatch):
    build, calls = mid_dispatch_setup
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH", False)
    shared = build(backend="deepep_auto", mk_overlap=False)
    x = torch.ones(2, 4)
    # Force MULTI_STREAM + early launch path.
    monkeypatch.setattr(henvs, "VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP", True)
    # With mid off and eplb overlap + early: need eplb+deepep_auto for can_overlap.
    shared = runtime.SharedExperts(
        shared._layer,
        SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                enable_eplb=True,
                all2all_backend="deepep_auto",
                use_fi_nvl_two_sided_kernels=False,
            )
        ),
        False,
        lambda: False,
        lambda: True,
    )
    shared.maybe_sync_shared_experts_stream(x)
    assert ("enter", "aux") in calls
    assert shared._output_pending_on_stream[0] is True
    calls.clear()
    shared.maybe_join_aux_before_dispatch_recv()
    assert calls == [("wait", "main", "aux")]
    assert shared._output_pending_on_stream[0] is False


def test_prepare_runs_shared_before_recv_hook(mid_dispatch_setup, monkeypatch):
    """Contract: prepare_async → mid shared → hook → receiver (SGL decode)."""
    build, _ = mid_dispatch_setup
    x = torch.ones(2, 4)
    order = []

    def hook():
        order.append("hook")

    def receiver():
        order.append("receiver")
        return (x, None, None, None, None)

    pf = MagicMock()
    pf.supports_async.return_value = True
    pf.begin_moe_call = None

    def prepare_async(*_a, **_k):
        order.append("prepare_async")
        return hook, receiver

    pf.prepare_async.side_effect = prepare_async

    fused = MagicMock()
    fused.activation_format.return_value = object()  # not BatchedExperts
    fused.expects_unquantized_inputs = False
    fused.quant_config = None
    fused.moe_config = SimpleNamespace(moe_parallel_config=None)

    impl = FusedMoEKernelModularImpl(pf, fused)
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.modular_kernel.dbo_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.modular_kernel.dbo_maybe_run_recv_hook",
        lambda: None,
    )

    shared = build()
    orig_forward = shared.forward

    def tracked_forward(*a, **k):
        order.append("shared")
        return orig_forward(*a, **k)

    shared.forward = tracked_forward
    impl._prepare(
        x,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.long),
        global_num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        shared_experts=shared,
        shared_experts_input=x,
    )
    assert order == ["prepare_async", "shared", "hook", "receiver"]
