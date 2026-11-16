# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def patch(monkeypatch):
    directory = Path(__file__).parents[2] / "vllm_hcu/patch/worker/op_opt"
    package = ModuleType("_inverse_rope_rows_contract")
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        info_once=lambda *args: None
    )
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    for stem in ("_common", "patch_deepseek_v41_inverse_rope_rows"):
        name = package.__name__ + "." + stem
        spec = importlib.util.spec_from_file_location(name, directory / (stem + ".py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_register_inverse_rope_fake", lambda: False)
    return module


def original(o, positions, cos_sin_cache, rope_head_dim):
    return ("original", o.shape[0], rope_head_dim)


def target(patch, function=original):
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_inverse_rope_rows_ = function
    return module


def test_registration_and_fallback(monkeypatch, patch):
    module = target(patch)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: False)
    o = torch.ones(3, 2, 512, dtype=torch.bfloat16)
    result = module.rocm_inverse_rope_rows_(o, torch.arange(3), torch.empty(8, 64), 64)
    assert result == ("original", 3, 64)


@pytest.mark.parametrize(
    "flag", ["enabled", "master", "child", "grad", "cpu", "empty", "fp32", "rope_dim"]
)
def test_guards(monkeypatch, flag, patch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0" if flag == "master" else "1")
    monkeypatch.setenv(
        "VLLM_HCU_INVERSE_ROPE_SUFFIX", "0" if flag == "child" else "1"
    )
    monkeypatch.setattr(torch.version, "hip", "test")
    rows = 0 if flag == "empty" else 8
    rope_dim = 32 if flag == "rope_dim" else 64
    o = SimpleNamespace(
        is_cuda=flag != "cpu",
        dtype=torch.float32 if flag == "fp32" else torch.bfloat16,
        ndim=3,
        shape=(rows, 2, 512),
        device=torch.device("cpu"),
        stride=lambda dim: 1 if dim == -1 else 512,
    )
    positions = torch.arange(max(rows, 1))[:rows]
    cache = torch.empty(1024, rope_dim)
    cache = SimpleNamespace(
        device=o.device,
        ndim=2,
        dtype=torch.float32,
        shape=(1024, rope_dim),
        stride=lambda dim: 1 if dim == -1 else rope_dim,
    )
    scope = torch.enable_grad() if flag == "grad" else torch.no_grad()
    with scope:
        assert patch._eligible(o, positions, cache, rope_dim) == (flag == "enabled")


def test_optimized_path_uses_hcu_op(monkeypatch, patch):
    module = target(patch)
    patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    calls = []

    def inverse_rope(rope, positions, cache):
        calls.append((rope.shape, positions.shape[0]))
        rope.mul_(-1)

    monkeypatch.setattr(
        torch.ops.hcu_ops, "deepseek_v4_inv_rope", inverse_rope, raising=False
    )
    o = torch.randn(4, 2, 512, dtype=torch.bfloat16)
    expected = o.clone()
    expected[..., -64:] *= -1
    module.rocm_inverse_rope_rows_(o, torch.arange(4), torch.empty(16, 64), 64)
    torch.testing.assert_close(o, expected)
    assert calls == [((4, 2, 64), 4)]
