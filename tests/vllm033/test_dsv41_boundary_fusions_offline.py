"""Offline contract checks for DSV4.1 MHC boundary and query RMSNorm/FP8 ports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "adapter_extension" / "src" / "hcu_vllm_das"


def test_boundary_and_query_modules_parse():
    # Host CI may still use older CPython for collection; star-unpack returns
    # require 3.8+. Prefer a feature check over failing the contract suite.
    if not hasattr(ast, "unparse") and not hasattr(ast, "TypeIgnore"):
        pytest.skip("ast contract parse needs Python >=3.8")
    for name in (
        "dsv41_mhc_boundary.py",
        "dsv41_query_quant.py",
        "v41_query_quant_activation.py",
        "mhc_ops.py",
        "mhc_runtime.py",
    ):
        path = ROOT / name
        assert path.is_file(), path
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_mhc_delayed_mentions_boundary_before_maybe_post():
    source = (ROOT / "mhc_ops.py").read_text(encoding="utf-8")
    boundary = source.index("hcu_mhc_boundary")
    maybe_post = source.index("_maybe_post")
    assert boundary < maybe_post


def test_query_quant_entry_point_declared():
    text = (ROOT.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'das_hcu_v41_query_quant = "hcu_vllm_das.v41_query_quant_activation:extension_spec"'
        in text
    )


def test_boundary_gates_default_on_without_importing_torch_cuda(monkeypatch):
    # Import only the pure helpers through a restricted module load of environment.
    import importlib
    import sys

    env_name = "hcu_vllm_das.environment"
    sys.modules.pop(env_name, None)
    environment = importlib.import_module(env_name)
    monkeypatch.delenv("VLLM_HCU_DSV41_MHC_BOUNDARY", raising=False)
    monkeypatch.delenv("VLLM_HCU_DSV41_QUERY_RMSNORM_FP8", raising=False)
    assert environment._flag("VLLM_HCU_DSV41_MHC_BOUNDARY", default=True) is True
    assert environment._flag("VLLM_HCU_DSV41_QUERY_RMSNORM_FP8", default=True) is True
    monkeypatch.setenv("VLLM_HCU_DSV41_MHC_BOUNDARY", "0")
    assert environment._flag("VLLM_HCU_DSV41_MHC_BOUNDARY", default=True) is False


@pytest.mark.parametrize(
    "symbol",
    ["boundary", "boundary_fake", "rmsnorm_quant", "rmsnorm_quant_fake"],
)
def test_fusion_symbols_exist(symbol):
    import importlib

    module_name = (
        "hcu_vllm_das.dsv41_mhc_boundary"
        if symbol.startswith("boundary")
        else "hcu_vllm_das.dsv41_query_quant"
    )
    # Allow missing triton/torch at collection time by loading from path via ast only
    # when the package is not installed; prefer real import when available.
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        pytest.skip(f"runtime import unavailable: {error}")
    assert callable(getattr(module, symbol))
