# SPDX-License-Identifier: Apache-2.0
"""Public graph-integration contract for the owned V4.1 FP8 indexer reference.

The contract itself moved to `hcu_adapter.v41_fp8_indexer`: which capture modes
can break out a host-metadata op is a vLLM `CompilationConfig` rule, not plugin
mechanism. This module stays the plugin's stable import path and the single
patch point: it binds the adapter's `graph_allowed` resolver to a hook that
reads this module's attribute, so a patch here reaches the adapter, including
its FP4 admission check, without the adapter importing the plugin.
"""
from hcu_adapter import v41_fp8_indexer as _impl
from hcu_adapter.v41_fp8_indexer import (
    OP_NAME,
    graph_capture_allowed,
    register_splitting_op,
)

__all__ = ['OP_NAME', 'graph_capture_allowed', 'register_splitting_op']

# The plugin activates as a single owner; binding under the same name from this
# module is idempotent. Binding here (not only from `v41_indexer`) keeps the
# patch point live for processes that import just this module -- the CPU FP4
# admission tests patch `graph_capture_allowed` below.
OWNER = 'hcu_vllm_das.v41_indexer'


def _graph_capture_patch_point(config):
    return graph_capture_allowed(config)


def fp4_decode_graph_capture_allowed(config):
    """Admit only the native DP8/EP8 DSpark FP4 decode graph contract.

    The LightOp FP4 reader uses fixed-shape device metadata during decode.
    Prefill still needs a breakable PIECEWISE graph; the separate host-metadata
    FP8 reference retains its full-capture rejection.
    """
    mode = getattr(getattr(config, 'compilation_config', None),
                   'cudagraph_mode', None)
    if getattr(mode, 'name', None) != 'FULL_AND_PIECEWISE':
        return False
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    from hcu_adapter.fp4_indexer import applicable
    from .dspark_graph import native_decode_dspark, token_capacity, validate_graph_capacity

    if (not is_breakable_cudagraph_enabled() or not applicable(config)
            or not native_decode_dspark(config)):
        return False
    token_capacity(config)
    validate_graph_capacity(config)
    return True


_impl.bind_runtime_resolvers(
    OWNER, graph_allowed=lambda: _graph_capture_patch_point,
    fp4_graph_allowed=lambda: fp4_decode_graph_capture_allowed)
