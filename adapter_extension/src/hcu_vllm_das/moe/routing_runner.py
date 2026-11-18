# SPDX-License-Identifier: Apache-2.0
"""MoE fusions on vLLM 0.33's supported out-of-tree runner extension point."""
from __future__ import annotations

import os

from vllm.model_executor.custom_op import op_registry_oot
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

logger = init_logger(__name__)


def _enabled(name):
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


class ChannelFP8MoERunner(MoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hcu_gate_fusion = _enabled("VLLM_HCU_USE_BF16_MEGA_GATE")
        self._hcu_dbo = kwargs.get("enable_dbo", False)
        logger.info_once("Channel-FP8 MoERunner: method=%s, shared=%s, gate=%s",
                         type(self.routed_experts.quant_method).__name__,
                         type(kwargs.get("shared_experts")).__name__,
                         type(self.gate).__name__)
        if _enabled("VLLM_HCU_USE_FUSED_SILU_MUL_QUANT"):
            from .shared_act_quant import install_shared_activation

            install_shared_activation(kwargs.get("shared_experts"))

    def _can_fuse_gate(self, hidden_states, input_ids):
        from ..quantization.channel_fp8_moe import HcuDSV41ChannelFP8MoEMethod
        from vllm_hcu.model_executor.layers.fused_moe.bf16_mega_gate import (
            can_use_bf16_mega_gate,
        )

        method = self.routed_experts.quant_method
        return (
            self._hcu_gate_fusion and not self._hcu_dbo
            and isinstance(method, HcuDSV41ChannelFP8MoEMethod)
            and method._backend_mode == "ll"
            and self.gate is not None and not self._fse_fuse_gate
            and not self.do_naive_dispatch_combine
            and self.moe_config.pcp_size == self.moe_config.sp_size == 1
            and self.routed_input_transform is None
            and self.routed_output_transform is None
            and can_use_bf16_mega_gate(
                hidden_states, self.gate, self.router,
                hash_indices_table=getattr(self.router, "_hash_indices_table", None),
                input_ids=input_ids,
            )
        )

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input,
                      input_ids=None):
        if not self._can_fuse_gate(hidden_states, input_ids):
            return super()._forward_impl(hidden_states, router_logits,
                                         shared_experts_input, input_ids)
        from vllm_hcu.model_executor.layers.fused_moe.bf16_mega_gate import (
            try_run_bf16_mega_gate,
        )

        self.routed_experts._ensure_moe_quant_config_init()
        selected = try_run_bf16_mega_gate(
            hidden_states, self.gate, self.router,
            indices_dtype=self._quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
        if selected is None:
            return super()._forward_impl(hidden_states, router_logits,
                                         shared_experts_input, input_ids)
        logger.info_once("Using Channel-FP8 BF16 mega-gate on the native MoERunner.")
        weights, ids, _ = selected
        overlapping = (self._shared_experts is not None
                       and self._shared_experts.maybe_forward_async(shared_experts_input))
        with self._sequence_parallel_context():
            self._maybe_apply_shared_experts(shared_experts_input,
                                             SharedExpertsOrder.NO_OVERLAP)
            output = self.routed_experts.forward_modular(
                x=hidden_states, topk_weights=weights, topk_ids=ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )
            if overlapping:
                self._shared_experts.wait()
            shared = self._shared_experts.output if self._shared_experts is not None else None
            return self._maybe_combine(shared, output)


def install_channel_moe_runner():
    if not _enabled("VLLM_HCU_USE_CUSTOM_OPS"):
        return
    if not any(_enabled(name) for name in (
        "VLLM_HCU_USE_BF16_MEGA_GATE", "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT",
    )):
        return
    existing = op_registry_oot.get("MoERunner")
    if existing is ChannelFP8MoERunner:
        return
    if existing is not None:
        raise RuntimeError("MoERunner already has another out-of-tree owner")
    MoERunner.register_oot(ChannelFP8MoERunner)
