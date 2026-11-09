"""Fuse the greedy DSv4.1 Markov loop on gfx938 (full-vocab or DP/TP-sharded)."""

from __future__ import annotations

import functools
import logging
from types import ModuleType

import torch
import torch.nn.functional as F

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.op_opt.dspark.greedy_markov"
_MARKER = "_vllm_hcu_greedy_markov_applied"
logger = logging.getLogger(__name__)


@functools.cache
def _log_selected(mode: str) -> None:
    logger.info("DSv4.1 DSpark: using fused %s greedy Markov sampling on gfx938", mode)


def _parallel_group():
    """Only tensor-parallel peers share the same request / vocab-parallel head.

    Data-parallel ranks own different requests, so virtual vocab sharding over
    the DP group would mix unrelated logits. DP LM-head vocab parallelism is
    not wired yet; until then TP>1 is required for the sharded path.
    """
    from vllm.distributed import get_tp_group

    return get_tp_group()


def _eligible(speculator, num_reqs, head_hidden) -> bool:
    model = speculator.model
    if (
        type(model).__module__ != "vllm.models.deepseek_v41.amd.dspark"
        or type(model).__name__ != "DSparkDeepseekV4ForCausalLM"
        or speculator.draft_logits is not None
        or speculator._draft_topk is not None
        or speculator.use_confidence_head
        or speculator.acceptance_estimator is not None
        or speculator.draft_watermarker is not None
        or model.draft_id_to_target_id is not None
        or not 1 <= num_reqs <= 16
        or not 1 <= speculator.num_speculative_steps <= 8
        or head_hidden.device.type != "cuda"
        or head_hidden.dtype != torch.bfloat16
        or torch.version.hip is None
    ):
        return False
    processor = model.logits_processor
    if (
        processor.scale != 1.0
        or processor.soft_cap is not None
        or processor.logits_as_input
        or processor.head_dtype not in (None, torch.bfloat16)
    ):
        return False
    markov = model.model.markov_head
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    projection = markov.markov_w2
    if (
        not isinstance(projection.quant_method, UnquantizedEmbeddingMethod)
        or not projection.disable_tp
        or getattr(projection, "bias", None) is not None
    ):
        return False
    weights = (markov.markov_w1.weight, projection.weight)
    if any(
        weight.ndim != 2
        or weight.shape[1] != 256
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.device != head_hidden.device
        for weight in weights
    ):
        return False
    if (
        weights[0].shape[0] != model.config.vocab_size
        or weights[1].shape[0] < model.config.vocab_size
    ):
        return False
    return (
        torch.cuda.get_device_properties(head_hidden.device).gcnArchName.split(":")[0]
        == "gfx938"
    )


def _try_sharded(speculator, num_reqs, head_hidden, base_logits) -> bool:
    """SGLang-style sharded greedy over DP/TP vocabulary partitions."""
    group = _parallel_group()
    world = int(getattr(group, "world_size", 1))
    if world <= 1:
        return False

    from vllm_hcu.model_executor.layers.dsv41_sharded_greedy import sharded_greedy_step

    steps = speculator.num_speculative_steps
    vocab = int(speculator.model.config.vocab_size)
    rank = int(group.rank_in_group)
    # Even padding so every rank owns the same shard width (final rank may be empty).
    per = (vocab + world - 1) // world
    start = rank * per
    end = min(start + per, vocab)
    width = max(0, end - start)

    markov = speculator.model.model.markov_head
    w1 = markov.markov_w1.weight
    w2 = markov.markov_w2.weight
    draft = speculator.draft_tokens
    anchors = speculator._anchor_idx[:num_reqs]
    prev = speculator.input_buffers.input_ids[anchors].to(torch.int64)
    base = base_logits.view(num_reqs, steps, vocab)

    for step in range(steps):
        embed = F.embedding(prev, w1)
        # Real shard width may be 0 on the final padded rank; transport width is
        # always ``per`` so every rank publishes the same all_gather shape.
        if width > 0:
            bias = F.linear(embed, w2[start:end]).contiguous()
            base_slice = base[:, step, start:end]
        else:
            bias = embed.new_empty((num_reqs, 0), dtype=torch.bfloat16)
            base_slice = base.new_empty((num_reqs, 0))
        if base_slice.shape[1] < per:
            base_local = F.pad(base_slice, (0, per - base_slice.shape[1]))
        else:
            base_local = base_slice
        prev = sharded_greedy_step(
            bias,
            base_local.contiguous(),
            group=group,
            vocab_start=start,
            width=width,
        )
        draft[:num_reqs, step] = prev.to(draft.dtype)

    _log_selected("sharded")
    return True


def _try_sample(speculator, num_reqs, head_hidden) -> bool:
    if not _eligible(speculator, num_reqs, head_hidden):
        return False

    steps = speculator.num_speculative_steps
    sample_hidden = head_hidden[speculator.sample_indices[: num_reqs * steps]]
    base_logits = speculator.model.compute_draft_logits(sample_hidden)
    vocab = int(speculator.model.config.vocab_size)
    if base_logits.dtype != torch.bfloat16 or base_logits.shape != (
        num_reqs * steps,
        vocab,
    ):
        return False

    if _try_sharded(speculator, num_reqs, head_hidden, base_logits):
        return True

    from vllm_hcu.model_executor.layers.dsv41_greedy_markov import greedy_markov

    markov = speculator.model.model.markov_head
    greedy_markov(
        base_logits.view(num_reqs, steps, -1),
        markov.markov_w1.weight,
        markov.markov_w2.weight,
        speculator.input_buffers.input_ids,
        speculator._anchor_idx,
        speculator.draft_tokens,
    )
    _log_selected("full-vocab")
    return True


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator")
    original = require_callable(
        cls, "_sample_sequential", f"{TARGET_MODULE}._sample_sequential"
    )
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("stale DSpark greedy Markov patch")
        return False
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.DSparkSpeculator._sample_sequential",
        positional=("self", "num_reqs", "head_hidden"),
    )

    @functools.wraps(original)
    def sample_sequential(self, num_reqs, head_hidden):
        if not _try_sample(self, num_reqs, head_hidden):
            return original(self, num_reqs, head_hidden)

    setattr(sample_sequential, _MARKER, True)
    cls._sample_sequential = sample_sequential
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
