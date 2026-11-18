# SPDX-License-Identifier: Apache-2.0
"""Explicit native LL draft experiment, without changing the AG/RS default."""

KEY = 'hcu_dsv41_dspark_draft_backend'


def native_draft_ll_requested(config):
    extra = getattr(config, 'additional_config', None)
    if extra is None:
        return False
    if not isinstance(extra, dict):
        raise ValueError('DSpark additional_config must be a dict')
    backend = extra.get(KEY, 'allgather_reducescatter')
    if backend == 'allgather_reducescatter':
        return False
    if backend != 'deepep_low_latency':
        raise ValueError('Unsupported HCU DSpark draft backend: ' + str(backend))

    from .dspark_agrs import _owned_quant_kind, draft_config, is_owned_draft
    from .dspark_graph import native_decode_dspark

    if (_owned_quant_kind(config) != 'channel_fp8'
            or not native_decode_dspark(config) or is_owned_draft(config)):
        raise ValueError('Native draft LL requires the owned Channel-FP8 DP8/EP8 target')
    # Reuse all existing topology/feature/token-budget admission checks. This
    # temporary shallow copy allocates no communication or model resources;
    # callers keep the original config and native manager for the LL path.
    draft_config(config)
    return True
