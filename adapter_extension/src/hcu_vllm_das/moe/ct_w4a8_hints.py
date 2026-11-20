# SPDX-License-Identifier: Apache-2.0
"""Host-only scheduling hint; device expert counts remain the row bounds."""


def expected_m(local_tokens: int, dispatchers: int, topk: int,
               global_experts: int, capacity: int) -> int:
    if (local_tokens < 0 or dispatchers <= 0 or topk <= 0
            or global_experts <= 0 or capacity <= 0):
        raise ValueError("Invalid low-latency MoE scheduling dimensions")
    # Reuse SGLang's one-row margin and the native round-to-16 policy.
    average = (local_tokens * dispatchers * topk + global_experts) // global_experts
    rounded = max(16, ((average + 15) // 16) * 16)
    return min(capacity, rounded)


def masked_gemm(function, inputs, weights, output, counts, hint):
    """Honor the selected native kernel's weight descriptor without copying.

    Mode1070 consumes the packed storage through an [E,N,K/2] descriptor;
    other selected N32 modes accept the N32 view installed at model loading.
    The storage is identical. Choose by provider ABI rather than whitelisting
    the token counts that happened to select a compatible mode in one run.
    """
    from deepgemm.configs.w4a8_masked_hipc import select_w4a8_hipc_masked_config
    config = select_w4a8_hipc_masked_config(
        hint, output.shape[2], inputs[0].shape[2], inputs[0].shape[0])
    weight, scale = weights
    if config.get('MODE') == 1070:
        weight = weight.view(inputs[0].shape[0], output.shape[2], inputs[0].shape[2] // 2)
    return function(inputs, (weight, scale), output, counts, hint, config=config)
