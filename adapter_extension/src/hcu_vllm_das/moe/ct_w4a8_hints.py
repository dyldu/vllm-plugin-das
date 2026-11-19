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
