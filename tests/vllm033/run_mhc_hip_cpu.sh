#!/usr/bin/env bash
# CPU public CustomOp integration; no install/model/service/device work.
set -euo pipefail
mhc_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
mhc_core=$(realpath "${VLLM_TARGET_ROOT:-${mhc_repo}/../vllm}")
mhc_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${mhc_repo}/../adapter-das}")
mhc_output=${1:?Provide a dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
mkdir -p "${mhc_output}"
mhc_output=$(realpath "${mhc_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_ROOT="${mhc_core}"
export PYTHONPATH="${mhc_core}:${mhc_adapter}:${mhc_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${mhc_repo}/tests/vllm033/replay_mhc_hip.py" \
  "${mhc_output}/report.json" 2>&1 | tee "${mhc_output}/replay.log"
