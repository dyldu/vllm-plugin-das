#!/usr/bin/env bash
# Channel-FP8 HE32 revalidation in duyl-033-1112, based on the supplied acc-w4.sh.
# Set HCU_VLLM_DIR, HCU_ADAPTER_SRC, MODEL_PATH, RUN_ROOT to explicit worktrees
# and outputs. Real prefill and real DSpark acceptance; see PROTOCOL.md.
# Usage: bash accuracy_fp8.sh [--check]
set -uo pipefail

cudagraph_mode="${CUDAGRAPH_MODE:-PIECEWISE}"
case "${cudagraph_mode}" in
  PIECEWISE|FULL_AND_PIECEWISE) ;;
  *) echo "Unsupported CUDAGRAPH_MODE: ${cudagraph_mode}" >&2; exit 2 ;;
esac

check_only=0
for argument in "$@"; do
  case "${argument}" in
    --check) check_only=1 ;;
    *) echo "Usage: bash ${BASH_SOURCE[0]} [--check]" >&2; exit 2 ;;
  esac
done

###############################################################################
# 固定部分（不需要修改）
###############################################################################

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${server_pid:-}" ]] && kill -0 -- "-${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for ((i=0; i<60; i++)); do
      kill -0 -- "-${server_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  if [[ -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    printf '%s\n' "${rc}" > "${run_dir}/exit_code"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local host="$1" port="$2" server_pid="$3" service_log="$4"
  local max_attempts="${5:-3000}" interval_sec="${6:-2}"
  local timeout_sec=$((max_attempts * interval_sec))
  local models_response

  echo "Waiting for vLLM Decode server on ${host}:${port} ..."
  echo "Server log: ${service_log}"
  echo "Server wait timeout: ${timeout_sec}s (${max_attempts} attempts x ${interval_sec}s)"

  for _ in $(seq 1 "${max_attempts}"); do
    if grep -q 'EngineCore failed to start' "${service_log}" 2>/dev/null; then
      tail -n 100 "${service_log}" >&2
      return 1
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "vLLM server exited early. Last logs:" >&2
      tail -n 100 "${service_log}" >&2
      return 1
    fi
    if models_response="$(curl --noproxy '*' --max-time 5 -fsS "http://${host}:${port}/v1/models" 2>/dev/null)"; then
      if ! printf '%s' "${models_response}" | "${PYTHON_BIN}" -S -c '
import json
import sys

models = json.load(sys.stdin).get("data", [])
if not any(model.get("id") == sys.argv[1] and model.get("root") == sys.argv[2] for model in models):
    raise SystemExit(f"acc-w4.sh: 端口上的模型不匹配，期望 {sys.argv[1]} ({sys.argv[2]})，实际 {models}")
' "${served_model_name}" "${model_path}"; then
        return 1
      fi
      echo "vLLM server is ready."
      return 0
    fi
    sleep "${interval_sec}"
  done
  echo "Timed out waiting for vLLM server. Last logs:" >&2
  tail -n 100 "${service_log}" >&2
  return 1
}

save_config_yaml() {
  {
    echo "# vLLM Server Configuration (acc-w4 / aligned laibao/033/w4/acc_d)"
    echo "timestamp: \"$(date '+%Y-%m-%d %H:%M:%S')\""
    echo "run_dir: \"${run_dir}\""
    echo "mode: acc_channel_fp8_he32"
    echo "cudagraph_mode: \"${cudagraph_mode}\""
    echo ""
    echo "# 进程里实际生效的变量（外部漏进来的也在这里现形）"
    echo "effective_env:"
    env | grep -E '^(HCU_|VLLM_|ROCSHMEM_|DEEPEP_|NCCL_|GLOO_|HIP_|GPU_MAX)' | sort | sed 's/^/  /' || echo "  {}"
    echo ""
    echo "# 服务与评测的完整命令行见同目录 serve_cmd.txt / eval_cmd.txt"
  } > "${run_dir}/config.yaml"
  echo "Configuration saved to: ${run_dir}/config.yaml"
}

save_model_config_snapshot() {
  local src="${model_path%/}/config.json"
  local dst="${run_dir}/model_config.json"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
}

script_path="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${script_path}")"
writable_script_path="/duyl/run/dsv4.1/033/acc-w4.sh"
if [[ ! -w "${SCRIPT_DIR}" && "${writable_script_path}" -ef "${script_path}" ]]; then
  script_path="${writable_script_path}"
  SCRIPT_DIR="$(dirname "${script_path}")"
fi
cd "${SCRIPT_DIR}" || exit 1

run_root="${RUN_ROOT:?set output root}"
mkdir -p "${run_root}" || {
  echo "acc-w4.sh: 无法创建输出目录 ${run_root}" >&2
  exit 1
}
write_probe="$(mktemp "${run_root}/.write_check.XXXXXX")" || {
  echo "acc-w4.sh: 输出目录不可写 ${run_root}" >&2
  exit 1
}
rm -f "${write_probe}" || exit 1
echo "Output root: ${run_root}"

PYTHON_BIN="$(command -v python3)" || {
  echo "acc-w4.sh: 找不到 python3" >&2; exit 1; }

# 自动探测 RDMA/RoCE 业务网卡（避免误选管理网导致 DeepEP/NCCL 超时）。
# 输出：line1=网卡名(逗号)  line2=IB设备(逗号)  line3=业务网段首个IPv4  line4=master NIC
detect_rdma_nics() {
  PYTHONPATH= "${PYTHON_BIN}" -S -B - <<'PY'
import fcntl
import os
import socket
import struct

def addr(name):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        d = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
        return socket.inet_ntoa(d[20:24])
    except OSError:
        return None
    finally:
        s.close()

def operstate(name):
    try:
        return open(f"/sys/class/net/{name}/operstate").read().strip()
    except OSError:
        return "?"

vids = ("lo", "docker", "veth", "br-", "virbr", "cni", "flannel", "kube")
cand = []
for name in sorted(os.listdir("/sys/class/net")):
    if any(name == v or name.startswith(v) for v in vids):
        continue
    ibdir = f"/sys/class/net/{name}/device/infiniband"
    if not os.path.isdir(ibdir) or operstate(name) != "up":
        continue
    ip = addr(name)
    if ip:
        cand.append((name, ip, sorted(os.listdir(ibdir))))

mgt = os.environ.get("HCU_MGT_PREFIX", "12.12.")
biz = [c for c in cand if not c[1].startswith(mgt)]
pick = biz if biz else cand
print(",".join(c[0] for c in cand))
print(",".join(d for c in cand for d in c[2]))
print(pick[0][1] if pick else "")
print(pick[0][0] if pick else "")
PY
}

###############################################################################
# 源码路径（保留本脚本 overlay；起服/评测口径走 laibao/033/w4/acc_d）
###############################################################################
HCU_VLLM_DIR="${HCU_VLLM_DIR:?set worktree}"
HCU_VLLM_DAS_SRC="${HCU_VLLM_DIR}/adapter_extension/src"
HCU_ADAPTER_SRC="${HCU_ADAPTER_SRC:?set adapter worktree}"
unset PYTHONPATH
[[ -f "${HCU_VLLM_DIR}/vllm_hcu/__init__.py" ]] || {
  echo "acc-w4.sh: HCU_VLLM_DIR 无效: ${HCU_VLLM_DIR}" >&2; exit 1; }
[[ -f "${HCU_VLLM_DAS_SRC}/hcu_vllm_das/__init__.py" ]] || {
  echo "acc-w4.sh: HCU_VLLM_DAS_SRC 无效: ${HCU_VLLM_DAS_SRC}" >&2; exit 1; }
[[ -f "${HCU_ADAPTER_SRC}/hcu_adapter/__init__.py" ]] || {
  echo "acc-w4.sh: HCU_ADAPTER_SRC 无效: ${HCU_ADAPTER_SRC}" >&2; exit 1; }
export PYTHONPATH="${HCU_VLLM_DIR}:${HCU_VLLM_DAS_SRC}:${HCU_ADAPTER_SRC}"
echo "[dev-overlay] PYTHONPATH=${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=hcu_adapter
export HCU_ADAPTER_HOST=vllm

###############################################################################
# 本次运行配置 —— 全部固定值（对齐 laibao/033/w4/acc_d.sh）
###############################################################################

# ============ 环境变量 ============
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
ulimit -l unlimited
export VLLM_HCU_USE_CUSTOM_OPS="1"
export HCU_ADAPTER_EXTENSION_OVERLAY=auto
export HCU_VLLM_MHC_MEGA=1
export VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH="${VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH:-1}"
export VLLM_HCU_USE_FUSED_SILU_MUL_QUANT="${VLLM_HCU_USE_FUSED_SILU_MUL_QUANT:-1}"
export VLLM_HCU_USE_BF16_MEGA_GATE="${VLLM_HCU_USE_BF16_MEGA_GATE:-1}"
# FP4 indexer 实现走上游默认（lightop + triton + stock），不显式设置。
# 真正决定走不走 FP4 的是 --attention-config 里的 indexer_kv_dtype=mxfp4。
# FP4 与 FP8 reference 互斥：走 FP4 就不能同时开 reference。
export VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE="0"
export VLLM_USE_V2_MODEL_RUNNER="1"
export VLLM_USE_BREAKABLE_CUDAGRAPH="1"
export VLLM_ENGINE_READY_TIMEOUT_S="3600"
# 图内 kernel 逐个 dispatch，便于观察。
export HIP_GRAPH_ACCUMULATE_DISPATCH="0"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export ROCSHMEM_BOOTSTRAP_TIMEOUT="1200"
export GPU_MAX_HW_QUEUES="3"
export ROCSHMEM_MAX_NUM_CONTEXTS="48"
export ROCSHMEM_DISABLE_HDP_FLUSH="1"
export DEEPEP_ENABLE_LL_LAYERED_OPT="1"

# ============ 网卡（每台节点网卡名不同，按节点自动探测） ============
_rdma_detect="$(detect_rdma_nics)"
_rdma_nics="$(printf '%s\n' "${_rdma_detect}" | sed -n '1p')"
_rdma_ibs="$(printf '%s\n' "${_rdma_detect}" | sed -n '2p')"
_rdma_ip="$(printf '%s\n' "${_rdma_detect}" | sed -n '3p')"
_rdma_master_nic="$(printf '%s\n' "${_rdma_detect}" | sed -n '4p')"
if [[ -z "${_rdma_nics}" || -z "${_rdma_ip}" ]]; then
  echo "acc-w4.sh: 未自动探测到带 RDMA 的业务网卡" >&2
  exit 1
fi
export VLLM_HOST_IP="${_rdma_ip}"
export NCCL_SOCKET_IFNAME="${_rdma_nics}"
export GLOO_SOCKET_IFNAME="${_rdma_master_nic:-${_rdma_nics}}"
export NCCL_IB_HCA="${_rdma_ibs}"
export UCX_NET_DEVICES="$(printf '%s' "${_rdma_ibs}" | cut -d, -f1):1"
echo "Network: VLLM_HOST_IP=${VLLM_HOST_IP} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_IB_HCA=${NCCL_IB_HCA}"

# ============ 服务与评测参数 ============
# Explicit Channel-FP8 checkpoint; the supplied script's workload is retained.
model_path="${MODEL_PATH:-/models/DeepSeek-V4.1-Flash-Channel-FP8}"
tensor_parallel_size="1"
data_parallel_size="8"
host="127.0.0.1"
port="6551"
served_model_name="dsv41-flash-channel-fp8"

# HumanEval first32, batch16, temperature0, seed42; see PROTOCOL.md.
datasets="humaneval"
eval_batch_size=16
eval_seed=42
eval_enable_thinking=false
generation_config='{"timeout":120000,"max_tokens":16384,"temperature":0,"top_p":0.95,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'

###############################################################################
# 环境自检
###############################################################################
HCU_ADAPTER_HOST=none "${PYTHON_BIN}" - <<'PY' || exit 1
import sys

if sys.version_info < (3, 11):
    raise SystemExit("acc-w4.sh: vLLM 0.33 插件需要 Python 3.11 或更新版本")

from importlib import metadata, util

versions = {}
missing = []
for name in ("vllm", "sgl-kernel", "hcu-vllm-das-plugin"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit(
        f"acc-w4.sh: {sys.executable} 缺少已安装包: {', '.join(missing)}"
    )

from packaging.version import Version

version = Version(versions["vllm"])
if version.epoch != 0 or version.release[:2] != (0, 33):
    raise SystemExit(f"acc-w4.sh: 需要 vLLM 0.33.x，当前为 {version}")
for name, ver in versions.items():
    print(f"{name}: {ver}", flush=True)
for name in ("vllm", "hcu_adapter", "hcu_vllm_das"):
    spec = util.find_spec(name)
    if spec is None:
        raise SystemExit(f"acc-w4.sh: 找不到已安装模块 {name}")
    print(f"{name}: {spec.origin}", flush=True)
for group, name in (
    ("vllm.platform_plugins", "hcu_adapter"),
    ("vllm.general_plugins", "hcu_adapter"),
    ("hcu_adapter.extensions", "das_hcu_dsv41"),
    ("hcu_adapter.extensions", "das_hcu_ct_w4a8"),
):
    entries = [entry for entry in metadata.entry_points(group=group) if entry.name == name]
    if not entries:
        raise SystemExit(f"acc-w4.sh: 缺少入口 {group}:{name}")
    # 多个 dist（sgl-kernel / hcu-adapter）可能注册同名入口。
    # 只要指向同一个 value 就是幂等的（vLLM 加载时覆盖，不冲突）；值不同才是真冲突。
    values = {entry.value for entry in entries}
    if len(values) != 1:
        raise SystemExit(
            f"acc-w4.sh: 入口 {group}:{name} 指向多个不同实现: {sorted(values)}"
        )
    if len(entries) > 1:
        dists = sorted({entry.dist.name for entry in entries if entry.dist})
        print(f"{group}:{name} -> {entries[0].value} (来自 {', '.join(dists)})", flush=True)
PY

for executable in setsid curl evalscope; do
  command -v "${executable}" >/dev/null || {
    echo "acc-w4.sh: 缺少命令 ${executable}，请在评测环境中安装" >&2
    exit 1
  }
done
[[ -r "${model_path%/}/config.json" ]] || {
  echo "acc-w4.sh: 找不到模型配置 ${model_path%/}/config.json" >&2
  exit 1
}
if [[ "${check_only}" == 1 ]]; then
  echo "acc-w4.sh: 安装、入口、模型路径、依赖及输出目录写入检查通过；未启动服务或评测"
  exit 0
fi

###############################################################################
# 启动服务（对齐 laibao/033/w4/acc_d.sh CLI）
###############################################################################
run_dir="${run_root}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}" || exit 1
cp "${script_path}" "${run_dir}/accuracy_fp8.sh" || exit 1
service_log="${run_dir}/vllm_server.log"

serve_cmd=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${model_path}"
  --tensor-parallel-size "${tensor_parallel_size}"
  --data-parallel-size "${data_parallel_size}"
  --enable-expert-parallel
  --moe-backend deep_gemm
  --all2all-backend deepep_low_latency
  --gpu-memory-utilization 0.90
  --block-size 64
  --max-num-seqs 8
  --max-model-len 32768 \
  --max-num-batched-tokens 64
  --kernel-config '{"enable_jit_warmup":false}'
  --cudagraph-capture-sizes 1 4 8 16 32 40 48 64
  -cc "{\"cudagraph_mode\":\"${cudagraph_mode}\"}"
  --attention-config '{"indexer_kv_dtype":"mxfp4"}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"moe_backend":"deep_gemm"}'
  --engram-config '{"cpu_offload": true, "dp_shared_memory": false, "embedding_across_dp": true}'
  --no-enable-prefix-caching
  --reasoning-parser deepseek_v41
  --skip-mm-profiling
  --host "${host}"
  --port "${port}"
  --served-model-name "${served_model_name}"
)
printf '%s\n' "${serve_cmd[@]}" > "${run_dir}/serve_cmd.txt"

setsid "${serve_cmd[@]}" > "${service_log}" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${run_dir}/server-pgid.txt"

wait_for_server "${host}" "${port}" "${server_pid}" "${service_log}" || exit 1
save_config_yaml
save_model_config_snapshot

echo "Server is ready. Running evalscope on Decode..."
set +e

eval_cmd=(
  evalscope eval
  --model "${served_model_name}"
  --api-url "http://${host}:${port}/v1"
  --api-key EMPTY
  --eval-type openai_api
  --generation-config "${generation_config}"
  --stream
  --eval-batch-size "${eval_batch_size}"
  --timeout 600000
  --work-dir "${run_dir}"
  --limit 32
  --datasets "${datasets}"
  --seed "${eval_seed}"
)
printf '%s\n' "${eval_cmd[@]}" > "${run_dir}/eval_cmd.txt"

HCU_ADAPTER_HOST=none "${eval_cmd[@]}" 2>&1 | tee "${run_dir}/evalscope.log"
eval_exit_code=${PIPESTATUS[0]}
set -e

if [[ ${eval_exit_code} -ne 0 ]]; then
  echo "WARNING: evalscope exited with code ${eval_exit_code}" >&2
fi

echo "Accuracy test finished. Stopping server..."
exit "${eval_exit_code}"
