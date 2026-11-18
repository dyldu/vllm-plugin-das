#!/usr/bin/env bash
# Channel-FP8 paired decode benchmark and torch profiler; see PROTOCOL.md.
# 起服务 + 测试口径对齐 laibao/033/w4/prof_d.sh（DecodeBench fake prefill + DSpark + vllm bench serve）。
# 源码路径保留本脚本的 HCU_* overlay（vllm-hcu033 / adapter-das / adapter_extension）。
# 容器：duyl-033-1112（/duyl -> /public/home/duyunlong，/models -> llm-models 挂载）。
#
# Usage (容器内):
#   HCU_VLLM_DIR=... HCU_ADAPTER_SRC=... MODEL_PATH=... RUN_ROOT=... bash profile_fp8_paired.sh
#
# 所有参数已写死（对齐 laibao/033/w4/prof_d.sh），不再接受环境变量覆盖。

set -uo pipefail

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${prof_stop_pid:-}" ]]; then
    kill "${prof_stop_pid}" 2>/dev/null || true
  fi
  if [[ -n "${tail_pid:-}" ]] && kill -0 "${tail_pid}" 2>/dev/null; then
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid:-}" ]] && kill -0 -- "-${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for ((i = 0; i < 60; i++)); do
      kill -0 -- "-${server_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  if [[ -n "${run_dir:-}" ]]; then
    echo "${rc}" > "${run_dir}/exit_code"
    chmod -R a+rX "${run_dir}" 2>/dev/null || true
  fi
    echo "Channel-FP8 decode (033) workflow exit=${rc}; results=${run_dir:-}"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local host="$1" port="$2" server_pid="$3" service_log="$4"
  local max_attempts="${5:-3000}" interval_sec="${6:-2}"
  local timeout_sec=$((max_attempts * interval_sec))
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
    if curl --noproxy '*' -fsS "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
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
    echo "# vLLM Server Configuration (prof-w4 / aligned laibao/033/prof_d)"
    echo "timestamp: \"$(date '+%Y-%m-%d %H:%M:%S')\""
    echo "run_dir: \"${run_dir}\""
    echo "mode: channel_fp8_paired_decode"
    echo ""
    echo "# 源码路径"
    echo "paths:"
    echo "  HCU_DEV_OVERLAY: \"${HCU_DEV_OVERLAY}\""
    echo "  HCU_VLLM_DIR: \"${HCU_VLLM_DIR}\""
    echo "  HCU_VLLM_DAS_SRC: \"${HCU_VLLM_DAS_SRC}\""
    echo "  HCU_ADAPTER_SRC: \"${HCU_ADAPTER_SRC}\""
    echo "  PYTHONPATH: \"${PYTHONPATH:-}\""
    echo ""
    echo "# 服务参数"
    echo "server:"
    echo "  model_path: \"${model_path}\""
    echo "  tensor_parallel_size: \"${tensor_parallel_size}\""
    echo "  data_parallel_size: \"${data_parallel_size}\""
    echo "  host: \"${host}\""
    echo "  port: ${port}"
    echo "  served_model_name: \"${served_model_name}\""
    echo "  fake_prefill: DecodeBenchConnector"
    echo "  dspark_synthetic: ${dspark_synthetic} p=${dspark_accept_p} n=${num_spec_tokens}"
    echo "  max_concurrency: ${prof_max_concurrency}"
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
writable_script_path="/duyl/run/dsv4.1/033/prof-w4.sh"
if [[ ! -w "${SCRIPT_DIR}" && "${writable_script_path}" -ef "${script_path}" ]]; then
  script_path="${writable_script_path}"
  SCRIPT_DIR="$(dirname "${script_path}")"
fi
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")" || {
  echo "prof_decode_fake_prefill.sh: 找不到 python3" >&2
  exit 1
}

# Derive RDMA fabric（对齐 laibao/033/prof_d.sh：NCCL 多 NIC，GLOO 绑 master NIC）
detect_net() {
  PYTHONPATH= "${PYTHON_BIN}" -S -B - <<'NETPY'
import fcntl, os, socket, struct

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
NETPY
}

###############################################################################
# 源码路径（保留本脚本 overlay；起服/测试口径走 laibao/033/prof_d）
###############################################################################
HCU_DEV_OVERLAY=1
HCU_VLLM_DIR="${HCU_VLLM_DIR:?set worktree}"
HCU_VLLM_DAS_SRC="${HCU_VLLM_DIR}/adapter_extension/src"
HCU_ADAPTER_SRC="${HCU_ADAPTER_SRC:?set adapter worktree}"
unset PYTHONPATH
if [[ "${HCU_DEV_OVERLAY}" != "0" ]]; then
  [[ -f "${HCU_VLLM_DIR}/vllm_hcu/__init__.py" ]] || {
    echo "prof-w4.sh: HCU_VLLM_DIR 无效: ${HCU_VLLM_DIR}" >&2
    exit 1
  }
  [[ -f "${HCU_VLLM_DAS_SRC}/hcu_vllm_das/__init__.py" ]] || {
    echo "prof-w4.sh: HCU_VLLM_DAS_SRC 无效: ${HCU_VLLM_DAS_SRC}" >&2
    exit 1
  }
  [[ -f "${HCU_ADAPTER_SRC}/hcu_adapter/__init__.py" ]] || {
    echo "prof-w4.sh: HCU_ADAPTER_SRC 无效: ${HCU_ADAPTER_SRC}" >&2
    exit 1
  }
  export PYTHONPATH="${HCU_VLLM_DIR}:${HCU_ADAPTER_SRC}:${HCU_VLLM_DAS_SRC}"
  echo "[dev-overlay] PYTHONPATH=${PYTHONPATH}"
else
  echo "[dev-overlay] off (installed site-packages only)"
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=hcu_adapter
export HCU_ADAPTER_HOST=vllm
export HCU_ADAPTER_EXTENSION_OVERLAY=auto

###############################################################################
# 环境参数（对齐 laibao/033/prof_d.sh）
###############################################################################
# Engram cpu_offload pins ~12GiB/rank; docker default memlock is 64k and
# silently kills EngineCore during model init (Failed core proc(s): {}).
ulimit -l unlimited 2>/dev/null || true

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"

export VLLM_HCU_USE_CUSTOM_OPS=1
export HCU_VLLM_MHC_MEGA=1
export VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH="${VLLM_HCU_SHARED_EXPERTS_MID_DISPATCH:-1}"
export VLLM_HCU_USE_FUSED_SILU_MUL_QUANT="${VLLM_HCU_USE_FUSED_SILU_MUL_QUANT:-1}"
export VLLM_HCU_USE_BF16_MEGA_GATE="${VLLM_HCU_USE_BF16_MEGA_GATE:-1}"
# DSpark + 离线 decode-bench connector 需要这个 opt-in；不写默认 "0" 会拒绝启动。
export VLLM_HCU_DSPARK_ALLOW_DECODE_BENCH=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# 图内 kernel 逐个 dispatch，便于 profiler 采集。
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
_net_detect="$(detect_net)"
_net_nics="$(printf '%s\n' "${_net_detect}" | sed -n '1p')"
_net_ibs="$(printf '%s\n' "${_net_detect}" | sed -n '2p')"
_net_ip="$(printf '%s\n' "${_net_detect}" | sed -n '3p')"
_net_master_nic="$(printf '%s\n' "${_net_detect}" | sed -n '4p')"
export VLLM_HOST_IP="${_net_ip}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export ROCSHMEM_BOOTSTRAP_TIMEOUT=1200
export GPU_MAX_HW_QUEUES=3
export NCCL_SOCKET_IFNAME="${_net_nics}"
export GLOO_SOCKET_IFNAME="${_net_master_nic:-${_net_nics}}"
export NCCL_IB_HCA="${_net_ibs}"
export UCX_NET_DEVICES="$(printf '%s' "${_net_ibs}" | cut -d, -f1):1"
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
echo "Network: VLLM_HOST_IP=${VLLM_HOST_IP} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_IB_HCA=${NCCL_IB_HCA}"

vllm_cache_root="/tmp/vllm_cache_duyl"
mkdir -p "${vllm_cache_root}/inductor" "${vllm_cache_root}/triton"
export VLLM_CACHE_ROOT="${vllm_cache_root}"
export TRITON_CACHE_DIR="${vllm_cache_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${vllm_cache_root}/inductor"

for executable in setsid curl; do
  command -v "${executable}" >/dev/null || {
    echo "prof_decode_fake_prefill.sh: 缺少命令 ${executable}" >&2
    exit 1
  }
done

###############################################################################
# 服务参数（对齐 laibao/033/prof_d.sh）
###############################################################################
# Channel-FP8 checkpoint; retain the supplied workload and mxfp4 indexer.
model_path="${MODEL_PATH:-/models/DeepSeek-V4.1-Flash-Channel-FP8}"
[[ -r "${model_path%/}/config.json" ]] || {
  echo "找不到 Channel-FP8 模型配置 ${model_path%/}/config.json" >&2
  exit 1
}
echo "[channel-fp8] model_path=${model_path}"
tensor_parallel_size="1"
data_parallel_size="8"
host="127.0.0.1"
port="6552"
served_model_name="dsv41-flash-channel-fp8"

# bench 参数与 laibao/033/prof_d.sh 一致（提前赋值，避免 set -u）
prof_warmup_requests=8
prof_measure_requests=64
prof_input_len=4096
prof_output_len=128
prof_max_concurrency=64
prof_stop_after=0

kv_transfer_flag=(
  --kv-transfer-config
  '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.0,"fill_std":0.0}}'
)
echo "[fake-prefill] DecodeBenchConnector enabled (fill_mean=0.0)"

# laibao/033/w4/prof_d.sh: indexer KV dtype is mxfp4, impl stays lightop+triton.
extra_args=(
  --attention-config '{"indexer_kv_dtype":"mxfp4"}'
)
echo "[indexer] FP4 lightop/triton + indexer_kv_dtype=mxfp4 (upstream defaults)"

dspark_synthetic=1
dspark_accept_p=0.6
num_spec_tokens=5
speculative_config="{\"method\":\"dspark\",\"num_speculative_tokens\":${num_spec_tokens},\"moe_backend\":\"deep_gemm\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rates\":[0.6,0.6,0.6,0.6,0.6]}"
echo "[dspark] synthetic acceptance rates=[0.6,0.6,0.6,0.6,0.6]"

run_root="${RUN_ROOT:?set output root}"
mkdir -p "${run_root}" || exit 1
run_dir="$(mktemp -d "${run_root}/$(date +%Y%m%d_%H%M%S)_channel_fp8_XXXXXX")"
chmod 755 "${run_dir}"
run_dir="$(realpath "${run_dir}")"
cp -f "${script_path}" "${run_dir}/profile_fp8_paired.sh"
service_log="${run_dir}/vllm_server.log"
profiler_root="${run_dir}/torch_profiler"
mkdir -p "${profiler_root}"
: > "${service_log}"
chmod 644 "${service_log}"

server_pid=""
tail_pid=""
prof_stop_pid=""

###############################################################################
# 启动服务（对齐 laibao/033/prof_d.sh CLI）
###############################################################################
setsid env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${model_path}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --data-parallel-size "${data_parallel_size}" \
  --enable-expert-parallel \
  --moe-backend deep_gemm \
  --all2all-backend deepep_low_latency \
  --gpu-memory-utilization 0.90 \
  --block-size 64 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 64 \
  --kernel-config '{"enable_jit_warmup":false}' \
  --max-model-len 8192 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 40 48 64 \
  -cc '{"cudagraph_mode":"PIECEWISE"}' \
  --engram-config '{"cpu_offload": true, "dp_shared_memory": false, "embedding_across_dp": true}' \
  --speculative-config "${speculative_config}" \
  --no-enable-prefix-caching \
  --reasoning-parser deepseek_v41 \
  --skip-mm-profiling \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profiler_root}\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true,\"wait_iterations\":5,\"warmup_iterations\":2,\"active_iterations\":10,\"delay_iterations\":5,\"max_iterations\":0}" \
  "${kv_transfer_flag[@]}" \
  "${extra_args[@]}" \
  --host "${host}" \
  --port "${port}" \
  --served-model-name "${served_model_name}" \
  >>"${service_log}" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" > "${run_dir}/server-pgid.txt"

tail_pid=""

echo "Starting 033 Channel-FP8 paired decode/profile (aligned laibao/033/prof_d): PID=${server_pid}"
echo "  model: ${model_path}"
echo "  results=${run_dir}"
echo "  log: ${service_log}"

wait_for_server "${host}" "${port}" "${server_pid}" "${service_log}" || exit 1
save_config_yaml
save_model_config_snapshot

###############################################################################
# 请求口径对齐 laibao/033/prof_d：vllm bench serve（random 4k 输入 / 128 输出）
# 4k 输入 + 64 并发才覆盖稳态 decode；单条小请求只能抓到启动和 dummy batch。
###############################################################################
# bench 在 warmup 之后自动 POST /start_profile；Kineto 真正开启要等
# delay_iterations 步跑完（日志出现 "Starting profiler after delay"）。
# 等到这个标志后再过 N 秒统一 stop，避免尾部 dummy batch 混进 trace。
if [[ "${prof_stop_after}" != "0" ]]; then
  (
    set +e
    for _ in $(seq 1 36000); do
      grep -q "Profiler started\." "${service_log}" 2>/dev/null && break
      sleep 0.1
    done
    for _ in $(seq 1 18000); do
      grep -q "Starting profiler after delay" "${service_log}" 2>/dev/null && break
      sleep 0.1
    done
    sleep "${prof_stop_after}"
    if curl --noproxy '*' -fsS -X POST "http://${host}:${port}/stop_profile" >/dev/null 2>&1; then
      echo "[prof] early /stop_profile sent ${prof_stop_after}s after Kineto start"
    fi
  ) &
  prof_stop_pid=$!
fi

# Formal performance is measured with profiling disabled. Profile is a separate pass.
for phase in formal_1 formal_2 profile; do
profile_args=()
if [[ "${phase}" == profile ]]; then profile_args=(--profile); fi
echo "Running vllm bench serve; phase=${phase}"
"${PYTHON_BIN}" -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --host "${host}" \
  --port "${port}" \
  --endpoint /v1/completions \
  --model "${served_model_name}" \
  --tokenizer "${model_path}" \
  --dataset-name random \
  --random-input-len "${prof_input_len}" \
  --random-output-len "${prof_output_len}" \
  --num-prompts "${prof_measure_requests}" \
  --num-warmups "${prof_warmup_requests}" \
  --max-concurrency "${prof_max_concurrency}" \
  --ignore-eos \
  --temperature 0 \
  --seed 0 \
  "${profile_args[@]}" \
  --disable-tqdm \
  --save-result \
  --result-dir "${run_dir}" \
  --result-filename "${phase}_result.json" \
  2>&1 | tee "${run_dir}/${phase}_serve.log"
bench_rc=${PIPESTATUS[0]}
printf '%s\n' "${bench_rc}" > "${run_dir}/${phase}.exit"
[[ "${bench_rc}" == 0 ]] || exit "${bench_rc}"
done
if [[ -n "${prof_stop_pid}" ]]; then
  kill "${prof_stop_pid}" 2>/dev/null || true
  prof_stop_pid=""
fi
curl --noproxy '*' -s -X POST "http://${host}:${port}/stop_profile" >/dev/null 2>&1 || true

echo "Profiling finished. Results saved in: ${profiler_root}"
echo "Bench results: ${run_dir}/formal_{1,2}_result.json"
ls -la "${profiler_root}" || true
n_traces="$(find "${profiler_root}" -type f \( -name '*.json' -o -name '*.json.gz' \) 2>/dev/null | wc -l)"
echo "Trace file count: ${n_traces} (expect ~8 for DP8)"
echo "Stopping server..."
exit "${bench_rc}"
