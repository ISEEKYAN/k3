#!/usr/bin/env bash
# Kimi-K3 proxy offline generate: the proven vLLM external launcher at TP8+EP8.
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=k3-proxy-generate
#SBATCH --output=k3-proxy-generate-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${MODEL_PATH:?set MODEL_PATH to the 12-layer, 56-expert proxy checkpoint}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a shared output directory}"

unset RAY_ADDRESS
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH=/usr/include/x86_64-linux-gnu
export PYTHONPATH="${K3_VLLM_SITE}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${K3_VLLM_SITE}/tvm_ffi/lib"
export XDG_CACHE_HOME="${K3_CACHE_ROOT}/vllm-generate/xdg"
export VLLM_CACHE_ROOT="${K3_CACHE_ROOT}/vllm-generate/vllm"
export VLLM_CONFIG_ROOT="${K3_CACHE_ROOT}/vllm-generate/vllm-config"
export FLASHINFER_WORKSPACE_BASE="${K3_CACHE_ROOT}/vllm-generate/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${K3_CACHE_ROOT}/vllm-generate/torchinductor"
export TRITON_CACHE_DIR="${K3_CACHE_ROOT}/vllm-generate/triton"
mkdir -p \
  "${OUTPUT_ROOT}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}"

head_node=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
head_ip=$(
  /usr/bin/python3 -c \
    'import socket,sys; print(socket.gethostbyname(sys.argv[1]))' "${head_node}"
)
export MASTER_ADDR="${head_ip}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export K3_MODEL_PATH="${MODEL_PATH}"
export K3_RESPONSE_FILE="${OUTPUT_ROOT}/k3-proxy-generate-${SLURM_JOB_ID}.json"
export WANDB_ENTITY="${WANDB_ENTITY:-megatron-core-moe-dev}"
export WANDB_PROJECT="${WANDB_PROJECT:-k3-proxy-generate}"
export RUN_NAME="${RUN_NAME:-k3-proxy-12l-56e-generate}"

diagnose() {
  echo "K3_GENERATE_DIAG phase=$1 job=${SLURM_JOB_ID} time=$(date -Is)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
  ps -C python3 -o pid,ppid,stat,etime,cmd
}
(
  sleep 120
  diagnose t_plus_2m
  sleep 180
  diagnose t_plus_5m
) &
diag_pid=$!

set +e
srun \
  --account=coreai_devtech_all \
  --container-image="${K3_TRAINING_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,PATH=${PATH},CC=${CC},CXX=${CXX},CPATH=${CPATH},PYTHONNOUSERSITE=${PYTHONNOUSERSITE},OMP_NUM_THREADS=${OMP_NUM_THREADS},PYTHONPATH=${PYTHONPATH},LD_LIBRARY_PATH=${LD_LIBRARY_PATH},XDG_CACHE_HOME=${XDG_CACHE_HOME},VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT},VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT},FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE},TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR},TRITON_CACHE_DIR=${TRITON_CACHE_DIR},MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},K3_MODEL_PATH=${K3_MODEL_PATH},K3_RESPONSE_FILE=${K3_RESPONSE_FILE},WANDB_ENTITY=${WANDB_ENTITY},WANDB_PROJECT=${WANDB_PROJECT},RUN_NAME=${RUN_NAME} \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH=/usr/include/x86_64-linux-gnu
unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
exec /usr/bin/python3 "'"${recipe_dir}"'/run_proxy_generate.py"
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
[[ "${run_rc}" == "0" ]]
[[ -s "${K3_RESPONSE_FILE}" ]]
cat "${K3_RESPONSE_FILE}"
