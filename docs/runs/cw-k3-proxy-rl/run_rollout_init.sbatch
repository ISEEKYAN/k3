#!/usr/bin/env bash
# Rollout-only K3 proxy initialization and one real generation at TP2.
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --job-name=k3-rollout-init
#SBATCH --output=k3-rollout-init-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${MODEL_PATH:?set MODEL_PATH to the 12-layer, 56-expert proxy checkpoint}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new shared output directory}"

unset RAY_ADDRESS
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export PATH=/cm/shared/apps/slurm/current/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH=/usr/include/x86_64-linux-gnu
export PYTHONPATH="${K3_VLLM_SITE}:${K3_CUTLASS_DSL_SITE}:${recipe_dir}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${K3_VLLM_SITE}/tvm_ffi/lib"
export XDG_CACHE_HOME="${K3_CACHE_ROOT}/rollout-init/xdg"
export VLLM_CACHE_ROOT="${K3_CACHE_ROOT}/rollout-init/vllm"
export VLLM_CONFIG_ROOT="${K3_CACHE_ROOT}/rollout-init/vllm-config"
export FLASHINFER_WORKSPACE_BASE="${K3_CACHE_ROOT}/rollout-init/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${K3_CACHE_ROOT}/rollout-init/torchinductor"
export TRITON_CACHE_DIR="${K3_CACHE_ROOT}/rollout-init/triton"
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
export K3_RESPONSE_FILE="${OUTPUT_ROOT}/k3-rollout-init-${SLURM_JOB_ID}.json"
export K3_ROLLOUT_TP=2
export K3_SUCCESS_MARKER=K3_ROLLOUT_INIT_OK
export K3_LOG_WANDB=0

diagnose() {
  echo "K3_ROLLOUT_DIAG phase=$1 job=${SLURM_JOB_ID} time=$(date -Is)"
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
  --no-container-entrypoint \
  --container-mounts=/lustre:/lustre \
  --export=PATH,CC,CXX,CPATH,PYTHONNOUSERSITE,OMP_NUM_THREADS,PYTHONPATH,LD_LIBRARY_PATH,XDG_CACHE_HOME,VLLM_CACHE_ROOT,VLLM_CONFIG_ROOT,FLASHINFER_WORKSPACE_BASE,TORCHINDUCTOR_CACHE_DIR,TRITON_CACHE_DIR,MASTER_ADDR,MASTER_PORT,K3_MODEL_PATH,K3_RESPONSE_FILE,K3_ROLLOUT_TP,K3_SUCCESS_MARKER,K3_LOG_WANDB \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
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
