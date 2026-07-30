#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=k3-k2-proxy-bench
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/k3-k2-proxy-bench-20260729/slurm-%j.out

set -euo pipefail

: "${K3_PROXY_ARM:?set K3_PROXY_ARM}"
: "${HARNESS_COMMIT:?set HARNESS_COMMIT}"

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
RUN=${BASE}/runtime/k3-k2-proxy-bench-20260729
K3_SOURCE=${RUN}/k3-361e01b
MLITE_SOURCE=${RUN}/mlite-648f0b8
CONTAINER_IMAGE=${BASE}/verl_optimize/verl.vllm016.dev.qwen3_5.sqsh
CU12_SITE=${BASE}/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
QWEN_RUN=${BASE}/qwen35_dapo_rlrun
TE_PATCH=${QWEN_RUN}/te_patch/permutation.py
TE_DST=/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/permutation.py
HARNESS=${RUN}/k3_k2_same_size_bench.py
K3_SOURCE_COMMIT=361e01bfe208c44f2df6f36ea5c5d4071585a584
MLITE_SOURCE_COMMIT=648f0b884ac60bdea8427e31801a64ea6d367407

mkdir -p "${RUN}/home" "${RUN}/torchrun-logs"
echo "SELF_PROOF arm=${K3_PROXY_ARM} harness=${HARNESS_COMMIT} k3=${K3_SOURCE_COMMIT} mlite=${MLITE_SOURCE_COMMIT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

srun --unbuffered \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/lustre:/lustre,"${TE_PATCH}":"${TE_DST}" \
  --container-workdir="${RUN}" \
  --export=ALL \
  env \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${QWEN_RUN}/fla_override:${K3_SOURCE}/src:${CU12_SITE}:${MLITE_SOURCE}/experimental/lite:${RUN}" \
    LD_LIBRARY_PATH="${CU12_SITE}/tvm_ffi/lib" \
    HOME="${RUN}/home" \
    XDG_CACHE_HOME="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/xdg" \
    TRITON_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/triton" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/inductor" \
    TILELANG_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/tilelang" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_DEBUG=INFO \
    WANDB_PROJECT=k3-k2-proxy-benchmark \
    K3_PROXY_ARM="${K3_PROXY_ARM}" \
    K3_SOURCE_COMMIT="${K3_SOURCE_COMMIT}" \
    MLITE_SOURCE_COMMIT="${MLITE_SOURCE_COMMIT}" \
    HARNESS_COMMIT="${HARNESS_COMMIT}" \
    /usr/bin/python -c "import wandb; from mlite_k3.config import K3Config; from megatron.lite.model.kimi_k2.config import KimiK2Config; print('CONFIG_AND_WANDB_IMPORT_OK')" \
  && srun --unbuffered \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/lustre:/lustre,"${TE_PATCH}":"${TE_DST}" \
  --container-workdir="${RUN}" \
  --export=ALL \
  env \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${QWEN_RUN}/fla_override:${K3_SOURCE}/src:${CU12_SITE}:${MLITE_SOURCE}/experimental/lite:${RUN}" \
    LD_LIBRARY_PATH="${CU12_SITE}/tvm_ffi/lib" \
    HOME="${RUN}/home" \
    XDG_CACHE_HOME="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/xdg" \
    TRITON_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/triton" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/inductor" \
    TILELANG_CACHE_DIR="/tmp/k3-k2-${SLURM_JOB_ID}-${K3_PROXY_ARM}/tilelang" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_DEBUG=INFO \
    WANDB_PROJECT=k3-k2-proxy-benchmark \
    K3_PROXY_ARM="${K3_PROXY_ARM}" \
    K3_SOURCE_COMMIT="${K3_SOURCE_COMMIT}" \
    MLITE_SOURCE_COMMIT="${MLITE_SOURCE_COMMIT}" \
    HARNESS_COMMIT="${HARNESS_COMMIT}" \
    /usr/bin/python -m torch.distributed.run \
      --standalone \
      --nproc_per_node=8 \
      --redirects=3 \
      --tee=3 \
      --log-dir="${RUN}/torchrun-logs/${SLURM_JOB_ID}-${K3_PROXY_ARM}" \
      "${HARNESS}"
