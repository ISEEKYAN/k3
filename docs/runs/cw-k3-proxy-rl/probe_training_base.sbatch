#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --job-name=k3-probe-train-base
#SBATCH --output=k3-probe-train-base-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
: "${K3_ROOT:?set K3_ROOT to the pinned K3 checkout}"
: "${MLITE_ROOT:?set MLITE_ROOT to the pinned MLite checkout}"
: "${VERL_ROOT:?set VERL_ROOT to the pinned VERL checkout}"
: "${MODEL_PATH:?set MODEL_PATH to the proxy checkpoint}"

base_image=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh
mlite_sm90_site=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/mlite-2604-verl-dsa-sm90-overlay/lib/python3.12/site-packages
vllm_site=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/mlite-2612-cu13-canonical/vllm0251-site
qwen35_site=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/mlite-newenv-cache/qwen35-proven-canary-site
megatron_root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/ds4-csacp-parity-eaa5b486d/mcore
verl_deps=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/verl-latest-v1-deps

srun \
  --account=coreai_devtech_all \
  --container-image="${base_image}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TILELANG_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/k3-tilelang-cache"
export TILELANG_TMP_DIR="${SLURM_TMPDIR:-/tmp}/k3-tilelang-tmp"
export PYTHONPATH="'"${K3_ROOT}"'/src:'"${MLITE_ROOT}"'/experimental/lite:'"${MLITE_ROOT}"'/experimental/lite/examples/verl:'"${megatron_root}"':'"${VERL_ROOT}"':'"${vllm_site}"':'"${qwen35_site}"':'"${mlite_sm90_site}"'/nvidia_cutlass_dsl/python_packages:'"${mlite_sm90_site}"':'"${verl_deps}"'"
/usr/bin/python3 "'"${recipe_dir}"'/probe_training_base.py"
'
