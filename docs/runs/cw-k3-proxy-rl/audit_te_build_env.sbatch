#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=k3-audit-te
#SBATCH --output=k3-audit-te-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${TE_SOURCE:?set TE_SOURCE to the pinned Transformer Engine checkout}"
: "${TE_BUILD_DEPS:?set TE_BUILD_DEPS to a new build-dependency directory}"
if [[ -e "${TE_BUILD_DEPS}" ]]; then
  echo "FATAL TE_BUILD_DEPS already exists: ${TE_BUILD_DEPS}" >&2
  exit 2
fi
mkdir -p "${TE_BUILD_DEPS}"

srun \
  --container-image="${K3_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export NVTE_WITH_NCCL_EP=0

/usr/bin/python3 -m pip install \
  --no-deps \
  --target "${TE_BUILD_DEPS}" \
  -r "'"${recipe_dir}"'/te-build-requirements.txt"
nvcc=$(find "${TE_BUILD_DEPS}" -type f -name nvcc -print -quit)
if [[ -n "${nvcc}" ]]; then
  export CUDA_HOME
  CUDA_HOME=$(dirname "$(dirname "${nvcc}")")
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export PYTHONPATH="${TE_BUILD_DEPS}"
/usr/bin/python3 "'"${recipe_dir}"'/audit_te_build_env.py"
'
