#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=k3-build-te
#SBATCH --output=k3-build-te-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${TE_SOURCE:?set TE_SOURCE to the pinned Transformer Engine checkout}"
: "${TE_OVERLAY:?set TE_OVERLAY to a new output directory}"

actual_sha=$(git -C "${TE_SOURCE}" rev-parse HEAD)
if [[ "${actual_sha}" != "${TE_SOURCE_SHA}" ]]; then
  echo "FATAL TE source mismatch expected=${TE_SOURCE_SHA} actual=${actual_sha}" >&2
  exit 2
fi
if [[ -e "${TE_OVERLAY}" ]]; then
  echo "FATAL TE_OVERLAY already exists: ${TE_OVERLAY}" >&2
  exit 2
fi

mkdir -p "${TE_OVERLAY}"
srun \
  --container-image="${K3_RUNTIME_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
base_python_site=/usr/local/lib/python3.12/dist-packages
base_cudnn_root="${base_python_site}/nvidia/cudnn"
export CPATH="${base_cudnn_root}/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${base_cudnn_root}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/targets/x86_64-linux/lib:${base_cudnn_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-32}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-32}"
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
# The K3 training path uses TE PyTorch modules, not the optional TE NCCL-EP
# extension. The published image exposes versioned NCCL runtime libraries but
# no development linker name, so fail-loudly disable only that optional build.
export NVTE_WITH_NCCL_EP=0

build_tmp_root="${SLURM_TMPDIR:-/tmp}"
export TMPDIR
TMPDIR=$(mktemp -d "${build_tmp_root}/k3-te-build.XXXXXX")
export PIP_CACHE_DIR="${TMPDIR}/pip-cache"
mkdir -p "${PIP_CACHE_DIR}"

/usr/bin/python3 -m pip install \
  --no-cache-dir \
  --no-deps \
  --target "${TE_OVERLAY}/cuda-toolchain" \
  -r "'"${recipe_dir}"'/te-build-requirements.txt"
export PYTHONPATH="${TE_OVERLAY}/cuda-toolchain"
nvcc --version

/usr/bin/python3 -m pip install \
  --no-cache-dir \
  --no-deps \
  --no-build-isolation \
  --target "${TE_OVERLAY}" \
  "${TE_SOURCE}"

PYTHONPATH="${TE_OVERLAY}" /usr/bin/python3 "'"${recipe_dir}"'/validate_te.py"
'

{
  printf 'SOURCE_SHA=%s\n' "${TE_SOURCE_SHA}"
  printf 'IMAGE_INDEX_DIGEST=%s\n' "${K3_IMAGE_INDEX_DIGEST}"
  printf 'IMAGE_AMD64_DIGEST=%s\n' "${K3_IMAGE_AMD64_DIGEST}"
} >"${TE_OVERLAY}/BUILD_OK"
