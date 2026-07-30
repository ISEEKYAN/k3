#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_ROOT:?set K3_ROOT to the pinned K3 checkout}"
: "${MLITE_ROOT:?set MLITE_ROOT to the pinned Megatron-LM checkout}"
: "${VERL_ROOT:?set VERL_ROOT to the pinned VERL checkout}"
: "${VERL_DEPS_SITE:?set VERL_DEPS_SITE to the pinned VERL dependency site}"
: "${K3_CACHE_ROOT:?set K3_CACHE_ROOT to a persistent shared cache root}"
: "${MEGATRON_ROOT:=${K3_MEGATRON_ROOT}}"
: "${FLA_SITE:=${K3_FLA_SITE}}"
: "${VLLM_SITE:=${K3_VLLM_SITE}}"
export MEGATRON_ROOT FLA_SITE VLLM_SITE MLITE_SOURCE_SHA

assert_source_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(git -C "${path}" rev-parse HEAD)
  if [[ "${actual}" != "${expected}" && "${actual}" != "${expected}"* ]]; then
    echo "FATAL ${label} source mismatch expected=${expected} actual=${actual}" >&2
    return 2
  fi
}

assert_source_sha "${MLITE_ROOT}" "${MLITE_SOURCE_SHA}" MLite
assert_source_sha "${VERL_ROOT}" "${VERL_SOURCE_SHA}" VERL
assert_source_sha "${MEGATRON_ROOT}" "${MCORE_SOURCE_SHA}" MCore
k3_source_sha=$(git -C "${K3_ROOT}" rev-parse HEAD)
if ! git -C "${K3_ROOT}" merge-base --is-ancestor "${K3_BASE_SOURCE_SHA}" "${k3_source_sha}"; then
  echo "FATAL K3 source is not based on ${K3_BASE_SOURCE_SHA}: ${k3_source_sha}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
unset \
  CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV \
  CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
  CC CXX CFLAGS CPPFLAGS CXXFLAGS LDFLAGS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLITE_K3_AUTO_REGISTER=1
export PYTHONPATH="${recipe_dir}:${K3_ROOT}/src:${MLITE_ROOT}/experimental/lite:${MLITE_ROOT}/experimental/lite/examples/verl:${MEGATRON_ROOT}:${VERL_ROOT}:${VLLM_SITE}:${FLA_SITE}:${VERL_DEPS_SITE}"

python_bin=$(command -v python3)
python_version=$("${python_bin}" -c 'import platform; print(platform.python_version())')
torch_version=$("${python_bin}" -c 'import torch; print(torch.__version__)')
te_version=$("${python_bin}" -c 'import importlib.metadata; print(importlib.metadata.version("transformer-engine"))')
fla_version=$("${python_bin}" -c 'import importlib.metadata; print(importlib.metadata.version("flash-linear-attention"))')
vllm_version=$("${python_bin}" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')
if [[ -n "${K3_GPU_CC:-}" ]]; then
  gpu_cc="${K3_GPU_CC}"
else
  gpu_cc_output=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
  gpu_cc="${gpu_cc_output%%$'\n'*}"
  gpu_cc="${gpu_cc//[[:space:].]/}"
fi

training_image_stat=$(stat -Lc "%s:%Y" "${K3_TRAINING_IMAGE}")
runtime_image_fingerprint="sqsh-stat:${K3_TRAINING_IMAGE}:${training_image_stat}"
recipe_fingerprint=$(
  sha256sum "${recipe_dir}/image.env" "${BASH_SOURCE[0]}" \
    | sha256sum \
    | cut -d' ' -f1
)

fingerprint_input="$(
  printf '%s\n' \
    "${runtime_image_fingerprint}" \
    "${recipe_fingerprint}" \
    "${k3_source_sha}" \
    "${MLITE_SOURCE_SHA}" \
    "${MCORE_SOURCE_SHA}" \
    "${VERL_SOURCE_SHA}" \
    "${python_version}" \
    "${torch_version}" \
    "${te_version}" \
    "${fla_version}" \
    "${vllm_version}" \
    "${gpu_cc}" \
    "${PYTHONPATH}"
)"
fingerprint=$(printf '%s' "${fingerprint_input}" | sha256sum | cut -c1-20)
cache_dir="${K3_CACHE_ROOT}/${fingerprint}"
manifest="${cache_dir}/fingerprint.txt"
mkdir -p "${cache_dir}"
if [[ -e "${manifest}" ]]; then
  actual=$(<"${manifest}")
  if [[ "${actual}" != "${fingerprint_input}" ]]; then
    echo "FATAL JIT cache fingerprint mismatch path=${manifest}" >&2
    exit 2
  fi
else
  printf '%s' "${fingerprint_input}" >"${manifest}"
fi

export K3_JIT_CACHE_FINGERPRINT="${fingerprint}"
export TRITON_CACHE_DIR="${cache_dir}/triton"
export TORCHINDUCTOR_CACHE_DIR="${cache_dir}/torchinductor"
export TILELANG_CACHE_DIR="${cache_dir}/tilelang"
export TILELANG_TMP_DIR="${cache_dir}/tilelang-tmp"
export PYTHONPYCACHEPREFIX="${cache_dir}/pycache"
mkdir -p \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}" \
  "${PYTHONPYCACHEPREFIX}"
echo "K3_JIT_CACHE_OK fingerprint=${fingerprint} root=${cache_dir}" >&2
