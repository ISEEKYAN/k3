#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_ROOT:?set K3_ROOT to the pinned K3 checkout}"
: "${MLITE_ROOT:?set MLITE_ROOT to the pinned Megatron-LM checkout}"
: "${VERL_ROOT:?set VERL_ROOT to the pinned VERL checkout}"
: "${TE_SITE:?set TE_SITE to the pinned Python 3.12 Transformer Engine site}"
: "${VERL_DEPS_SITE:?set VERL_DEPS_SITE to the pinned VERL dependency site}"
: "${K3_CACHE_ROOT:?set K3_CACHE_ROOT to a persistent shared cache root}"

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
k3_source_sha=$(git -C "${K3_ROOT}" rev-parse HEAD)
if ! git -C "${K3_ROOT}" merge-base --is-ancestor "${K3_BASE_SOURCE_SHA}" "${k3_source_sha}"; then
  echo "FATAL K3 source is not based on ${K3_BASE_SOURCE_SHA}: ${k3_source_sha}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLITE_K3_AUTO_REGISTER=1
export PYTHONPATH="${recipe_dir}:${K3_ROOT}/src:${MLITE_ROOT}/experimental/lite:${MLITE_ROOT}/experimental/lite/examples/verl:${MLITE_ROOT}:${VERL_ROOT}:${TE_SITE}:${VERL_DEPS_SITE}"

python_bin=$(command -v python3)
python_version=$("${python_bin}" -c 'import platform; print(platform.python_version())')
torch_version=$("${python_bin}" -c 'import torch; print(torch.__version__)')
te_version=$("${python_bin}" -c 'import importlib.metadata; print(importlib.metadata.version("transformer-engine"))')
fla_version=$("${python_bin}" -c 'import importlib.metadata; print(importlib.metadata.version("flash-linear-attention"))')
if [[ -n "${K3_GPU_CC:-}" ]]; then
  gpu_cc="${K3_GPU_CC}"
else
  gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:].')
fi

runtime_image_fingerprint="${K3_IMAGE_AMD64_DIGEST}"
if [[ -n "${K3_IMAGE_SQSH:-}" ]]; then
  sqsh_sidecar="${K3_IMAGE_SQSH}.sha256"
  if [[ ! -r "${sqsh_sidecar}" ]]; then
    echo "FATAL cached image SHA256 sidecar is missing: ${sqsh_sidecar}" >&2
    exit 2
  fi
  read -r sqsh_sha sqsh_recorded_path <"${sqsh_sidecar}"
  if [[ ! "${sqsh_sha}" =~ ^[0-9a-f]{64}$ || "${sqsh_recorded_path}" != "${K3_IMAGE_SQSH}" ]]; then
    echo "FATAL cached image SHA256 sidecar mismatch: ${sqsh_sidecar}" >&2
    exit 2
  fi
  runtime_image_fingerprint="sqsh:${sqsh_sha}:${K3_IMAGE_SQSH}"
fi

fingerprint_input="$(
  printf '%s\n' \
    "${runtime_image_fingerprint}" \
    "${k3_source_sha}" \
    "${MLITE_SOURCE_SHA}" \
    "${TE_SOURCE_SHA}" \
    "${VERL_SOURCE_SHA}" \
    "${python_version}" \
    "${torch_version}" \
    "${te_version}" \
    "${fla_version}" \
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
export PYTHONPYCACHEPREFIX="${cache_dir}/pycache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
echo "K3_JIT_CACHE_OK fingerprint=${fingerprint} root=${cache_dir}" >&2
