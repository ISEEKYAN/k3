#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
source "${recipe_dir}/image.env"

: "${MCORE_BASE_ROOT:?set MCORE_BASE_ROOT to the pinned clean MCore checkout}"
: "${K3_MEGATRON_ROOT:?image.env must define the K3-only MCore overlay}"

patch_files=(
  "${recipe_dir}/mcore-nvrx-capability.patch"
  "${recipe_dir}/mcore-fp32-hybrid-leaf.patch"
)
expected_changed=$'megatron/core/dist_checkpointing/strategies/nvrx.py\nmegatron/core/optimizer/distrib_optimizer.py'

if [[ ! -d "${K3_MEGATRON_ROOT}/.git" ]]; then
  git clone --no-hardlinks "${MCORE_BASE_ROOT}" "${K3_MEGATRON_ROOT}"
fi

actual=$(git -C "${K3_MEGATRON_ROOT}" rev-parse HEAD)
if [[ "${actual}" != "${MCORE_SOURCE_SHA}" ]]; then
  echo "FATAL MCore overlay source mismatch expected=${MCORE_SOURCE_SHA} actual=${actual}" >&2
  exit 2
fi

for patch_file in "${patch_files[@]}"; do
  if git -C "${K3_MEGATRON_ROOT}" apply --check "${patch_file}"; then
    git -C "${K3_MEGATRON_ROOT}" apply "${patch_file}"
  elif ! git -C "${K3_MEGATRON_ROOT}" apply --reverse --check "${patch_file}"; then
    echo "FATAL MCore overlay is neither clean nor exactly patched: ${patch_file}" >&2
    exit 2
  fi
done

changed=$(git -C "${K3_MEGATRON_ROOT}" diff --name-only)
if [[ "${changed}" != "${expected_changed}" ]]; then
  echo "FATAL unexpected MCore overlay changes: ${changed}" >&2
  exit 2
fi
echo "K3_MCORE_OVERLAY_OK root=${K3_MEGATRON_ROOT} source=${actual}"
