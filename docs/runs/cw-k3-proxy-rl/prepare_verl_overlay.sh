#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${VERL_BASE_ROOT:?set VERL_BASE_ROOT to the pinned clean VERL checkout}"
: "${K3_VERL_ROOT:=${K3_VERL_OVERLAY}}"
: "${VERL_SOURCE_SHA:?set VERL_SOURCE_SHA to the pinned VERL commit}"

patch_file="${recipe_dir}/verl-mxfp4-layerwise-reload.patch"
expected_changed="verl/workers/rollout/vllm_rollout/utils.py"

if [[ ! -d "${K3_VERL_ROOT}/.git" ]]; then
  git clone --no-hardlinks "${VERL_BASE_ROOT}" "${K3_VERL_ROOT}"
fi

actual=$(git -C "${K3_VERL_ROOT}" rev-parse HEAD)
if [[ "${actual}" != "${VERL_SOURCE_SHA}" ]]; then
  echo "FATAL VERL overlay source mismatch expected=${VERL_SOURCE_SHA} actual=${actual}" >&2
  exit 2
fi

if git -C "${K3_VERL_ROOT}" apply --check "${patch_file}"; then
  git -C "${K3_VERL_ROOT}" apply "${patch_file}"
elif ! git -C "${K3_VERL_ROOT}" apply --reverse --check "${patch_file}"; then
  echo "FATAL VERL overlay is neither clean nor exactly patched: ${patch_file}" >&2
  exit 2
fi

changed=$(git -C "${K3_VERL_ROOT}" diff --name-only)
if [[ "${changed}" != "${expected_changed}" ]]; then
  echo "FATAL unexpected VERL overlay changes: ${changed}" >&2
  exit 2
fi
echo "K3_VERL_OVERLAY_OK root=${K3_VERL_ROOT} source=${actual}"
