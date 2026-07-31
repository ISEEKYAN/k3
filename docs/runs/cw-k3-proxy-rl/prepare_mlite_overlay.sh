#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${MLITE_BASE_ROOT:?set MLITE_BASE_ROOT to the pinned clean MLite checkout}"
: "${K3_MLITE_ROOT:=${K3_MLITE_OVERLAY}}"
: "${MLITE_SOURCE_SHA:?set MLITE_SOURCE_SHA to the pinned MLite commit}"

patch_file="${recipe_dir}/mlite-router-replay-model-route-selection.patch"
expected_changed=$'experimental/lite/megatron/lite/primitive/optimizers/megatron_wrap.py\nexperimental/lite/megatron/lite/runtime/backends/mlite/router_replay.py'

if [[ ! -d "${K3_MLITE_ROOT}/.git" ]]; then
  git clone --no-hardlinks "${MLITE_BASE_ROOT}" "${K3_MLITE_ROOT}"
fi

actual=$(git -C "${K3_MLITE_ROOT}" rev-parse HEAD)
if [[ "${actual}" != "${MLITE_SOURCE_SHA}" ]]; then
  echo "FATAL MLite overlay source mismatch expected=${MLITE_SOURCE_SHA} actual=${actual}" >&2
  exit 2
fi

if git -C "${K3_MLITE_ROOT}" apply --check "${patch_file}"; then
  git -C "${K3_MLITE_ROOT}" apply "${patch_file}"
elif ! git -C "${K3_MLITE_ROOT}" apply --reverse --check "${patch_file}"; then
  echo "FATAL MLite overlay is neither clean nor exactly patched: ${patch_file}" >&2
  exit 2
fi

changed=$(git -C "${K3_MLITE_ROOT}" diff --name-only)
if [[ "${changed}" != "${expected_changed}" ]]; then
  echo "FATAL unexpected MLite overlay changes: ${changed}" >&2
  exit 2
fi
echo "K3_MLITE_OVERLAY_OK root=${K3_MLITE_ROOT} source=${actual}"
