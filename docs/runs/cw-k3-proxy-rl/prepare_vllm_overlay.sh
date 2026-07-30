#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_VLLM_SITE:?image.env must define the K3 vLLM site}"
patch_file="${recipe_dir}/vllm-k3-routed-stream-threshold.patch"

if patch --dry-run --silent -p1 -d "${K3_VLLM_SITE}" <"${patch_file}"; then
  patch --silent -p1 -d "${K3_VLLM_SITE}" <"${patch_file}"
elif ! patch --dry-run --silent --reverse -p1 -d "${K3_VLLM_SITE}" <"${patch_file}"; then
  echo "FATAL K3 vLLM overlay is neither clean nor exactly patched: ${patch_file}" >&2
  exit 2
fi

python3 -m py_compile \
  "${K3_VLLM_SITE}/vllm/models/kimi_k3/nvidia/model.py"
echo "K3_VLLM_OVERLAY_READY"
