#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_VLLM_SITE:?image.env must define the K3 vLLM site}"
: "${K3_VLLM_OVERLAY_SOURCE:?image.env must define the source vLLM overlay}"
patch_files=(
  "${recipe_dir}/vllm-k3-routed-stream-threshold.patch"
  "${recipe_dir}/vllm-moe-sum-abi.patch"
  "${recipe_dir}/vllm-routed-expert-topk-alias.patch"
  "${recipe_dir}/vllm-k3-warmup-import.patch"
  "${recipe_dir}/vllm-optional-router-warmup.patch"
)

if [[ ! -e "${K3_VLLM_OVERLAY}" ]]; then
  cp -a "${K3_VLLM_OVERLAY_SOURCE}" "${K3_VLLM_OVERLAY}"
fi
if [[ ! -d "${K3_VLLM_SITE}/vllm" ]]; then
  echo "FATAL private K3 vLLM overlay is incomplete: ${K3_VLLM_SITE}" >&2
  exit 2
fi

for patch_file in "${patch_files[@]}"; do
  if patch --batch --forward --dry-run --silent -p1 -d "${K3_VLLM_SITE}" <"${patch_file}"; then
    patch --batch --forward --silent -p1 -d "${K3_VLLM_SITE}" <"${patch_file}"
  elif ! patch --batch --reverse --force --dry-run --silent -p1 \
    -d "${K3_VLLM_SITE}" <"${patch_file}"; then
    echo "FATAL K3 vLLM overlay is neither clean nor exactly patched: ${patch_file}" >&2
    exit 2
  fi
done

python3 -m py_compile \
  "${K3_VLLM_SITE}/vllm/_custom_ops.py" \
  "${K3_VLLM_SITE}/vllm/models/kimi_k3/nvidia/model.py" \
  "${K3_VLLM_SITE}/vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py" \
  "${K3_VLLM_SITE}/vllm/model_executor/warmup/kernel_warmup.py" \
  "${recipe_dir}/k3_vllm_warmup.py"
echo "K3_VLLM_OVERLAY_READY"
