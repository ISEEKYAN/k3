#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=k3-overlay-contract
#SBATCH --output=k3-overlay-contract-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

srun \
  --container-image="${K3_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export K3_GPU_CC="${K3_GPU_CC:-90}"
source "'"${recipe_dir}"'/k3_training_env.sh"
/usr/bin/python3 - <<'"'"'PY'"'"'
import importlib.metadata
import json

import fla
import megatron.core
import megatron.lite
import mlite_k3
import torch
import transformer_engine
import transformer_engine.pytorch as te
import verl
import vllm
from megatron.lite.model.registry import (
    get_model_package,
    resolve_runtime_model_name,
)

assert get_model_package("k3").__name__ == "mlite_k3"
assert resolve_runtime_model_name("k3", "lite") == "k3"
result = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformer_engine": importlib.metadata.version("transformer-engine"),
    "fla": importlib.metadata.version("flash-linear-attention"),
    "megatron_core": megatron.core.__file__,
    "megatron_lite": megatron.lite.__file__,
    "mlite_k3": mlite_k3.__file__,
    "verl": verl.__file__,
    "te_rmsnorm": te.RMSNorm.__name__,
}
print("K3_TRAINING_OVERLAY_OK", json.dumps(result, sort_keys=True), flush=True)
PY
'
