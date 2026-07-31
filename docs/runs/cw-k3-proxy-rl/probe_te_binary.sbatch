#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=k3-probe-te-wheel
#SBATCH --output=k3-probe-te-wheel-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

srun \
  --account=coreai_devtech_all \
  --container-image="${K3_RUNTIME_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
probe_tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/k3-te-wheel-probe.XXXXXX")
export PIP_CACHE_DIR="${probe_tmp}/pip-cache"
mkdir -p "${probe_tmp}/wheels"

/usr/bin/python3 "'"${recipe_dir}"'/probe_te_binary.py"
if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  /usr/local/cuda-13.0/bin/nvcc --version
else
  echo "K3_NVCC absent"
fi

set +e
/usr/bin/python3 -m pip download \
  --only-binary=:all: \
  --no-deps \
  --dest "${probe_tmp}/wheels" \
  transformer-engine-cu13==2.17.0
core_rc=$?
/usr/bin/python3 -m pip download \
  --only-binary=:all: \
  --no-deps \
  --dest "${probe_tmp}/wheels" \
  transformer-engine-torch==2.17.0
torch_rc=$?
set -e

find "${probe_tmp}/wheels" -maxdepth 1 -type f -name "*.whl" -printf "%f\n" |
  sort
echo "K3_TE_WHEEL_PROBE core_cu13_rc=${core_rc} torch_binding_rc=${torch_rc}"
if [[ "${core_rc}" -ne 0 ]]; then
  echo "FATAL no compatible precompiled CUDA 13 TE core wheel" >&2
  exit 2
fi
'
