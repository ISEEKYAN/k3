#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=k3-build-proxy
#SBATCH --output=k3-build-proxy-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_ROOT:?set K3_ROOT to the K3 delivery checkout}"
: "${K3_CHECKPOINT:?set K3_CHECKPOINT to the public Kimi K3 checkpoint}"
: "${K3_PROXY_CHECKPOINT:?set K3_PROXY_CHECKPOINT to a new output directory}"

if [[ -e "${K3_PROXY_CHECKPOINT}" ]]; then
  echo "FATAL K3_PROXY_CHECKPOINT already exists: ${K3_PROXY_CHECKPOINT}" >&2
  exit 2
fi

srun \
  --container-image="${K3_RUNTIME_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
/usr/bin/python3 "'"${K3_ROOT}"'/tools/build_proxy_checkpoint.py" \
  --source "${K3_CHECKPOINT}" \
  --output "${K3_PROXY_CHECKPOINT}" \
  --layers 12 \
  --experts 56
'
