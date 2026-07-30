#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=k3-cache-image
#SBATCH --output=k3-cache-image-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${K3_IMAGE_SQSH:?set K3_IMAGE_SQSH to a new shared .sqsh path}"
if [[ -e "${K3_IMAGE_SQSH}" || -e "${K3_IMAGE_SQSH}.sha256" ]]; then
  echo "FATAL cached image output already exists: ${K3_IMAGE_SQSH}" >&2
  exit 2
fi
mkdir -p "$(dirname "${K3_IMAGE_SQSH}")"

srun \
  --container-image="${K3_IMAGE}" \
  --container-save="${K3_IMAGE_SQSH}" \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
uname -m
/usr/bin/python3 -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
'
test -s "${K3_IMAGE_SQSH}"
sha256sum "${K3_IMAGE_SQSH}" >"${K3_IMAGE_SQSH}.sha256"
echo "K3_IMAGE_CACHE_OK path=${K3_IMAGE_SQSH}"
