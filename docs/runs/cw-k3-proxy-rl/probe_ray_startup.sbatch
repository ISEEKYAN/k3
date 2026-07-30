#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=k3-ray-startup
#SBATCH --output=k3-ray-startup-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a shared output directory}"

srun \
  --account=coreai_devtech_all \
  --container-image="${K3_TRAINING_IMAGE}" \
  --no-container-entrypoint \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "'"${recipe_dir}"'/k3_training_env.sh"

export RAY_TMPDIR="/tmp/k3-ray-${SLURM_JOB_ID}"
ray_log_archive="${OUTPUT_ROOT}/ray-logs-${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
archive_ray_logs() {
  if [[ -d "${RAY_TMPDIR}" ]]; then
    mkdir -p "${ray_log_archive}"
    cp -a "${RAY_TMPDIR}/." "${ray_log_archive}/"
  fi
}
trap archive_ray_logs EXIT

python3 - <<'"'"'PY'"'"'
import os

import ray

context = ray.init(
    num_cpus=1,
    include_dashboard=False,
    _temp_dir=os.environ["RAY_TMPDIR"],
)
print(
    "K3_RAY_STARTUP_OK"
    f" address={context.address_info['address']}"
    f" node={context.address_info['node_id']}",
    flush=True,
)
ray.shutdown()
PY
'
