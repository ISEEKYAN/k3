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
cleanup_ray() {
  env PYTHONPATH="${VLLM_SITE}" MLITE_K3_AUTO_REGISTER=0 \
    python3 -m ray.scripts.scripts stop --force || true
  archive_ray_logs
}
trap cleanup_ray EXIT
env PYTHONPATH="${VLLM_SITE}" MLITE_K3_AUTO_REGISTER=0 \
  python3 -m ray.scripts.scripts start --head \
    --num-cpus=1 \
    --disable-usage-stats \
    --temp-dir="${RAY_TMPDIR}"
export RAY_ADDRESS=auto

python3 - <<'"'"'PY'"'"'
import ray

context = ray.init(address="auto")
print(
    "K3_RAY_STARTUP_OK"
    f" address={context.address_info['address']}"
    f" node={context.address_info['node_id']}",
    flush=True,
)
ray.shutdown()
PY
'
