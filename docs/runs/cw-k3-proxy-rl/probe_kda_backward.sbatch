#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --job-name=k3-kda-bwd-probe
#SBATCH --output=k3-kda-bwd-probe-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

diagnose() {
  echo "K3_GPU_DIAG component=kda phase=$1 job=${SLURM_JOB_ID} time=$(date -Is)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
  ps -C python3 -o pid,ppid,stat,etime,cmd
}
(
  sleep 120
  diagnose t_plus_2m
  sleep 180
  diagnose t_plus_5m
) &
diag_pid=$!

set +e
srun \
  --container-image="${K3_TRAINING_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "'"${recipe_dir}"'/k3_training_env.sh"
/usr/bin/python3 "'"${recipe_dir}"'/probe_kda_backward.py"
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
