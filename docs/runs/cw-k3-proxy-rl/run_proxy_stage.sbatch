#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=k3-proxy-stage
#SBATCH --output=k3-proxy-stage-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${STAGE:?set STAGE to import, construct, fwbw, or qat}"
: "${MODEL_PATH:?set MODEL_PATH to the 12-layer, 56-expert proxy checkpoint}"

case "${STAGE}" in
  import | construct | fwbw | qat) ;;
  *)
    echo "FATAL unsupported STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

diagnose() {
  echo "K3_GPU_DIAG stage=${STAGE} phase=$1 job=${SLURM_JOB_ID} time=$(date -Is)"
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
  --account=coreai_devtech_all \
  --container-image="${K3_TRAINING_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=STAGE,MODEL_PATH,K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "'"${recipe_dir}"'/k3_training_env.sh"
if [[ "${STAGE}" == import ]]; then
  /usr/bin/python3 "'"${recipe_dir}"'/validate_training_overlay.py"
else
  /usr/bin/python3 -m torch.distributed.run \
    --standalone \
    --nproc-per-node=8 \
    "${K3_ROOT}/tools/run_proxy_stage.py" \
    --stage "${STAGE}" \
    --model-path "${MODEL_PATH}"
fi
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
