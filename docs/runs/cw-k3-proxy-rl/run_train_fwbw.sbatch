#!/usr/bin/env bash
# Training-only K3 proxy initialization and one real BF16 forward/backward at EP2.
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --job-name=k3-train-fwbw
#SBATCH --output=k3-train-fwbw-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${MODEL_PATH:?set MODEL_PATH to the 12-layer, 56-expert proxy checkpoint}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new shared output directory}"
mkdir -p "${OUTPUT_ROOT}"

diagnose() {
  echo "K3_TRAIN_DIAG phase=$1 job=${SLURM_JOB_ID} time=$(date -Is)"
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
  --no-container-entrypoint \
  --container-mounts=/lustre:/lustre \
  --export=MODEL_PATH,OUTPUT_ROOT,K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "'"${recipe_dir}"'/k3_training_env.sh"
/usr/bin/python3 -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  "${K3_ROOT}/tools/run_proxy_stage.py" \
  --stage fwbw \
  --model-path "${MODEL_PATH}" \
  --trainable-parameter layers.0.input_layernorm.weight \
  --result-path "${OUTPUT_ROOT}/k3-train-fwbw-${SLURM_JOB_ID}.json" \
  --success-marker K3_TRAIN_FWDBWD_OK
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
