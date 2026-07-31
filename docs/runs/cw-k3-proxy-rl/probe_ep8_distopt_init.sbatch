#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --job-name=k3-ep8-distopt-init
#SBATCH --output=k3-ep8-distopt-init-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
source "${recipe_dir}/image.env"
: "${MODEL_PATH:?set MODEL_PATH}"

(
  sleep 120
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
  sleep 180
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
) &
diag_pid=$!

set +e
srun \
  --container-image="${K3_TRAINING_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --export=MODEL_PATH,K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
source "'"${recipe_dir}"'/k3_training_env.sh"
/usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  "'"${recipe_dir}"'/probe_ep8_distopt_init.py"
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
