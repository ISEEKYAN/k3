#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --job-name=k3-ep8-dec-bwd
#SBATCH --output=k3-ep8-dec-bwd-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
source "${recipe_dir}/image.env"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DECODER_LAYERS:=4}"
: "${TOKENS:=3072}"
: "${NUM_SEQS:=16}"
: "${ATTENTION_BACKEND_OVERRIDE:=auto}"
if [[ ! "${DECODER_LAYERS}" =~ ^[1-4]$ ]]; then
  echo "FATAL DECODER_LAYERS must be between 1 and 4; got ${DECODER_LAYERS}" >&2
  exit 64
fi
if [[ ! "${TOKENS}" =~ ^[0-9]+$ ]] || ((TOKENS < 1 || TOKENS > 4096)); then
  echo "FATAL TOKENS must be between 1 and 4096; got ${TOKENS}" >&2
  exit 64
fi
if [[ ! "${NUM_SEQS}" =~ ^[0-9]+$ ]] || ((NUM_SEQS < 1 || TOKENS % NUM_SEQS)); then
  echo "FATAL NUM_SEQS must be positive and divide TOKENS; got NUM_SEQS=${NUM_SEQS}, TOKENS=${TOKENS}" >&2
  exit 64
fi

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
  --export=MODEL_PATH,DECODER_LAYERS,TOKENS,NUM_SEQS,ATTENTION_BACKEND_OVERRIDE,K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
source "'"${recipe_dir}"'/k3_training_env.sh"
/usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  "'"${recipe_dir}"'/probe_ep8_decoder_backward.py"
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
