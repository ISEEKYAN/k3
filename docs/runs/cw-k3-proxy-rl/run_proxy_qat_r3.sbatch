#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=k3-proxy-qat-r3
#SBATCH --output=k3-proxy-qat-r3-%j.out

set -euo pipefail
recipe_dir="${K3_RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -L)}}"
# shellcheck source=image.env
source "${recipe_dir}/image.env"

: "${MODEL_PATH:?set MODEL_PATH to the 12-layer, 56-expert proxy checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to an RL parquet dataset}"
: "${VAL_FILES:?set VAL_FILES to an RL parquet validation dataset}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a shared output directory}"

diagnose() {
  local phase=$1
  echo "K3_GPU_DIAG phase=${phase} job=${SLURM_JOB_ID} time=$(date -Is)"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv
  ps -eo pid,ppid,stat,etime,cmd --forest
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
  --export=ALL \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "'"${recipe_dir}"'/k3_training_env.sh"

export WANDB_ENTITY="${WANDB_ENTITY:-megatron-core-moe-dev}"
export WANDB_PROJECT="${WANDB_PROJECT:-k3-proxy-qat-r3}"
export PROJECT_NAME="${WANDB_PROJECT}"
export RUN_NAME="${RUN_NAME:-k3-proxy-12l-56e-qat-r3}"
export LOGGER="[console,file,wandb]"
export NNODES=1
export NGPUS_PER_NODE=8
export MLITE_MODEL_NAME=k3
export MLITE_IMPL=lite
export MLITE_OPTIMIZER_BACKEND=dist_opt
export ACTOR_TP=1
export ACTOR_EP=8
export ACTOR_ETP=1
export ACTOR_PP=1
export ACTOR_CP=1
export OPTIMIZER_OFFLOAD=True
export PARAM_OFFLOAD=False
export GRAD_OFFLOAD=False
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-128}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-512}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_memory_monitor_refresh_ms=0
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

log_file="${OUTPUT_ROOT}/${RUN_NAME}.launcher.log"
mkdir -p "${OUTPUT_ROOT}"
(
  deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if [[ -f "${log_file}" ]]; then
      url=$(grep -Eo "https://wandb.ai/[^[:space:]]+/runs/[^[:space:]]+" "${log_file}" | head -1 || true)
      if [[ -n "${url}" ]]; then
        echo "K3_WANDB_URL ${url}" | tee "${OUTPUT_ROOT}/wandb-url.txt"
        exit 0
      fi
    fi
    sleep 5
  done
  echo "FATAL W&B run URL was not observed within 15 minutes" >&2
  exit 2
) &
wandb_watch_pid=$!

set +e
bash "${MLITE_ROOT}/experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh" \
  "+actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true" \
  "+actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4" \
  "+actor_rollout_ref.actor.engine.impl_cfg.qat.group_size=32" \
  "+actor_rollout_ref.actor.engine.impl_cfg.moe_router_fusion=false" \
  "++actor_rollout_ref.actor.engine.router_replay_mode=R3" \
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=True" \
  2>&1 | tee "${log_file}"
run_rc=${PIPESTATUS[0]}
set -e

if (( run_rc != 0 )); then
  kill "${wandb_watch_pid}" 2>/dev/null || true
  wait "${wandb_watch_pid}" 2>/dev/null || true
  exit "${run_rc}"
fi
if ! wait "${wandb_watch_pid}"; then
  exit 2
fi
exit "${run_rc}"
'
run_rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
exit "${run_rc}"
