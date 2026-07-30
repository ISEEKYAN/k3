from pathlib import Path


ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "docs/runs/cw-k3-proxy-rl"


def read(name: str) -> str:
    return (RECIPE / name).read_text()


def test_image_contract_uses_x86_multiarch_k3_release():
    image = read("image.env")

    assert "K3_REQUESTED_IMAGE=docker://aosheninferact/vllm-openai:kimi-k3" in image
    assert "K3_IMAGE=docker://vllm/vllm-openai:kimi-k3" in image
    assert "K3_IMAGE_INDEX_DIGEST=sha256:" in image
    assert "K3_IMAGE_AMD64_DIGEST=sha256:" in image
    assert "MLITE_SOURCE_SHA=85eacfbc1" in image
    assert "K3_RUNTIME_IMAGE" in image


def test_image_cache_recipe_saves_once_and_runtime_recipes_reuse_it():
    cache = read("cache_image.sbatch")

    assert "#SBATCH --partition=batch_short" in cache
    assert "#SBATCH --gres=gpu:1" in cache
    assert "#SBATCH --mem=512G" in cache
    assert '--container-image="${K3_IMAGE}"' in cache
    assert '--container-save="${K3_IMAGE_SQSH}"' in cache
    for name in (
        "audit_te_build_env.sbatch",
        "build_proxy_checkpoint.sbatch",
        "build_te_overlay.sbatch",
        "run_proxy_qat_r3.sbatch",
    ):
        assert '--container-image="${K3_RUNTIME_IMAGE}"' in read(name)
    assert '--container-image="${K3_TRAINING_IMAGE}"' in read("run_proxy_stage.sbatch")
    assert '--container-image="${K3_TRAINING_IMAGE}"' in read(
        "validate_training_overlay.sbatch"
    )


def test_training_environment_preserves_three_pollution_boundaries():
    env = read("k3_training_env.sh")

    assert 'PATH="${CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin"' in env
    assert (
        'LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
        in env
    )
    assert 'CC="${CC:-/usr/bin/gcc}"' in env
    assert "PYTHONNOUSERSITE=1" in env
    assert "OMP_NUM_THREADS=1" in env


def test_jit_cache_is_persistent_keyed_and_fail_loud():
    env = read("k3_training_env.sh")

    assert "SLURM_JOB_ID" not in env
    assert "K3_TRAINING_IMAGE" in env
    assert "training_image_stat" in env
    assert "k3_source_sha" in env
    assert "gpu_cc" in env
    assert "TRITON_CACHE_DIR" in env
    assert "TORCHINDUCTOR_CACHE_DIR" in env
    assert "FATAL JIT cache fingerprint mismatch" in env
    assert "runtime_image_fingerprint" in env
    assert "TILELANG_CACHE_DIR" in env
    assert "TILELANG_TMP_DIR" in env
    assert "gpu_cc_output=" in env
    assert "head -1" not in env


def test_overlay_validation_is_inside_srun():
    sbatch = read("validate_training_overlay.sbatch")
    validation = read("validate_training_overlay.py")

    assert "#SBATCH --partition=cpu_short" in sbatch
    assert "srun" in sbatch
    assert "validate_training_overlay.py" in sbatch
    assert "transformer_engine.pytorch" in validation
    assert "import fla" in validation
    assert "import megatron.lite" in validation
    assert "import mlite_k3" in validation
    assert "import verl" in validation
    assert 'import_module("verl.trainer.main_ppo")' in validation
    assert 'import_module("verl_mlite.engine.mlite_engine")' in validation


def test_te_binary_probe_is_short_cpu_srun_and_never_builds_source():
    probe = read("probe_te_binary.sbatch")

    assert "#SBATCH --account=coreai_devtech_all" in probe
    assert "#SBATCH --partition=cpu_short" in probe
    assert "#SBATCH --time=00:05:00" in probe
    assert "srun" in probe
    assert "--account=coreai_devtech_all" in probe
    assert "transformer-engine-cu13==2.17.0" in probe
    assert "transformer-engine-torch==2.17.0" in probe
    assert "--only-binary=:all:" in probe
    assert "pip install" not in probe


def test_training_base_probe_reuses_proven_pytorch_and_rollout_overlays():
    probe = read("probe_training_base.sbatch")

    assert "#SBATCH --partition=cpu_short" in probe
    assert "#SBATCH --time=00:05:00" in probe
    assert "--account=coreai_devtech_all" in probe
    assert "pytorch_26.04-py3.sqsh" in probe
    assert "mlite-2604-verl-dsa-sm90-overlay" in probe
    assert "mlite-2612-cu13-canonical/vllm0251-site" in probe
    assert "mlite-newenv-cache/qwen35-proven-canary-site" in probe
    assert "ds4-csacp-parity-eaa5b486d/mcore" in probe
    assert "OMP_NUM_THREADS=1" in probe
    assert "TILELANG_CACHE_DIR" in probe
    assert "TILELANG_TMP_DIR" in probe
    assert "pip install" not in probe


def test_te_audit_uses_node_local_dependencies_and_reports_missing_git():
    sbatch = read("audit_te_build_env.sbatch")
    audit = read("audit_te_build_env.py")
    build = read("build_te_overlay.sbatch")
    requirements = read("te-build-requirements.txt")

    assert "SLURM_TMPDIR" in sbatch
    assert "CUDA_HOME=/usr/local/cuda-13.0" in sbatch
    assert "TE_SUBMODULE_STATUS" in sbatch
    assert 'command("git", "--version")' in audit
    assert '"required": False' in audit
    assert '"tmp_disk"' in audit
    assert '"cuda_header_smoke"' in audit
    assert '"output_tail"' in audit
    assert "FileNotFoundError" not in audit
    assert "nvidia-cuda-nvcc" not in requirements
    assert "nvidia-cudnn-frontend==1.26.0" in requirements
    assert "nvidia-cuda-profiler-api==13.0.85" in requirements
    assert "nvidia-nvml-dev==13.0.87" in requirements
    assert "CUDA_HOME=/usr/local/cuda-13.0" in build
    assert "nvidia/cudnn" in build
    assert "nvidia/nccl" in build
    assert "nvidia/cu13" in build
    assert "base_cudnn_root}/include" in build
    assert '#include "nvtx.h"' in audit
    assert '#include "util/logging.h"' in audit
    assert "#include <cuda_profiler_api.h>" in audit
    assert "#include <nvml.h>" in audit
    assert '"te_source_cuda_headers"' in audit
    assert "PIP_CACHE_DIR" in build
    assert "TMPDIR" in build
    assert "--no-cache-dir" in build
    assert "te_build_source" in build
    assert "--exclude=./build" in build
    assert '"${te_build_source}"' in build


def test_gpu_recipe_is_one_interactive_node_with_qat_r3_and_wandb():
    runner = read("run_proxy_qat_r3.sbatch")

    assert "#SBATCH --partition=interactive" in runner
    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --gpus-per-node=8" in runner
    assert 'LOGGER="[console,file,wandb]"' in runner
    assert "megatron-core-moe-dev" in runner
    assert "impl_cfg.qat.enabled=true" in runner
    assert "impl_cfg.qat.format=mxfp4" in runner
    assert "router_replay_mode=R3" in runner
    assert "enable_rollout_routing_replay=True" in runner
    assert "impl_cfg.moe_router_fusion=false" in runner
    assert "sleep 120" in runner
    assert "sleep 180" in runner


def test_proxy_checkpoint_build_is_a_zero_gpu_srun():
    builder = read("build_proxy_checkpoint.sbatch")

    assert "#SBATCH --partition=cpu_short" in builder
    assert "--gpus" not in builder
    assert "srun" in builder
    assert "--layers 12" in builder
    assert "--experts 56" in builder


def test_fail_local_gpu_carrier_has_four_ordered_stages():
    carrier = read("run_proxy_stage.sbatch")

    assert "#SBATCH --partition=interactive" in carrier
    assert "#SBATCH --nodes=1" in carrier
    assert "#SBATCH --gpus-per-node=8" in carrier
    assert "import | construct | fwbw | qat" in carrier
    assert "validate_training_overlay.py" in carrier
    assert "run_proxy_stage.py" in carrier
    assert "OMP_NUM_THREADS" in read("k3_training_env.sh")
