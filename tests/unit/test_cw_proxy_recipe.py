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


def test_training_environment_preserves_three_pollution_boundaries():
    env = read("k3_training_env.sh")

    assert 'PATH="/usr/local/cuda/bin:${PATH}"' in env
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
    assert "K3_IMAGE_AMD64_DIGEST" in env
    assert "k3_source_sha" in env
    assert "gpu_cc" in env
    assert "TRITON_CACHE_DIR" in env
    assert "TORCHINDUCTOR_CACHE_DIR" in env
    assert "FATAL JIT cache fingerprint mismatch" in env


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
    assert "sleep 120" in runner
    assert "sleep 180" in runner


def test_proxy_checkpoint_build_is_a_zero_gpu_srun():
    builder = read("build_proxy_checkpoint.sbatch")

    assert "#SBATCH --partition=cpu_short" in builder
    assert "--gpus" not in builder
    assert "srun" in builder
    assert "--layers 12" in builder
    assert "--experts 56" in builder
