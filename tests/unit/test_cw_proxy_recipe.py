from pathlib import Path


ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "docs/runs/cw-k3-proxy-rl"


def read(name: str) -> str:
    return (RECIPE / name).read_text()


def test_image_contract_reuses_the_proven_k3_vllm_overlay():
    image = read("image.env")

    assert "K3_TRAINING_IMAGE=" in image
    assert "pytorch_26.04-py3.sqsh" in image
    assert "K3_VLLM_OVERLAY_SOURCE=" in image
    assert "K3_VLLM_OVERLAY=" in image
    assert "k3-vllm-main-prs-overlay" in image
    assert "k3-vllm-overlay-r1" in image
    assert 'K3_VLLM_SITE="${K3_VLLM_OVERLAY}/lib/python3.12/site-packages"' in image
    assert "qwen35-cp-overlay-20260613/site" in image
    assert "nvidia_cutlass_dsl/python_packages" in image
    assert "mcore-fp32-hybrid-leaf" in image
    assert "MLITE_SOURCE_SHA=cc4efe6a1" in image


def test_active_runtime_recipes_reuse_the_proven_training_base():
    for name in (
        "run_proxy_generate.sbatch",
        "run_proxy_qat_r3.sbatch",
        "run_proxy_stage.sbatch",
    ):
        assert '--container-image="${K3_TRAINING_IMAGE}"' in read(name)
    assert '--container-image="${K3_TRAINING_IMAGE}"' in read("run_proxy_stage.sbatch")
    assert '--container-image="${K3_TRAINING_IMAGE}"' in read(
        "validate_training_overlay.sbatch"
    )


def test_qat_recipe_disables_cudagraph_capture_while_using_k3_aux_streams():
    recipe = read("run_proxy_qat_r3.sbatch")

    assert (
        'export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"'
        in recipe
    )
    assert (
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.'
        'cudagraph_mode=NONE'
    ) in recipe


def test_training_environment_preserves_three_pollution_boundaries():
    env = read("k3_training_env.sh")

    assert 'PATH="/usr/local/bin:/usr/bin:/bin:${CUDA_HOME}/bin"' in env
    assert (
        'LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
        in env
    )
    assert "CC=/usr/bin/gcc" in env
    assert "PYTHONNOUSERSITE=1" in env
    assert "OMP_NUM_THREADS=1" in env
    assert "VLLM_SITE:=${K3_VLLM_SITE}" in env
    assert "TRAINING_VLLM_SITE" not in env
    assert "MLITE_SM90_SITE" not in env
    assert (
        'export PYTHONPATH="${VERL_ROOT}:${VLLM_SITE}:${VERL_PRUNED_SITE}:${FLA_SITE}:'
    ) in env
    assert (
        "${FLA_SITE}:${CUTLASS_DSL_SITE}:${recipe_dir}:${K3_ROOT}/src:"
        "${MEGATRON_ROOT}:"
        "${MLITE_ROOT}/experimental/lite/examples/verl:"
        '${MLITE_ROOT}/experimental/lite:/vllm"'
    ) in env
    assert "CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH" in env
    assert "CC CXX CFLAGS CPPFLAGS CXXFLAGS LDFLAGS" in env


def test_jit_cache_is_persistent_keyed_and_fail_loud():
    env = read("k3_training_env.sh")

    assert "SLURM_JOB_ID" not in env
    assert "K3_TRAINING_IMAGE" in env
    assert "training_image_stat" in env
    assert "recipe_fingerprint" in env
    assert '"${recipe_dir}/mcore-nvrx-capability.patch"' in env
    assert '"${recipe_dir}/mcore-fp32-hybrid-leaf.patch"' in env
    assert "k3_source_sha" in env
    assert "gpu_cc" in env
    assert "TRITON_CACHE_DIR" in env
    assert "TORCHINDUCTOR_CACHE_DIR" in env
    assert "FLASHINFER_WORKSPACE_BASE" in env
    assert "VLLM_CACHE_ROOT" in env
    assert 'HOME="${cache_dir}/home"' in env
    assert 'XDG_CACHE_HOME="${cache_dir}/xdg"' in env
    assert "FATAL JIT cache fingerprint mismatch" in env
    assert "runtime_image_fingerprint" in env
    assert "TILELANG_CACHE_DIR" in env
    assert "TILELANG_TMP_DIR" in env
    assert "gpu_cc_output=" in env
    assert "head -1" not in env


def test_k3_mcore_overlay_detaches_fp32_dist_opt_shards():
    prepare = read("prepare_mcore_overlay.sh")
    patch = read("mcore-fp32-hybrid-leaf.patch")
    nvrx_patch = read("mcore-nvrx-capability.patch")
    env = read("k3_training_env.sh")

    assert "git clone --no-hardlinks" in prepare
    assert 'apply --check "${patch_file}"' in prepare
    assert 'apply --reverse --check "${patch_file}"' in prepare
    assert "diff --name-only" in prepare
    assert "git reset" not in prepare
    assert "model_param.detach().view(-1)" in patch
    assert "model_param.view(-1)" in patch
    assert "if not is_nvrx_min_version():" in nvrx_patch
    assert "return False" in nvrx_patch
    assert 'for mcore_patch in "${mcore_patches[@]}"' in env
    assert 'apply --reverse --check "${mcore_patch}"' in env


def test_k3_vllm_overlay_uses_the_upstream_local_stream_threshold():
    prepare = read("prepare_vllm_overlay.sh")
    patch_text = read("vllm-k3-routed-stream-threshold.patch")
    moe_abi_patch = read("vllm-moe-sum-abi.patch")
    expert_alias_patch = read("vllm-routed-expert-topk-alias.patch")
    warmup_import_patch = read("vllm-k3-warmup-import.patch")
    optional_warmup_patch = read("vllm-optional-router-warmup.patch")
    env = read("k3_training_env.sh")

    assert "_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD = 256" in patch_text
    assert "envs.VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD" in patch_text
    assert "if num_tokens <= _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD" in patch_text
    assert 'cp -a "${K3_VLLM_OVERLAY_SOURCE}" "${K3_VLLM_OVERLAY}"' in prepare
    assert '"${recipe_dir}/vllm-moe-sum-abi.patch"' in prepare
    assert '"${recipe_dir}/vllm-routed-expert-topk-alias.patch"' in prepare
    assert '"${recipe_dir}/vllm-k3-warmup-import.patch"' in prepare
    assert '"${recipe_dir}/vllm-optional-router-warmup.patch"' in prepare
    assert "patch --batch --forward --dry-run --silent" in prepare
    assert "if topk_ids is None and expert_map is None:" in moe_abi_patch
    assert "torch.ops._moe_C.moe_sum(input, output)" in moe_abi_patch
    assert '"num_experts_per_token"' in expert_alias_patch
    assert "num_experts_per_tok," in expert_alias_patch
    assert (
        "from k3_vllm_warmup import kimi_k3_triton_warmup"
        in warmup_import_patch
    )
    assert 'find_spec("quack") is None' in optional_warmup_patch
    assert "Skipping ll_bf16 router GEMM warmup: quack is unavailable." in (
        optional_warmup_patch
    )
    assert 'for vllm_patch in "${vllm_patches[@]}"' in env
    assert '"${recipe_dir}/vllm-routed-expert-topk-alias.patch"' in env
    assert '"${recipe_dir}/vllm-k3-warmup-import.patch"' in env
    assert '"${recipe_dir}/vllm-optional-router-warmup.patch"' in env
    assert "patch --batch --reverse --force --dry-run --silent" in env
    assert 'mcore_changed=$(git -C "${MEGATRON_ROOT}" diff --name-only)' in env


def test_overlay_validation_is_inside_srun():
    sbatch = read("validate_training_overlay.sbatch")
    validation = read("validate_training_overlay.py")

    assert "#SBATCH --partition=cpu_short" in sbatch
    assert "srun" in sbatch
    assert "srun \\\n  --account=coreai_devtech_all" in sbatch
    assert "--no-container-entrypoint" in sbatch
    assert "validate_training_overlay.py" in sbatch
    assert "transformer_engine.pytorch" in validation
    assert '"vllm_file": vllm.__file__' in validation
    assert "kernel_warmup.kimi_k3_triton_warmup is kimi_k3_triton_warmup" in validation
    assert "kernel_warmup._warmup_ll_bf16_router_gemm(object())" in validation
    assert '"transformer_engine_file": transformer_engine.__file__' in validation
    assert "import fla" in validation
    assert '"fla_file": fla.__file__' in validation
    assert '"cutlass_cute_file": cutlass.cute.__file__' in validation
    assert 'fla_utils.device_platform == "cuda"' in validation
    assert "fla_utils.IS_NVIDIA" in validation
    assert "import megatron.lite" in validation
    assert '"megatron_lite": megatron.lite.__file__' in validation
    assert '"megatron_lite_version": os.environ["MLITE_SOURCE_SHA"]' in validation
    assert "import mlite_k3" not in validation
    assert "import verl" not in validation


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


def test_python_dev_header_probe_is_short_cpu_srun():
    probe = read("probe_python_dev_headers.sbatch")

    assert "#SBATCH --account=coreai_devtech_all" in probe
    assert "#SBATCH --partition=cpu_short" in probe
    assert "#SBATCH --time=00:05:00" in probe
    assert '--container-image="${K3_TRAINING_IMAGE}"' in probe
    assert "srun" in probe
    assert "--export=NONE" in probe
    assert "sysconfig.get_paths" in probe
    assert "pyconfig.h" in probe
    assert "Python.h" in probe
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


def test_gpu_recipe_is_one_interactive_node_with_qat_r3_and_optional_wandb():
    runner = read("run_proxy_qat_r3.sbatch")

    assert "#SBATCH --partition=interactive" in runner
    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --gpus-per-node=8" in runner
    assert "srun \\\n  --account=coreai_devtech_all" in runner
    assert "--no-container-entrypoint" in runner
    assert 'export LOGGER="${LOGGER:-[console,file]}"' in runner
    assert 'if [[ "${LOGGER}" == *wandb* ]]; then' in runner
    assert 'if [[ -n "${wandb_watch_pid}" ]]' in runner
    assert "megatron-core-moe-dev" in runner
    assert "++actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true" in runner
    assert "++actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4" in runner
    assert "++actor_rollout_ref.actor.engine.impl_cfg.qat.group_size=32" in runner
    assert "router_replay_mode=R3" in runner
    assert "enable_rollout_routing_replay=True" in runner
    assert "++actor_rollout_ref.actor.engine.impl_cfg.moe_router_fusion=false" in runner
    assert "trainer.use_v1=False" in runner
    assert "data.trust_remote_code=True" in runner
    assert "export PARAM_OFFLOAD=True" in runner
    assert "sleep 120" in runner
    assert "sleep 180" in runner
    assert 'RAY_TMPDIR="/tmp/k3-ray-${SLURM_JOB_ID}"' in runner
    assert "archive_ray_logs" in runner
    assert "ray-logs-${SLURM_JOB_ID}" in runner
    assert 'cp -a "${RAY_TMPDIR}/."' not in runner
    assert '"raylet.*" "gcs_server.*" "runtime_env_agent.*"' in runner
    assert 'PYTHONPATH="${VLLM_SITE}"' in runner
    assert "python3 -m ray.scripts.scripts start --head" in runner
    assert "--num-gpus=8" in runner
    assert "export RAY_ADDRESS=auto" in runner
    assert "unset ROCR_VISIBLE_DEVICES" in runner
    assert runner.index("unset ROCR_VISIBLE_DEVICES") < runner.index(
        "python3 -m ray.scripts.scripts start --head"
    )
    assert "/usr/bin/env -u ROCR_VISIBLE_DEVICES" in runner
    assert "-u PYTHONUSERBASE" in runner
    assert "-u PYTHONHOME" in runner
    assert "-u VIRTUAL_ENV" in runner
    assert "-u CONDA_PREFIX" in runner
    assert "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin" in runner
    assert "PYTHONNOUSERSITE=1" in runner
    assert "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1" in runner
    assert (
        "++ray_kwargs.ray_init.runtime_env.env_vars."
        'RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=\\"1\\"'
    ) in runner
    assert (
        '++ray_kwargs.ray_init.runtime_env.env_vars.MLITE_K3_AUTO_REGISTER=\\"1\\"'
    ) in runner
    assert (
        '++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\\"${PYTHONPATH}\\"'
    ) in runner
    assert 'python3 "${recipe_dir}/assert_ray_cuda_env.py"' in runner
    assert runner.index(
        'python3 "${recipe_dir}/assert_ray_cuda_env.py"'
    ) < runner.index(
        'bash "${MLITE_ROOT}/experimental/lite/examples/verl/scripts/'
        'run_qwen3moe_gsm8k_grpo.sh"'
    )
    assert 'python3 "${recipe_dir}/assert_runtime_package_paths.py"' in runner
    assert 'python3 "${K3_FULL_CLOSURE_PROBE}"' in runner
    assert runner.index('python3 "${K3_FULL_CLOSURE_PROBE}"') < runner.index(
        'python3 "${recipe_dir}/assert_runtime_package_paths.py"'
    )
    assert runner.index('python3 "${recipe_dir}/assert_runtime_package_paths.py"') < (
        runner.index("python3 -m ray.scripts.scripts start --head")
    )


def test_ray_cuda_environment_gate_checks_a_gpu_actor():
    gate = read("assert_ray_cuda_env.py")

    assert "@ray.remote(num_gpus=1)" in gate
    assert '"ROCR_VISIBLE_DEVICES" not in os.environ' in gate
    assert '"CUDA_VISIBLE_DEVICES" in os.environ' in gate
    assert "assert_runtime_package_paths" in gate
    assert '"PYTHONPATH": os.environ["PYTHONPATH"]' in gate
    assert '"MLITE_K3_AUTO_REGISTER": "1"' in gate
    assert '"VERL_PRUNED_SITE": os.environ["VERL_PRUNED_SITE"]' in gate
    assert 'resolve_runtime_model_name("k3", "lite") == "k3"' in gate
    assert "K3_RAY_CUDA_ENV_OK" in gate


def test_container_python_and_pruned_verl_site_fail_loud():
    image = read("image.env")
    env = read("k3_training_env.sh")
    validate = read("assert_runtime_package_paths.py")

    assert "K3_TENSORDICT_SITE=" not in image
    assert "K3_PYVERS_SITE=" not in image
    assert "K3_HYDRA_SITE=" not in image
    assert "K3_VERL_PRUNED_SITE=" in image
    assert "k3-verl-deps-pruned-site" in image
    assert "closure_full.py" in image
    assert (
        'export PYTHONPATH="${VERL_ROOT}:${VLLM_SITE}:${VERL_PRUNED_SITE}:${FLA_SITE}:'
    ) in env
    for package in (
        "huggingface_hub",
        "transformers",
        "vllm",
        "vllm._C",
        "tensordict",
        "ray",
        "pyvers",
        "hydra",
        "codetiming",
        "orjson",
        "accelerate",
        "wandb",
    ):
        assert package in validate
    assert '"omegaconf":' not in validate
    assert '"antlr4":' not in validate
    assert '"0.10.0"' in validate
    assert "nv26.05" in validate
    assert "/usr/local/lib/python3.12/dist-packages" in validate
    assert 'sys.executable != "/usr/bin/python3"' in validate
    assert validate.index("import torch") < validate.index("importlib.import_module")
    assert "K3_CONTAINER_PYTHON_OK" in validate
    assert validate.index("K3_CONTAINER_PYTHON_OK") < validate.index(
        "importlib.import_module"
    )
    assert "pyvers.__version__" not in validate
    assert 'importlib.metadata.version("pyvers")' not in validate
    assert "pyvers_version" not in validate
    assert "K3_RUNTIME_PACKAGE_PATHS_OK" in validate
    assert "os.path.realpath(actual)" in validate
    assert "os.path.realpath(expected_root) + os.sep" in validate
    assert ".is_relative_to(" not in validate
    for package in (
        "tensordict",
        "pyvers",
        "hydra",
        "codetiming",
        "orjson",
        "accelerate",
    ):
        assert f'"{package}": ("{package}", verl_pruned_site / "{package}")' in validate
    readme = read("README.md")
    assert "pyvers 0.2.2" in readme
    assert "pyvers<0.2.0" in readme


def test_ray_startup_probe_is_a_zero_gpu_full_environment_gate():
    probe = read("probe_ray_startup.sbatch")

    assert "#SBATCH --partition=cpu_short" in probe
    assert "--gpus" not in probe
    assert "srun \\\n  --account=coreai_devtech_all" in probe
    assert "--no-container-entrypoint" in probe
    assert "k3_training_env.sh" in probe
    assert 'RAY_TMPDIR="/tmp/k3-ray-${SLURM_JOB_ID}"' in probe
    assert 'PYTHONPATH="${VLLM_SITE}"' in probe
    assert "python3 -m ray.scripts.scripts start --head" in probe
    assert 'ray.init(address="auto")' in probe
    assert "K3_RAY_STARTUP_OK" in probe
    assert "ray-logs-${SLURM_JOB_ID}" in probe


def test_proxy_checkpoint_build_is_a_zero_gpu_srun():
    builder = read("build_proxy_checkpoint.sbatch")

    assert "#SBATCH --partition=cpu_short" in builder
    assert "--gpus" not in builder
    assert "srun \\\n  --account=coreai_devtech_all" in builder
    assert '--container-image="${K3_TRAINING_IMAGE}"' in builder
    assert "--layers 12" in builder
    assert "--experts 56" in builder


def test_fail_local_gpu_carrier_has_four_ordered_stages():
    carrier = read("run_proxy_stage.sbatch")

    assert "#SBATCH --partition=interactive" in carrier
    assert "#SBATCH --nodes=1" in carrier
    assert "#SBATCH --gpus-per-node=8" in carrier
    assert "srun \\\n  --account=coreai_devtech_all" in carrier
    assert "import | construct | fwbw | qat" in carrier
    assert "validate_training_overlay.py" in carrier
    assert "run_proxy_stage.py" in carrier
    assert "OMP_NUM_THREADS" in read("k3_training_env.sh")
    assert "ps -C python3" in carrier
    assert "ps -eo pid,ppid,stat,etime,cmd --forest" not in carrier
    assert "--export=ALL" not in carrier
    assert (
        "--export=STAGE,MODEL_PATH,K3_ROOT,MLITE_ROOT,VERL_ROOT,"
        "VERL_DEPS_SITE,K3_CACHE_ROOT"
    ) in carrier


def test_proxy_stage_emits_fail_local_phase_markers():
    runner = (ROOT / "tools/run_proxy_stage.py").read_text()

    for phase in (
        "dist_ready",
        "build_start",
        "build_done",
        "numel_done",
        "forward_start",
        "forward_done",
        "backward_start",
        "backward_done",
    ):
        assert f'"{phase}"' in runner


def test_proxy_generate_reuses_external_launcher_at_tp8():
    carrier = read("run_proxy_generate.sbatch")
    driver = read("run_proxy_generate.py")

    assert "#SBATCH --partition=interactive" in carrier
    assert "#SBATCH --nodes=1" in carrier
    assert "#SBATCH --ntasks-per-node=8" in carrier
    assert "#SBATCH --gpus-per-node=8" in carrier
    assert "#SBATCH --account=coreai_devtech_all" in carrier
    assert "srun \\\n  --account=coreai_devtech_all" in carrier
    assert '--container-image="${K3_TRAINING_IMAGE}"' in carrier
    assert "--no-container-entrypoint" in carrier
    assert "OMP_NUM_THREADS=1" in carrier
    assert "K3_VLLM_SITE" in carrier
    assert "K3_CACHE_ROOT=" in read("image.env")
    assert 'PYTHONPATH="${K3_VLLM_SITE}:${K3_CUTLASS_DSL_SITE}"' in carrier
    assert "PATH=/cm/shared/apps/slurm/current/bin:" in carrier
    assert "CC=/usr/bin/gcc" in carrier
    assert "CXX=/usr/bin/g++" in carrier
    assert "CPATH=/usr/include/x86_64-linux-gnu" in carrier
    assert "--export=ALL,PATH=${PATH},CC=${CC},CXX=${CXX},CPATH=${CPATH}," in carrier
    assert "unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS" in carrier
    assert "unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH" in carrier
    assert "LD_PRELOAD" not in carrier
    assert "K3_GENERATE_WANDB_URL" in driver
    assert 'assert int(os.environ["WORLD_SIZE"]) == 8' in driver
    assert "tensor_parallel_size=8" in driver
    assert "enable_expert_parallel=False" in driver
    assert 'distributed_executor_backend="external_launcher"' in driver
    assert "skip_tokenizer_init=True" in driver
    assert "TokensPrompt" in driver
    assert "ensure_k3_env_compatibility()" in driver
    assert 'setattr(envs, "VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD", 0)' in driver
    assert "ensure_moe_sum_compatibility()" in driver
    assert "torch.ops._moe_C.moe_sum.default._schema.arguments" in driver
    assert "ops.moe_sum = moe_sum_legacy_binary_compatibility" in driver
    assert "legacy two-argument _moe_C cannot apply an expert map" in driver
    assert "ensure_flash_attn_mla_compatibility()" in driver
    assert 'kwargs.get("cp_world_size", 1) <= 0' in driver
    assert 'kwargs["cp_world_size"] = 1' in driver
    assert "ensure_k3_warmup_compatibility()" in driver
    assert "kernel_warmup.kimi_k3_triton_warmup = kimi_k3_triton_warmup" in driver
    warmup = read("k3_vllm_warmup.py")
    assert "def _get_kda_layer(" in warmup
    assert "def _warm_attn_res(" in warmup
    assert "def _warm_recurrent_kda(" in warmup
    assert "def kimi_k3_triton_warmup(" in warmup
    assert "K3_PROXY_GENERATE_OK" in driver


def test_kda_backward_probe_is_one_gpu_and_uses_production_shape():
    carrier = read("probe_kda_backward.sbatch")
    probe = read("probe_kda_backward.py")

    assert "#SBATCH --partition=interactive" in carrier
    assert "#SBATCH --gpus-per-node=1" in carrier
    assert "#SBATCH --time=00:10:00" in carrier
    assert (
        "--export=K3_ROOT,MLITE_ROOT,VERL_ROOT,VERL_DEPS_SITE,K3_CACHE_ROOT" in carrier
    )
    assert "sleep 120" in carrier
    assert "sleep 180" in carrier
    assert 'backend="fla"' in probe
    assert "shape = (1, 16, 96, 128)" in probe
    assert '"backward_start"' in probe
    assert '"backward_done"' in probe
    assert "K3_KDA_BACKWARD_OK=" in probe


def test_ep8_moe_backward_probe_uses_one_production_shape_layer():
    carrier = read("probe_ep8_moe_backward.sbatch")
    probe = read("probe_ep8_moe_backward.py")

    assert "#SBATCH --partition=interactive" in carrier
    assert "#SBATCH --gpus-per-node=8" in carrier
    assert "#SBATCH --time=00:10:00" in carrier
    assert "--nproc-per-node=8" in carrier
    assert "ParallelLatentMoE" in probe
    assert "ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1)" in probe
    assert "hidden = torch.randn(" in probe
    assert "config.hidden_size" in probe
    assert '"backward_start"' in probe
    assert '"backward_done"' in probe
    assert "next(first_module.router.parameters())" in probe
    assert "K3_EP8_MOE_BACKWARD_OK=" in probe


def test_ep8_moe_backward_probe_can_isolate_multiple_layers():
    carrier = read("probe_ep8_moe_backward.sbatch")
    probe = read("probe_ep8_moe_backward.py")

    assert "MOE_LAYERS" in carrier
    assert "MOE_LAYERS" in probe
    assert "torch.nn.ModuleList" in probe
    assert '"forward_layer_done"' in probe
    assert '"backward_layer_enter"' in probe
    assert '"layers": layers' in probe


def test_ep8_decoder_probe_covers_kda_mla_dense_and_moe_blocks():
    carrier = read("probe_ep8_decoder_backward.sbatch")
    probe = read("probe_ep8_decoder_backward.py")

    assert "#SBATCH --gpus-per-node=8" in carrier
    assert "DECODER_LAYERS" in carrier
    assert "K3ParallelDecoderLayer" in probe
    assert "range(layers)" in probe
    assert "block_residual" in probe
    assert '"backward_layer_enter"' in probe
    assert "K3_EP8_DECODER_BACKWARD_OK=" in probe
