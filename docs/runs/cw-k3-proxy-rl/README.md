# Kimi K3 x86 training overlay and 8-GPU QAT+R3 proxy

This recipe layers training-only dependencies onto the published Kimi K3 vLLM
image. It does not rebuild vLLM. The image, K3 base, and external source inputs
are pinned in `image.env`; the exact K3 delivery commit is checked and included
in the runtime cache manifest.

The proxy keeps the public model widths, a complete 12-layer AttnRes block,
full-attention layers 4/8/12, and 56 of 896 routed experts. It keeps the public
top-16 routing contract. `tools/build_proxy_checkpoint.py` slices the public
checkpoint and rewrites both the config and safetensors index.

Required overlay inputs:

- an immutable local `.sqsh` produced once by `cache_image.sbatch`;
- a Python 3.12 Transformer Engine site built from the pinned source;
- the pinned Megatron-LM/MLite and VERL source trees;
- a VERL dependency site that does not replace image-owned torch, vLLM,
  Transformers, FLA, torchcodec, CUTLASS, or CUDA libraries.

The VERL dependency site is a pruned view of the established training overlay.
It excludes packages owned by the base image or K3 vLLM overlay, preventing
their torch, Hugging Face, vLLM, Ray, CUDA, and NVIDIA packages from being
shadowed. The full closure probe must print `FULL_CLOSURE_OK` before Ray starts.
The pruned site retains accelerate because PEFT imports it while constructing
the deeper PPO trainer graph; the base and K3 vLLM sites do not provide it.

Set `K3_IMAGE_SQSH` to a new shared path and run `cache_image.sbatch` once.
The job uses pyxis `--container-save`, records the squashfs SHA-256, and all
remaining recipes reuse that file instead of importing Docker layers again.

Run `validate_training_overlay.sbatch` first. Its import test is deliberately
inside `srun`; login-node imports are not evidence. Then submit
`run_proxy_stage.sbatch` as four separate one-node `interactive` jobs with
`STAGE=import`, `construct`, `fwbw`, and `qat`. Only after all four pass, submit
`run_proxy_qat_r3.sbatch` for the optimizer-backed RL step. The final job
enables MLite MXFP4 QAT and R3 router replay, rejects fused routers, and
requires W&B logging.

`k3_training_env.sh` preserves the image PATH, adds CUDA compatibility
libraries instead of replacing `LD_LIBRARY_PATH`, sets `CC`/`CXX`, forces
`OMP_NUM_THREADS=1`, and creates persistent Triton/Inductor caches. The cache
fingerprint includes the image digest, every source pin, Python, torch,
Transformer Engine, FLA, GPU compute capability, and exact Python path. An
existing manifest with different contents aborts instead of being reused.

Example:

```bash
python3 tools/build_proxy_checkpoint.py \
  --source /path/to/Kimi-K3 \
  --output /shared/checkpoints/Kimi-K3-12l-56e

export K3_IMAGE_SQSH=/shared/images/kimi-k3-amd64.sqsh
sbatch --export=ALL cache_image.sbatch
sbatch --export=ALL validate_training_overlay.sbatch
sbatch --export=ALL run_proxy_qat_r3.sbatch
```

The GPU job emits diagnostics after two and five minutes and writes the first
observed W&B URL to `wandb-url.txt`. A skipped test, a dry run, or a process
that never reaches an optimizer-backed RL step is not a successful run.

Known environment exception: tensordict 0.10.0 declares `pyvers<0.2.0`, while
the validated overlay contains pyvers 0.2.2. That pyvers distribution still
reports `__version__ = "0.1.0"` in its module. The runtime closure imports both
successfully, so the preflight checks pyvers importability and location instead
of either inconsistent version string.
