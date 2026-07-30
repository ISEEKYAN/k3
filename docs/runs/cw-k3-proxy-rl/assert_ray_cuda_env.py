#!/usr/bin/env python3
"""Fail fast when Ray leaks the ROCm visibility alias into CUDA workers."""

from __future__ import annotations

import os

import ray

from assert_runtime_package_paths import assert_runtime_package_paths


def _assert_cuda_only(scope: str) -> str:
    assert "ROCR_VISIBLE_DEVICES" not in os.environ, (
        f"{scope}: ROCR_VISIBLE_DEVICES must be absent on a CUDA-only worker"
    )
    assert "CUDA_VISIBLE_DEVICES" in os.environ, (
        f"{scope}: CUDA_VISIBLE_DEVICES must identify the assigned GPU"
    )
    return os.environ["CUDA_VISIBLE_DEVICES"]


ray.init(
    address="auto",
    runtime_env={
        "env_vars": {
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            "TENSORDICT_SITE": os.environ["TENSORDICT_SITE"],
            "VLLM_SITE": os.environ["VLLM_SITE"],
        }
    },
)


@ray.remote(num_gpus=1)
def check_gpu_actor_environment() -> dict[str, object]:
    cuda_visible_devices = _assert_cuda_only("Ray GPU actor")
    package_paths = assert_runtime_package_paths()
    return {
        "cuda_visible_devices": cuda_visible_devices,
        "package_paths": package_paths,
    }


actor_contract = ray.get(check_gpu_actor_environment.remote())
print(f"K3_RAY_CUDA_ENV_OK actor_contract={actor_contract}", flush=True)
ray.shutdown()
