#!/usr/bin/env python3
"""Fail before Ray startup if the composite K3 package resolution drifts."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch


def is_within_root(actual: Path, expected_root: Path) -> bool:
    """Check package provenance after resolving both sides of symlinks."""
    actual_real = os.path.realpath(actual)
    expected_prefix = os.path.realpath(expected_root) + os.sep
    return actual_real.startswith(expected_prefix)


def assert_runtime_package_paths() -> dict[str, object]:
    container_site = Path("/usr/local/lib/python3.12/dist-packages")
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"host Python leaked into container: {sys.executable}")
    torch_file = Path(torch.__file__).absolute()
    if "nv26.05" not in torch.__version__:
        raise RuntimeError(f"host torch leaked into container: {torch.__version__}")
    if not is_within_root(torch_file, container_site / "torch"):
        raise RuntimeError(f"torch resolved outside container site: {torch_file}")
    container_python = {
        "python_executable": sys.executable,
        "torch_file": str(torch_file),
        "torch_version": torch.__version__,
    }
    print(
        "K3_CONTAINER_PYTHON_OK " + json.dumps(container_python, sort_keys=True),
        flush=True,
    )

    vllm_site = Path(os.environ["VLLM_SITE"]).absolute()
    verl_pruned_site = Path(os.environ["VERL_PRUNED_SITE"]).absolute()
    expected_modules = {
        "huggingface_hub": (
            "huggingface_hub",
            container_site / "huggingface_hub",
        ),
        "transformers": ("transformers", vllm_site / "transformers"),
        "vllm": ("vllm", vllm_site / "vllm"),
        # This vLLM build names its ABI-stable CUDA extension explicitly.
        "vllm._C": ("vllm._C_stable_libtorch", vllm_site / "vllm"),
        "ray": ("ray", vllm_site / "ray"),
        "wandb": ("wandb", vllm_site / "wandb"),
        "tensordict": ("tensordict", verl_pruned_site / "tensordict"),
        "pyvers": ("pyvers", verl_pruned_site / "pyvers"),
        "hydra": ("hydra", verl_pruned_site / "hydra"),
        "codetiming": ("codetiming", verl_pruned_site / "codetiming"),
        "orjson": ("orjson", verl_pruned_site / "orjson"),
        "accelerate": ("accelerate", verl_pruned_site / "accelerate"),
    }
    resolved: dict[str, str] = {}
    for label, (module_name, expected_site) in expected_modules.items():
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__).absolute()
        if not is_within_root(module_file, expected_site):
            raise RuntimeError(
                f"{label} resolved outside expected site {expected_site}: {module_file}"
            )
        resolved[label] = str(module_file)

    tensordict_version = importlib.metadata.version("tensordict")
    if tensordict_version != "0.10.0":
        raise RuntimeError(f"unexpected tensordict version: {tensordict_version}")
    return {
        "packages": resolved,
        "tensordict_version": tensordict_version,
        **container_python,
    }


if __name__ == "__main__":
    result = assert_runtime_package_paths()
    print(
        "K3_RUNTIME_PACKAGE_PATHS_OK " + json.dumps(result, sort_keys=True),
        flush=True,
    )
