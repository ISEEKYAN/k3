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


def assert_runtime_package_paths() -> dict[str, object]:
    container_site = Path("/usr/local/lib/python3.12/dist-packages")
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"host Python leaked into container: {sys.executable}")
    torch_file = Path(torch.__file__).absolute()
    if "nv26.05" not in torch.__version__:
        raise RuntimeError(f"host torch leaked into container: {torch.__version__}")
    if not torch_file.is_relative_to(container_site):
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
    tensordict_site = Path(os.environ["TENSORDICT_SITE"]).absolute()
    base_site = Path(os.environ["VERL_DEPS_SITE"]).absolute()
    expected_modules = {
        "huggingface_hub": ("huggingface_hub", base_site),
        "transformers": ("transformers", vllm_site),
        "vllm": ("vllm", vllm_site),
        # This vLLM build names its ABI-stable CUDA extension explicitly.
        "vllm._C": ("vllm._C_stable_libtorch", vllm_site),
        "ray": ("ray", vllm_site),
        "tensordict": ("tensordict", tensordict_site),
        "pyvers": ("pyvers", tensordict_site),
    }
    resolved: dict[str, str] = {}
    for label, (module_name, expected_site) in expected_modules.items():
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__).absolute()
        if not module_file.is_relative_to(expected_site):
            raise RuntimeError(
                f"{label} resolved outside expected site {expected_site}: {module_file}"
            )
        resolved[label] = str(module_file)

    tensordict_version = importlib.metadata.version("tensordict")
    if tensordict_version != "0.10.0":
        raise RuntimeError(f"unexpected tensordict version: {tensordict_version}")
    pyvers = importlib.import_module("pyvers")
    pyvers_version = pyvers.__version__
    if pyvers_version != "0.1.0":
        raise RuntimeError(f"unexpected pyvers version: {pyvers_version}")
    return {
        "packages": resolved,
        "pyvers_version": pyvers_version,
        "tensordict_version": tensordict_version,
        **container_python,
    }


if __name__ == "__main__":
    result = assert_runtime_package_paths()
    print(
        "K3_RUNTIME_PACKAGE_PATHS_OK " + json.dumps(result, sort_keys=True),
        flush=True,
    )
