#!/usr/bin/env python3
"""Fail before Ray startup if the composite K3 package resolution drifts."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path


def assert_runtime_package_paths() -> dict[str, object]:
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
    return {"packages": resolved, "tensordict_version": tensordict_version}


if __name__ == "__main__":
    result = assert_runtime_package_paths()
    print(
        "K3_RUNTIME_PACKAGE_PATHS_OK " + json.dumps(result, sort_keys=True),
        flush=True,
    )
