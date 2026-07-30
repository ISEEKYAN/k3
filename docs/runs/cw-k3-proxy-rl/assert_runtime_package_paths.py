#!/usr/bin/env python3
"""Fail before Ray startup if the composite K3 package resolution drifts."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path


vllm_site = Path(os.environ["VLLM_SITE"]).absolute()
packages = ("huggingface_hub", "transformers", "vllm", "vllm._C", "tensordict")
resolved: dict[str, str] = {}
for name in packages:
    module = importlib.import_module(name)
    module_file = Path(module.__file__).absolute()
    if not module_file.is_relative_to(vllm_site):
        raise RuntimeError(
            f"{name} resolved outside precedence-owning vLLM site: {module_file}"
        )
    resolved[name] = str(module_file)

tensordict_version = importlib.metadata.version("tensordict")
if tensordict_version != "0.10.0":
    raise RuntimeError(f"unexpected tensordict version: {tensordict_version}")

print(
    "K3_RUNTIME_PACKAGE_PATHS_OK "
    + json.dumps(
        {"packages": resolved, "tensordict_version": tensordict_version},
        sort_keys=True,
    ),
    flush=True,
)
