"""Probe the proven MLite training base and existing rollout overlays."""

import importlib
import importlib.metadata
import json
import os
import traceback
from pathlib import Path

import torch


def probe_module(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(name.split(".", 1)[0])
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        return {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": version,
        }
    except Exception as error:  # noqa: BLE001 - this is a diagnostic boundary
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


modules = {
    name: probe_module(name)
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "fla",
        "vllm",
        "megatron.core",
        "megatron.lite",
        "mlite_k3",
        "verl",
        "verl.trainer.main_ppo",
        "verl_mlite.engine.mlite_engine",
    )
}
result: dict[str, object] = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "modules": modules,
}

config_path = Path(os.environ["MODEL_PATH"]) / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
result["model_type"] = config.get("model_type")
result["architectures"] = config.get("architectures")
if modules["vllm"]["ok"]:
    try:
        from vllm import ModelRegistry

        result["vllm_arch_supported"] = {
            architecture: ModelRegistry.is_model_supported(architecture)
            for architecture in config.get("architectures", [])
        }
    except Exception as error:  # noqa: BLE001 - this is a diagnostic boundary
        result["vllm_registry_error"] = f"{type(error).__name__}: {error}"

print("K3_TRAINING_BASE_PROBE", json.dumps(result, sort_keys=True), flush=True)
