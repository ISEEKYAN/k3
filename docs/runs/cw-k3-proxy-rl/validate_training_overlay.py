"""Import the complete K3 training stack inside a Slurm container step."""

import importlib.metadata
import json

import fla
import fla.utils as fla_utils
import megatron.core
import megatron.lite
import mlite_k3
import torch
import transformer_engine
import transformer_engine.pytorch as te
import verl
import vllm
from megatron.lite.model.registry import (
    get_model_package,
    resolve_runtime_model_name,
)

verl_main_ppo = importlib.import_module("verl.trainer.main_ppo")
verl_mlite_engine = importlib.import_module("verl_mlite.engine.mlite_engine")


assert get_model_package("k3").__name__ == "mlite_k3"
assert resolve_runtime_model_name("k3", "lite") == "k3"
if torch.cuda.is_available():
    assert fla_utils.device_platform == "cuda", fla_utils.device_platform
    assert fla_utils.IS_NVIDIA
result = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformer_engine": importlib.metadata.version("transformer-engine"),
    "fla": importlib.metadata.version("flash-linear-attention"),
    "fla_file": fla.__file__,
    "fla_device_platform": fla_utils.device_platform,
    "megatron_core": megatron.core.__file__,
    "megatron_lite": megatron.lite.__file__,
    "mlite_k3": mlite_k3.__file__,
    "verl": verl.__file__,
    "verl_main_ppo": verl_main_ppo.__file__,
    "verl_mlite_engine": verl_mlite_engine.__file__,
    "te_rmsnorm": te.RMSNorm.__name__,
    "transformer_engine_file": transformer_engine.__file__,
}
print("K3_TRAINING_OVERLAY_OK", json.dumps(result, sort_keys=True), flush=True)
