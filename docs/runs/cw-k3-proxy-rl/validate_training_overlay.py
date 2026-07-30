"""Import the complete K3 training stack inside a Slurm container step."""

import importlib.metadata
import json

import fla
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


assert get_model_package("k3").__name__ == "mlite_k3"
assert resolve_runtime_model_name("k3", "lite") == "k3"
result = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformer_engine": importlib.metadata.version("transformer-engine"),
    "fla": importlib.metadata.version("flash-linear-attention"),
    "fla_file": fla.__file__,
    "megatron_core": megatron.core.__file__,
    "megatron_lite": megatron.lite.__file__,
    "mlite_k3": mlite_k3.__file__,
    "verl": verl.__file__,
    "te_rmsnorm": te.RMSNorm.__name__,
    "transformer_engine_file": transformer_engine.__file__,
}
print("K3_TRAINING_OVERLAY_OK", json.dumps(result, sort_keys=True), flush=True)
