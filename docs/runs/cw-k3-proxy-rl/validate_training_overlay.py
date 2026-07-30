"""Import the complete K3 training stack inside a Slurm container step."""

import importlib.metadata
import json
import os
from pathlib import Path

import vllm
import transformer_engine
import transformer_engine.pytorch as te
import fla
import fla.utils as fla_utils
import cutlass.cute
import megatron.core
import megatron.lite
import torch


def _under(module_file: str, root: str) -> bool:
    return Path(module_file).resolve().is_relative_to(Path(root).resolve())


assert _under(vllm.__file__, os.environ["VLLM_SITE"]), vllm.__file__
assert _under(fla.__file__, os.environ["FLA_SITE"]), fla.__file__
assert _under(
    cutlass.cute.__file__, os.environ["CUTLASS_DSL_SITE"]
), cutlass.cute.__file__
assert _under(megatron.lite.__file__, os.environ["MLITE_ROOT"]), megatron.lite.__file__
assert _under(
    transformer_engine.__file__, "/usr/local/lib/python3.12/dist-packages"
), transformer_engine.__file__
if torch.cuda.is_available():
    assert fla_utils.device_platform == "cuda", fla_utils.device_platform
    assert fla_utils.IS_NVIDIA
result = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "vllm_file": vllm.__file__,
    "transformer_engine": importlib.metadata.version("transformer-engine"),
    "fla": importlib.metadata.version("flash-linear-attention"),
    "fla_file": fla.__file__,
    "fla_device_platform": fla_utils.device_platform,
    "cutlass_cute_file": cutlass.cute.__file__,
    "megatron_core": megatron.core.__file__,
    "megatron_lite": megatron.lite.__file__,
    "megatron_lite_version": os.environ["MLITE_SOURCE_SHA"],
    "te_rmsnorm": te.RMSNorm.__name__,
    "transformer_engine_file": transformer_engine.__file__,
}
print("K3_TRAINING_OVERLAY_OK", json.dumps(result, sort_keys=True), flush=True)
