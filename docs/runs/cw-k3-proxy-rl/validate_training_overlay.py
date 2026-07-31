"""Import the complete K3 training stack inside a Slurm container step."""

import ast
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import vllm
from k3_vllm_warmup import kimi_k3_triton_warmup
from vllm.model_executor.warmup import kernel_warmup
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
    is_available as is_ll_bf16_gemm_available,
)
import transformer_engine
import transformer_engine.pytorch as te
import fla
import fla.utils as fla_utils
import cutlass.cute
import megatron.core
import megatron.lite
import torch
from vllm.models.kimi_k3.nvidia.kda import is_flashkda_supported
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def _under(module_file: str, root: str) -> bool:
    return Path(module_file).resolve().is_relative_to(Path(root).resolve())


def _load_normalize_cp_world_size():
    source_path = (
        Path(os.environ["VLLM_SITE"])
        / "vllm/v1/attention/backends/mla/flashattn_mla.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_cp_world_size"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["normalize_cp_world_size"]


def _validate_layerwise_bucket_transaction() -> None:
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        freeze_load_plan,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    class TwoTensorModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.right = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.left.weight_loader = self._load
            self.right.weight_loader = self._load

        @staticmethod
        def _load(param, weight):
            param.data.copy_(weight)

        def load_weights(self, weights):
            loaded = set()
            for name, weight in weights:
                param = self.get_parameter(name)
                param.weight_loader(param, weight)
                loaded.add(name)
            return loaded

    model = TwoTensorModel()
    record_metadata_for_reloading(model)
    model.load_weights(
        (
            ("left", torch.tensor([1.0, 2.0])),
            ("right", torch.tensor([3.0, 4.0])),
        )
    )
    freeze_load_plan(model)

    initialize_layerwise_reload(model)
    model.load_weights((("left", torch.tensor([5.0, 6.0])),))
    model.load_weights((("right", torch.tensor([7.0, 8.0])),))
    finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))

    assert torch.equal(model.left, torch.tensor([5.0, 6.0]))
    assert torch.equal(model.right, torch.tensor([7.0, 8.0]))


def _validate_dummy_start_layerwise_bucket_transaction() -> None:
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    class TwoTensorModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.right = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.left.weight_loader = self._load
            self.right.weight_loader = self._load

        @staticmethod
        def _load(param, weight):
            param.data.copy_(weight)

        def load_weights(self, weights):
            for name, weight in weights:
                param = self.get_parameter(name)
                param.weight_loader(param, weight)

    model = TwoTensorModel()
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    model.load_weights((("left", torch.tensor([5.0, 6.0])),))
    model.load_weights((("right", torch.tensor([7.0, 8.0])),))
    finalize_layerwise_reload(
        model,
        SimpleNamespace(dtype=torch.float32),
        fail_on_incomplete=False,
    )

    assert torch.equal(model.left, torch.tensor([5.0, 6.0]))
    assert torch.equal(model.right, torch.tensor([7.0, 8.0]))


assert _under(vllm.__file__, os.environ["VLLM_SITE"]), vllm.__file__
assert kernel_warmup.kimi_k3_triton_warmup is kimi_k3_triton_warmup
kernel_warmup._warmup_ll_bf16_router_gemm(object())
assert not is_ll_bf16_gemm_available()
assert "import vllm._flashkda_C" in inspect.getsource(is_flashkda_supported)
normalize_cp_world_size = _load_normalize_cp_world_size()
assert [normalize_cp_world_size(value) for value in (-1, 0, 1, 2)] == [1, 1, 1, 2]
assert _under(fla.__file__, os.environ["FLA_SITE"]), fla.__file__
assert _under(cutlass.cute.__file__, os.environ["CUTLASS_DSL_SITE"]), (
    cutlass.cute.__file__
)
assert _under(megatron.lite.__file__, os.environ["MLITE_ROOT"]), megatron.lite.__file__

verl_extension_source = inspect.getsource(vLLMColocateWorkerExtension)
assert '"mxfp4-pack-quantized"' in verl_extension_source
assert verl_extension_source.index("initialize_layerwise_reload") < (
    verl_extension_source.index("receiver.receive_weights")
)
assert verl_extension_source.index("receiver.receive_weights") < (
    verl_extension_source.index("finalize_layerwise_reload")
)
_validate_layerwise_bucket_transaction()
_validate_dummy_start_layerwise_bucket_transaction()
assert _under(transformer_engine.__file__, "/usr/local/lib/python3.12/dist-packages"), (
    transformer_engine.__file__
)
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
    "mxfp4_layerwise_bucket_transaction": True,
    "kda_triton_fallback_contract": True,
    "fa3_cp_off_contract": True,
    "te_rmsnorm": te.RMSNorm.__name__,
    "transformer_engine_file": transformer_engine.__file__,
}
print("K3_TRAINING_OVERLAY_OK", json.dumps(result, sort_keys=True), flush=True)
