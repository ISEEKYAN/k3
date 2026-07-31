"""Report the K3 image ABI relevant to Transformer Engine wheels."""

import importlib.metadata
import importlib.util
import json
import platform

import torch


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


result = {
    "machine": platform.machine(),
    "glibc": platform.libc_ver(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformer_engine_spec": str(importlib.util.find_spec("transformer_engine")),
    "transformer_engine": package_version("transformer-engine"),
    "transformer_engine_cu13": package_version("transformer-engine-cu13"),
    "transformer_engine_torch": package_version("transformer-engine-torch"),
}
print("K3_TE_BINARY_ABI", json.dumps(result, sort_keys=True), flush=True)
