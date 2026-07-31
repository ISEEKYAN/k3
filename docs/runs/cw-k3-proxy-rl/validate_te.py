"""Fail-loud ABI check for a freshly built Transformer Engine layer."""

import importlib.metadata

import torch
import transformer_engine.pytorch as te


print(
    "K3_TE_BUILD_OK",
    f"torch={torch.__version__}",
    f"te={importlib.metadata.version('transformer-engine')}",
    f"rmsnorm={te.RMSNorm.__name__}",
    flush=True,
)
