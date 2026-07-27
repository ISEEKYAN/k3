"""Megatron Lite protocol for the Kimi K3 external package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mlite_k3.config import K3Config
from mlite_k3.model import K3Model


@dataclass(frozen=True)
class ImplConfig:
    parallel: Any = None
    optimizer: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"


def build_model_config(source: str | Path | dict, **overrides) -> K3Config:
    if isinstance(source, dict):
        return K3Config._from_hf_dict(source, **overrides)
    return K3Config.from_hf(str(source), **overrides)


def _parallel_size(parallel: Any, name: str) -> int:
    if parallel is None:
        return 1
    value = getattr(parallel, name, 1)
    return 1 if value is None else int(value)


def build_model(model_cfg: K3Config, *, impl_cfg: ImplConfig):
    """Build the verified single-rank reference bundle.

    Distributed KDA/MLA/EP kernels are intentionally fail-loud until their
    scheduler-backed validation is published.
    """
    dimensions = {
        name: _parallel_size(impl_cfg.parallel, name)
        for name in ("tp", "ep", "etp", "pp", "cp")
    }
    if any(size != 1 for size in dimensions.values()):
        raise NotImplementedError(
            f"K3 distributed execution is not validated yet: {dimensions}"
        )
    if impl_cfg.optimizer is not None:
        raise NotImplementedError("K3 optimizer integration is not validated yet")

    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.parallel import ParallelState

    dtype = getattr(torch, impl_cfg.dtype)
    model = K3Model(model_cfg).to(device=impl_cfg.device, dtype=dtype)
    return ModelBundle(
        chunks=[model],
        parallel_state=ParallelState(),
        forward_step=lambda chunk, batch: chunk(
            input_ids=batch.input_ids,
            labels=batch.labels,
        ),
        extras={
            "model_cfg": model_cfg,
            "validated_scope": "single_rank_reference",
        },
    )


def vocab_size(model_cfg: K3Config) -> int:
    return model_cfg.vocab_size


__all__ = ["ImplConfig", "build_model", "build_model_config", "vocab_size"]
