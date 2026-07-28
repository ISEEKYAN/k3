"""Megatron Lite protocol for the Kimi K3 external package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.lite.runtime.contracts import ParallelConfig

from mlite_k3.config import K3Config
from mlite_k3.model import K3Model


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
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
    """Build the CPU oracle or the axis-gated Lite parallel composition."""
    dimensions = {
        name: _parallel_size(impl_cfg.parallel, name)
        for name in ("tp", "ep", "etp", "pp", "cp")
    }
    for axis in ("ep", "etp", "pp", "cp"):
        if dimensions[axis] != 1:
            raise NotImplementedError(
                f"K3 {axis.upper()} execution is not validated yet: {dimensions[axis]}"
            )
    if dimensions["tp"] != 1 and impl_cfg.device != "cuda":
        raise NotImplementedError(
            "K3 TP execution requires the scheduler-backed CUDA path"
        )
    if impl_cfg.optimizer is not None:
        raise NotImplementedError("K3 optimizer integration is not validated yet")

    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.parallel import ParallelState, init_parallel

    dtype = getattr(torch, impl_cfg.dtype)
    use_parallel_model = impl_cfg.device == "cuda" and dist.is_initialized()
    if use_parallel_model:
        from mlite_k3.lite.model import K3ParallelModel

        ps = init_parallel(impl_cfg.parallel)
        model = K3ParallelModel(model_cfg, ps).to(dtype=dtype).cuda()
        validated_scope = "tp"
    else:
        if dimensions["tp"] != 1:
            raise RuntimeError(
                "K3 TP execution requires torch.distributed initialization"
            )
        ps = ParallelState()
        model = K3Model(model_cfg).to(device=impl_cfg.device, dtype=dtype)
        validated_scope = "single_rank_reference"
    return ModelBundle(
        chunks=[model],
        parallel_state=ps,
        forward_step=lambda chunk, batch: chunk(
            input_ids=batch.input_ids,
            labels=batch.labels,
        ),
        extras={
            "model_cfg": model_cfg,
            "validated_scope": validated_scope,
            "validated_axes": ["tp"] if use_parallel_model else [],
        },
    )


def vocab_size(model_cfg: K3Config) -> int:
    return model_cfg.vocab_size


__all__ = ["ImplConfig", "build_model", "build_model_config", "vocab_size"]
