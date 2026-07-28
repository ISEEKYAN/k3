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
    use_thd: bool = False
    use_deepep: bool = False
    deterministic: bool = False


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
    """Build the verified reference or tensor-parallel bundle.

    Distributed axes other than TP remain fail-loud until their
    scheduler-backed validation is published.
    """
    dimensions = {
        name: _parallel_size(impl_cfg.parallel, name)
        for name in ("tp", "ep", "etp", "pp", "cp")
    }
    blocked = {
        name: size
        for name, size in dimensions.items()
        if name != "tp" and size != 1
    }
    if blocked:
        raise NotImplementedError(
            f"K3 distributed axes are not validated yet: {blocked}"
        )
    if impl_cfg.use_thd:
        raise NotImplementedError("K3 THD execution is not validated yet")
    if impl_cfg.optimizer is not None:
        raise NotImplementedError("K3 optimizer integration is not validated yet")

    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.parallel import ParallelState, init_parallel

    dtype = getattr(torch, impl_cfg.dtype)
    use_distributed_model = impl_cfg.device != "cpu" or dimensions["tp"] > 1
    if use_distributed_model:
        if not torch.distributed.is_initialized():
            raise RuntimeError("K3 TP model requires torch.distributed initialization")
        from mlite_k3.lite.model import K3ParallelModel

        ps = init_parallel(impl_cfg.parallel)
        model = K3ParallelModel(
            model_cfg,
            ps,
            use_thd=False,
            use_deepep=impl_cfg.use_deepep,
            deterministic=impl_cfg.deterministic,
        ).to(device=impl_cfg.device, dtype=dtype)
        validated_scope = "tensor_parallel"
    else:
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
        },
    )


def vocab_size(model_cfg: K3Config) -> int:
    return model_cfg.vocab_size


__all__ = ["ImplConfig", "build_model", "build_model_config", "vocab_size"]
