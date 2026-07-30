"""Megatron Lite protocol for the Kimi K3 external package."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from megatron.lite.model.protocol_utils import (
    pack_r3_replay_mask,
    pack_routed_experts,
    router_replay_roots,
    unpack_thd_forward_output,
)

from mlite_k3.config import K3Config
from mlite_k3.model import K3Model

if TYPE_CHECKING:
    from megatron.lite.primitive.quantization.qat import QATSpec


_K3_MXFP4_QAT_IGNORES = (
    "embed_tokens",
    "lm_head",
    "router",
    "shared_experts",
    "self_attention",
    "self_attention_res_proj",
    "mlp",
    "mlp_res_proj",
    "output_attn_res_proj",
)


@dataclass(frozen=True)
class ImplConfig:
    parallel: Any = None
    optimizer: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_thd: bool = False
    use_deepep: bool = False
    deterministic: bool = False
    kda_cp_mode: str = "headwise"
    qat: QATSpec | dict[str, Any] | None = None


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
    """Build the verified reference or distributed bundle."""
    dimensions = {
        name: _parallel_size(impl_cfg.parallel, name)
        for name in ("tp", "ep", "etp", "pp", "cp")
    }
    if impl_cfg.optimizer is not None:
        raise NotImplementedError("K3 optimizer integration is not validated yet")
    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.parallel import ParallelState, init_parallel

    dtype = getattr(torch, impl_cfg.dtype)
    use_distributed_model = (
        impl_cfg.device != "cpu"
        or impl_cfg.use_thd
        or any(size > 1 for size in dimensions.values())
    )
    if use_distributed_model:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "K3 distributed model requires torch.distributed initialization"
            )
        from mlite_k3.lite.model import K3ParallelModel

        ps = init_parallel(impl_cfg.parallel)
        model = K3ParallelModel(
            model_cfg,
            ps,
            use_thd=impl_cfg.use_thd,
            use_deepep=impl_cfg.use_deepep,
            deterministic=impl_cfg.deterministic,
            kda_cp_mode=impl_cfg.kda_cp_mode,
        ).to(device=impl_cfg.device, dtype=dtype)
        validated_scope = "distributed"
    else:
        ps = ParallelState()
        model = K3Model(model_cfg).to(device=impl_cfg.device, dtype=dtype)
        validated_scope = "single_rank_reference"

    def forward_step(chunk, batch):
        if impl_cfg.use_thd:
            from megatron.lite.model.protocol_utils import pack_thd_forward_kwargs

            kwargs = pack_thd_forward_kwargs(chunk, batch)
            packed_seq_params = kwargs["packed_seq_params"]
            if ps.cp_size == 1:
                packed_seq_params.local_cp_size = 1
            elif packed_seq_params.local_cp_size != ps.cp_size:
                raise ValueError(
                    "K3 THD protocol must mark CP-local tensors with "
                    f"local_cp_size={ps.cp_size}"
                )
            packed_seq_params.total_tokens = int(
                packed_seq_params.cu_seqlens_q[-1].item()
            )
            return chunk(
                input_ids=kwargs["input_ids"],
                labels=kwargs["labels"],
                loss_mask=kwargs["loss_mask"],
                packed_seq_params=packed_seq_params,
            )
        return chunk(
            input_ids=batch.input_ids,
            labels=batch.labels,
            loss_mask=getattr(batch, "loss_mask", None),
        )

    chunks = [model]
    from megatron.lite.primitive.quantization.qat import (
        apply_qat_to_chunks,
        normalize_qat_spec,
    )

    qat_spec = normalize_qat_spec(impl_cfg.qat)
    if qat_spec.enabled and qat_spec.format == "mxfp4":
        qat_spec = replace(
            qat_spec,
            ignore_patterns=tuple(
                dict.fromkeys((*qat_spec.ignore_patterns, *_K3_MXFP4_QAT_IGNORES))
            ),
        )
    qat_stats = apply_qat_to_chunks(chunks, qat_spec)

    validated_axes = tuple(name for name, size in dimensions.items() if size > 1)
    if impl_cfg.use_thd:
        validated_axes += ("thd",)
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        forward_step=forward_step,
        extras={
            "model_cfg": model_cfg,
            "validated_scope": validated_scope,
            "validated_axes": validated_axes,
            "qat": qat_stats,
        },
    )


def vocab_size(model_cfg: K3Config) -> int:
    return model_cfg.vocab_size


def load_hf_weights(
    chunk: torch.nn.Module,
    hf_path: str,
    model_cfg: K3Config,
    ps: Any,
):
    """Load a public K3 checkpoint through the shared HFWeights primitive."""
    from mlite_k3.lite.checkpoint import load_hf_weights as load_impl

    return load_impl(chunk, hf_path, model_cfg, ps)


__all__ = [
    "ImplConfig",
    "build_model",
    "build_model_config",
    "load_hf_weights",
    "pack_r3_replay_mask",
    "pack_routed_experts",
    "router_replay_roots",
    "unpack_thd_forward_output",
    "vocab_size",
]
