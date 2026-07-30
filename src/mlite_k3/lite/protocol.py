"""Megatron Lite protocol for the Kimi K3 external package."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mlite_k3.validation_schema import (
    VALIDATION_AXES,
    is_verified_evidence_source,
)
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
_VALIDATION_AXES = VALIDATION_AXES
_VALIDATED_AXIS_EVIDENCE: dict[str, tuple[str, ...]] = {}
_VALIDATION_DOC_EVIDENCE = re.compile(
    r"<!-- K3_VALIDATED_AXIS_EVIDENCE_BEGIN -->\s*"
    r"```json\s*(\{.*?\})\s*```\s*"
    r"<!-- K3_VALIDATED_AXIS_EVIDENCE_END -->",
    re.DOTALL,
)


@dataclass(frozen=True)
class ImplConfig:
    parallel: Any = None
    optimizer: str | None = None
    optimizer_config: Any = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_thd: bool = False
    use_deepep: bool = False
    deterministic: bool = False
    kda_cp_mode: str = "headwise"
    qat: QATSpec | dict[str, Any] | None = None
    validation_evidence: Mapping[str, tuple[str, ...]] | None = None


def is_expert_param(name: str) -> bool:
    return ".moe.experts.fc" in name


def PLACEMENT_FN(param_name: str) -> list:
    from torch.distributed.tensor import Replicate, Shard

    if is_expert_param(param_name):
        if ".fc1." in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if ".fc2." in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if param_name.endswith(
        (
            ".self_attention.q_proj.linear.weight",
            ".self_attention.k_proj.linear.weight",
            ".self_attention.v_proj.linear.weight",
            ".self_attention.g_proj.linear.weight",
            ".self_attention.f_b_proj.linear.weight",
            ".self_attention.b_proj.linear.weight",
            ".self_attention.linear_q_up_proj.linear.weight",
            ".self_attention.linear_kv_up_proj.linear.weight",
            ".self_attention.linear_g_proj.linear.weight",
            ".mlp.gate_up.linear.weight",
            ".moe.shared_experts.gate_up.linear.weight",
        )
    ) or param_name.endswith(
        (
            ".self_attention.q_conv1d.weight",
            ".self_attention.k_conv1d.weight",
            ".self_attention.v_conv1d.weight",
            ".self_attention.A_log",
            ".self_attention.dt_bias",
        )
    ):
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if param_name.endswith(
        (
            ".self_attention.o_proj.linear.weight",
            ".self_attention.linear_proj.linear.weight",
            ".mlp.down.linear.weight",
            ".moe.shared_experts.down.linear.weight",
        )
    ):
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if param_name in {
        "embed_tokens.embedding.weight",
        "lm_head.col.linear.weight",
    }:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def _build_dist_opt_optimizer(
    chunks, model_cfg: K3Config, impl_cfg: ImplConfig, ps: Any
):
    from megatron.lite.primitive.optimizers.megatron_wrap import (
        build_dist_opt_training_optimizer,
    )

    return build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        model_name="k3",
        is_expert=is_expert_param,
        deterministic=impl_cfg.deterministic,
    )


def build_model_config(source: str | Path | dict, **overrides) -> K3Config:
    if isinstance(source, dict):
        return K3Config._from_hf_dict(source, **overrides)
    return K3Config.from_hf(str(source), **overrides)


def _parallel_size(parallel: Any, name: str) -> int:
    if parallel is None:
        return 1
    value = getattr(parallel, name, 1)
    return 1 if value is None else int(value)


def _resolve_validated_axes(
    dimensions: dict[str, int],
    *,
    use_thd: bool,
    validation_evidence: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    supplied = dict(
        _VALIDATED_AXIS_EVIDENCE if validation_evidence is None else validation_evidence
    )
    unknown = sorted(set(supplied) - set(_VALIDATION_AXES))
    missing_sources = sorted(axis for axis, sources in supplied.items() if not sources)
    invalid_sources = sorted(
        f"{axis}:{source}"
        for axis, sources in supplied.items()
        for source in sources
        if not is_verified_evidence_source(source)
    )
    if unknown or missing_sources or invalid_sources:
        raise RuntimeError(
            "invalid K3 validation evidence: "
            f"unknown_axes={unknown}, missing_sources={missing_sources}, "
            f"invalid_sources={invalid_sources}"
        )

    active = {
        axis
        for axis, size in dimensions.items()
        if axis in _VALIDATION_AXES and size > 1
    }
    if use_thd:
        active.add("thd")
    evidence = {
        axis: tuple(supplied[axis])
        for axis in _VALIDATION_AXES
        if axis in active and axis in supplied
    }
    return tuple(evidence), evidence


def _assert_validation_doc_contract(contents: str) -> None:
    match = _VALIDATION_DOC_EVIDENCE.search(contents)
    if match is None:
        raise RuntimeError("docs/validation.md is missing validated-axis evidence")
    raw = json.loads(match.group(1))
    documented = {
        str(axis): tuple(str(source) for source in sources)
        for axis, sources in raw.items()
    }
    if documented != _VALIDATED_AXIS_EVIDENCE:
        raise RuntimeError(
            "validated-axis evidence drift: "
            f"runtime={_VALIDATED_AXIS_EVIDENCE}, docs={documented}"
        )


def build_model(model_cfg: K3Config, *, impl_cfg: ImplConfig):
    """Build the verified reference or distributed bundle."""
    dimensions = {
        name: _parallel_size(impl_cfg.parallel, name)
        for name in ("tp", "ep", "etp", "pp", "cp")
    }
    if impl_cfg.optimizer not in (None, "dist_opt"):
        raise ValueError(f"Unknown K3 lite optimizer: {impl_cfg.optimizer!r}.")
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

    optimizer = None
    finalize_grads = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "dist_opt":
        optimizer, finalize_grads = _build_dist_opt_optimizer(
            chunks, model_cfg, impl_cfg, ps
        )
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        attach_model_sharded_state_dict(
            chunks,
            ps,
            get_placements=PLACEMENT_FN,
            is_expert=is_expert_param,
        )
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"

    validated_axes, validation_evidence = _resolve_validated_axes(
        dimensions,
        use_thd=impl_cfg.use_thd,
        validation_evidence=impl_cfg.validation_evidence,
    )
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "validated_scope": validated_scope,
            "validated_axes": validated_axes,
            "validation_evidence": validation_evidence,
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


def export_hf_weights(
    chunks: list[torch.nn.Module],
    model_cfg: K3Config,
    ps: Any,
    **kwargs,
):
    """Export gathered public K3 weights through the shared HF primitive."""
    from mlite_k3.lite.checkpoint import export_hf_weights as export_impl

    yield from export_impl(chunks, model_cfg, ps, **kwargs)


__all__ = [
    "ImplConfig",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "pack_r3_replay_mask",
    "pack_routed_experts",
    "router_replay_roots",
    "unpack_thd_forward_output",
    "vocab_size",
]
