"""Kimi K3 public-checkpoint loading helpers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt.fused_weights import (
    FusedWeightLayout,
    QuantizedWeight,
    WeightSegment,
)


_EXPECTED_MXFP4_IGNORES = frozenset(
    {
        r"re:.*self_attn.*",
        r"re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        r"re:.*lm_head.*",
        r"re:.*vision_tower.*",
        r"re:.*mm_projector.*",
    }
)
_ROUTED_MXFP4_KEY = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+"
    r"\.w[123]\.weight_(packed|scale)$"
)
_ROUTED_MXFP4_WEIGHT = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+"
    r"\.w[123]\.weight$"
)
_GROUPED_EXPERT_WEIGHT = re.compile(r"^(.*\.moe\.experts\.fc[12]\.weight)(\d+)$")
_MXFP4_ADAPTER_EXPERT = re.compile(
    r"^(layers\.\d+\.moe\.mxfp4\.w[123]_(?:packed|scale)\.weight)(\d+)$"
)


@dataclass(frozen=True)
class WeightIndexAudit:
    """Summary of the tensors reachable from a Hugging Face weight index."""

    quantized_weights: int
    plain_tensors: int
    shards: int


@dataclass(frozen=True)
class K3QuantizationMetadata:
    """Frozen compressed-tensors contract for the public K3 text weights."""

    format: str
    group_size: int
    num_bits: int
    scale_dtype: str
    ignored_modules: frozenset[str]


@dataclass(frozen=True)
class K3CheckpointManifest:
    """Metadata-only checkpoint inspection result."""

    quantization: K3QuantizationMetadata
    weights: WeightIndexAudit


@dataclass(frozen=True)
class K3RankWeight:
    """One local state tensor selected by a metadata-only rank dry-run."""

    native_name: str
    hf_names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype


class K3WeightSpec:
    """Exact Kimi K3 text-backbone mapping for the public release."""

    def __init__(
        self,
        config: Any,
        *,
        manifest: K3CheckpointManifest | None = None,
    ):
        self.config = config
        self.manifest = manifest
        self._mapping: dict[str, list[str]] | None = None
        self._fusion_layouts: dict[str, FusedWeightLayout] = {}
        self._raw_mxfp4_sources: dict[str, tuple[tuple[str, torch.Tensor], ...]] = {}

    @property
    def kda_rollout_layout(self) -> FusedWeightLayout:
        """The training-to-rollout KDA projection contract."""
        heads = int(self.config.kda_num_heads)
        head_dim = int(self.config.kda_head_dim)
        return FusedWeightLayout(
            name="in_proj_qkvgfab",
            segments=(
                WeightSegment("q", heads, head_dim),
                WeightSegment("k", heads, head_dim),
                WeightSegment("v", heads, head_dim),
                WeightSegment("g", heads, head_dim),
                WeightSegment("f_a", 1, head_dim, replicated=True),
                WeightSegment("b", heads, 1),
            ),
        )

    def _register_fusion(
        self,
        native_name: str,
        *,
        rows: int,
        segment_names: tuple[str, ...],
    ) -> None:
        self._fusion_layouts[native_name] = FusedWeightLayout(
            name=native_name,
            segments=tuple(
                WeightSegment(segment_name, 1, rows) for segment_name in segment_names
            ),
        )

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def _uses_release_mxfp4(self) -> bool:
        return (
            self.manifest is not None
            and self.manifest.quantization.format == "mxfp4-pack-quantized"
        )

    def _hf_sources(self, *names: str) -> list[str]:
        if not self._uses_release_mxfp4:
            return list(names)
        return [
            source for name in names for source in (f"{name}_packed", f"{name}_scale")
        ]

    def raw_hf_source(
        self,
        native_name: str,
        index: int,
        resolved_name: str,
    ) -> bool:
        """Keep paired release MXFP4 components raw until K3 materializes them."""
        del index
        return (
            self._uses_release_mxfp4
            and self.is_expert(native_name)
            and resolved_name.endswith(("_packed", "_scale"))
        )

    def weight_map(self) -> dict[str, list[str]]:
        if self._mapping is not None:
            return self._mapping
        mapping = {
            "embed_tokens.embedding.weight": [
                "language_model.model.embed_tokens.weight"
            ],
            "output_attn_res_norm.weight": [
                "language_model.model.output_attn_res_norm.weight"
            ],
            "output_attn_res_proj.weight": [
                "language_model.model.output_attn_res_proj.weight"
            ],
            "norm.weight": ["language_model.model.norm.weight"],
            "lm_head.col.linear.weight": ["language_model.lm_head.weight"],
        }
        for layer in range(self.config.num_hidden_layers):
            native = f"layers.{layer}"
            hf = f"language_model.model.layers.{layer}"
            for name in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attention_res_norm.weight",
                "mlp_res_norm.weight",
                "self_attention_res_proj.weight",
                "mlp_res_proj.weight",
            ):
                mapping[f"{native}.{name}"] = [f"{hf}.{name}"]
            if self.config.attention_type(layer) == "kda":
                self._add_kda(mapping, native, hf)
            else:
                self._add_mla(mapping, native, hf)
            if layer < self.config.first_k_dense_replace:
                fused_name = f"{native}.mlp.gate_up.linear.weight"
                mapping[fused_name] = [
                    f"{hf}.mlp.gate_proj.weight",
                    f"{hf}.mlp.up_proj.weight",
                ]
                self._register_fusion(
                    fused_name,
                    rows=int(self.config.intermediate_size),
                    segment_names=("gate", "up"),
                )
                mapping[f"{native}.mlp.down.linear.weight"] = [
                    f"{hf}.mlp.down_proj.weight"
                ]
            else:
                self._add_moe(mapping, native, hf)
        self._mapping = mapping
        return mapping

    @staticmethod
    def _add_kda(mapping: dict[str, list[str]], native: str, hf: str) -> None:
        names = {
            "A_log": "A_log",
            "dt_bias": "dt_bias",
            "q_proj.linear.weight": "q_proj.weight",
            "k_proj.linear.weight": "k_proj.weight",
            "v_proj.linear.weight": "v_proj.weight",
            "f_a_proj.weight": "f_a_proj.weight",
            "f_b_proj.linear.weight": "f_b_proj.weight",
            "b_proj.linear.weight": "b_proj.weight",
            "g_proj.linear.weight": "g_proj.weight",
            "o_norm.weight": "o_norm.weight",
            "o_proj.linear.weight": "o_proj.weight",
        }
        for native_name, hf_name in names.items():
            mapping[f"{native}.self_attention.{native_name}"] = [
                f"{hf}.self_attn.{hf_name}"
            ]
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            mapping[f"{native}.self_attention.{name}.weight"] = [
                f"{hf}.self_attn.{name}.weight"
            ]

    @staticmethod
    def _add_mla(mapping: dict[str, list[str]], native: str, hf: str) -> None:
        names = {
            "linear_q_down_proj.weight": "q_a_proj.weight",
            "linear_q_up_proj.linear.layer_norm_weight": "q_a_layernorm.weight",
            "linear_q_up_proj.linear.weight": "q_b_proj.weight",
            "linear_kv_down_proj.weight": "kv_a_proj_with_mqa.weight",
            "linear_kv_up_proj.linear.layer_norm_weight": "kv_a_layernorm.weight",
            "linear_kv_up_proj.linear.weight": "kv_b_proj.weight",
            "linear_g_proj.linear.weight": "g_proj.weight",
            "linear_proj.linear.weight": "o_proj.weight",
        }
        for native_name, hf_name in names.items():
            mapping[f"{native}.self_attention.{native_name}"] = [
                f"{hf}.self_attn.{hf_name}"
            ]

    def _add_moe(self, mapping: dict[str, list[str]], native: str, hf: str) -> None:
        prefix = f"{hf}.block_sparse_moe"
        fixed = {
            "router.expert_bias": "gate.e_score_correction_bias",
            "router.gate.weight": "gate.weight",
            "routed_expert_down_proj.weight": "routed_expert_down_proj.weight",
            "routed_expert_norm.weight": "routed_expert_norm.weight",
            "routed_expert_up_proj.weight": "routed_expert_up_proj.weight",
            "shared_experts.gate_up.linear.weight": (
                "shared_experts.gate_proj.weight",
                "shared_experts.up_proj.weight",
            ),
            "shared_experts.down.linear.weight": "shared_experts.down_proj.weight",
        }
        for native_suffix, hf_suffixes in fixed.items():
            if isinstance(hf_suffixes, str):
                hf_suffixes = (hf_suffixes,)
            mapping[f"{native}.moe.{native_suffix}"] = [
                f"{prefix}.{suffix}" for suffix in hf_suffixes
            ]
        shared_fused_name = f"{native}.moe.shared_experts.gate_up.linear.weight"
        self._register_fusion(
            shared_fused_name,
            rows=int(self.config.shared_expert_intermediate_size),
            segment_names=("gate", "up"),
        )
        for expert in range(self.config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert}"
            fused_name = f"{native}.moe.experts.fc1.weight{expert}"
            mapping[fused_name] = self._hf_sources(
                f"{expert_prefix}.w1.weight",
                f"{expert_prefix}.w3.weight",
            )
            self._register_fusion(
                fused_name,
                rows=int(self.config.moe_intermediate_size),
                segment_names=("gate", "up"),
            )
            mapping[f"{native}.moe.experts.fc2.weight{expert}"] = self._hf_sources(
                f"{expert_prefix}.w2.weight"
            )

    @staticmethod
    def _materialize_mxfp4_pair(
        packed: torch.Tensor,
        encoded_scale: torch.Tensor,
    ) -> torch.Tensor:
        if packed.dtype not in (torch.uint8, torch.int8):
            raise TypeError(
                f"MXFP4 packed tensor must be uint8/int8, got {packed.dtype}"
            )
        if encoded_scale.dtype != torch.uint8:
            raise TypeError(
                f"MXFP4 encoded scale must be uint8 E8M0, got {encoded_scale.dtype}"
            )
        from mlite_k3.primitive.mxfp4 import dequantize_mxfp4

        packed_i8 = packed if packed.dtype == torch.int8 else packed.view(torch.int8)
        return dequantize_mxfp4(
            packed_i8,
            encoded_scale.view(torch.float8_e8m0fnu),
        )

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        expected = self.weight_map().get(native_name)
        if expected is None:
            raise KeyError(f"unknown K3 native tensor {native_name!r}")
        if len(hf_tensors) != len(expected):
            raise ValueError(
                f"{native_name!r} requires {len(expected)} HF tensors, "
                f"got {len(hf_tensors)}"
            )
        layout = self._fusion_layouts.get(native_name)
        if self._uses_release_mxfp4 and self.is_expert(native_name):
            if len(hf_tensors) % 2:
                raise ValueError(f"{native_name!r} requires MXFP4 packed/scale pairs")
            self._raw_mxfp4_sources[native_name] = tuple(
                (source_name, tensor.detach())
                for source_name, tensor in zip(expected, hf_tensors, strict=True)
            )
            pairs = tuple(
                QuantizedWeight(packed, scale)
                for packed, scale in zip(
                    hf_tensors[0::2], hf_tensors[1::2], strict=True
                )
            )
            if layout is not None:
                return layout.fuse_quantized_ordered(
                    tuple(
                        zip(
                            (segment.name for segment in layout.segments),
                            pairs,
                            strict=True,
                        )
                    ),
                    materialize=lambda pair: self._materialize_mxfp4_pair(
                        pair.packed, pair.scale
                    ),
                )
            hf_tensors = [
                self._materialize_mxfp4_pair(pair.packed, pair.scale) for pair in pairs
            ]
        if layout is not None:
            return layout.fuse_ordered(
                tuple(
                    zip(
                        (segment.name for segment in layout.segments),
                        hf_tensors,
                        strict=True,
                    )
                )
            )
        tensor = hf_tensors[0]
        if native_name.endswith(".self_attention.A_log"):
            heads = int(getattr(self.config, "kda_num_heads", tensor.numel()))
            if tensor.ndim != 1 or tensor.numel() < heads:
                raise ValueError(
                    f"{native_name!r} must contain at least {heads} KDA heads, "
                    f"got shape {tuple(tensor.shape)}"
                )
            padding = tensor[heads:]
            if padding.numel() and torch.count_nonzero(padding).item():
                raise ValueError(
                    f"{native_name!r} A_log padding must be exactly zero, "
                    f"got {torch.count_nonzero(padding).item()} nonzero values"
                )
            tensor = tensor[:heads]
        elif native_name.endswith(".self_attention.dt_bias"):
            if not all(
                hasattr(self.config, name) for name in ("kda_num_heads", "kda_head_dim")
            ):
                return tensor
            heads = int(self.config.kda_num_heads)
            head_dim = int(self.config.kda_head_dim)
            expected = heads * head_dim
            if tensor.numel() != expected:
                raise ValueError(
                    f"{native_name!r} dt_bias must contain exactly {expected} values, "
                    f"got shape {tuple(tensor.shape)}"
                )
            tensor = tensor.reshape(heads, head_dim)
        if re.search(r"\.[qkv]_conv1d\.weight$", native_name):
            if tensor.ndim != 3 or tensor.size(1) != 1:
                raise ValueError(
                    f"{native_name!r} must have shape [channels, 1, kernel], "
                    f"got {tuple(tensor.shape)}"
                )
        return tensor

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        names = self.weight_map().get(native_name)
        if names is None:
            raise KeyError(f"unknown K3 native tensor {native_name!r}")
        names = list(
            dict.fromkeys(
                name.removesuffix("_packed").removesuffix("_scale") for name in names
            )
        )
        if native_name.endswith(".self_attention.A_log"):
            if not hasattr(self.config, "kda_num_heads"):
                return list(zip(names, (tensor,), strict=True))
            heads = int(self.config.kda_num_heads)
            if tensor.numel() != heads:
                raise ValueError(
                    f"{native_name!r} must contain exactly {heads} active heads, "
                    f"got shape {tuple(tensor.shape)}"
                )
            padded_heads = ((heads + 127) // 128) * 128
            tensor = torch.nn.functional.pad(
                tensor.reshape(-1),
                (0, padded_heads - heads),
            )
            parts = (tensor,)
        elif native_name.endswith(".self_attention.dt_bias"):
            if not all(
                hasattr(self.config, name) for name in ("kda_num_heads", "kda_head_dim")
            ):
                return list(zip(names, (tensor,), strict=True))
            heads = int(self.config.kda_num_heads)
            head_dim = int(self.config.kda_head_dim)
            if tuple(tensor.shape) != (heads, head_dim):
                raise ValueError(
                    f"{native_name!r} must have shape {(heads, head_dim)}, "
                    f"got {tuple(tensor.shape)}"
                )
            parts = (tensor.reshape(-1).contiguous(),)
        elif (layout := self._fusion_layouts.get(native_name)) is not None:
            split = layout.split(tensor)
            parts = tuple(split[segment.name] for segment in layout.segments)
        elif re.search(r"\.[qkv]_conv1d\.weight$", native_name):
            if tensor.ndim != 3 or tensor.size(1) != 1:
                raise ValueError(
                    f"{native_name!r} must have shape [channels, 1, kernel], "
                    f"got {tuple(tensor.shape)}"
                )
            parts = (tensor,)
        else:
            parts = (tensor,)
        return list(zip(names, parts, strict=True))

    def qkv_spec(self, native_name: str) -> None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        if self.is_expert(native_name):
            return (0, 1) if ".fc1." in native_name else (1, 1)
        if native_name in {
            "embed_tokens.embedding.weight",
            "lm_head.col.linear.weight",
        }:
            return (0, 0)
        if re.search(
            r"\.self_attention\."
            r"(q_proj|k_proj|v_proj|f_b_proj|b_proj|g_proj)\.linear\.weight$",
            native_name,
        ):
            return (0, 0)
        if re.search(
            r"\.self_attention\.(q|k|v)_conv1d\.weight$",
            native_name,
        ) or native_name.endswith((".self_attention.A_log", ".self_attention.dt_bias")):
            return (0, 0)
        if native_name.endswith(".self_attention.o_proj.linear.weight"):
            return (1, 0)
        if native_name.endswith(
            (
                ".self_attention.linear_q_up_proj.linear.weight",
                ".self_attention.linear_kv_up_proj.linear.weight",
                ".self_attention.linear_g_proj.linear.weight",
                ".mlp.gate_up.linear.weight",
                ".moe.shared_experts.gate_up.linear.weight",
            )
        ):
            return (0, 0)
        if native_name.endswith(
            (
                ".self_attention.linear_proj.linear.weight",
                ".mlp.down.linear.weight",
                ".moe.shared_experts.down.linear.weight",
            )
        ):
            return (1, 0)
        return None

    @staticmethod
    def is_expert(native_name: str) -> bool:
        return _GROUPED_EXPERT_WEIGHT.fullmatch(native_name) is not None

    def expert_global_id(self, native_name: str) -> int | None:
        match = _GROUPED_EXPERT_WEIGHT.fullmatch(native_name)
        return int(match.group(2)) if match is not None else None

    @staticmethod
    def expert_local_name(native_name: str, local_idx: int) -> str:
        match = _GROUPED_EXPERT_WEIGHT.fullmatch(native_name)
        if match is None:
            raise ValueError(f"{native_name!r} is not a K3 grouped-expert weight")
        return f"{match.group(1)}{local_idx}"


class _K3MXFP4CheckpointAdapter(nn.Module):
    """Model-owned raw release encoding used while its logical weight is unchanged."""

    def __init__(self, layer_indices: Iterable[int]):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.source_map: dict[str, str] = {}

    def add_source(
        self,
        *,
        global_name: str,
        local_name: str,
        source_name: str,
        tensor: torch.Tensor,
    ) -> None:
        module: nn.Module = self
        parts = local_name.split(".")
        for part in parts[:-1]:
            child = module._modules.get(part)
            if child is None:
                child = nn.Module()
                module.add_module(part, child)
            module = child
        module.register_buffer(parts[-1], tensor.detach(), persistent=True)
        self.source_map[global_name] = source_name


class _K3MXFP4CheckpointAdapterSpec:
    """Map model-owned raw buffers through MLite's ordinary ETP/EP/PP gather."""

    def __init__(self, source_map: Mapping[str, str], num_experts: int):
        self._source_map = dict(source_map)
        self.num_experts = int(num_experts)

    def weight_map(self) -> dict[str, list[str]]:
        return {native: [source] for native, source in self._source_map.items()}

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(hf_tensors) != 1:
            raise ValueError(f"{native_name!r} requires one raw MXFP4 component")
        return hf_tensors[0]

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        try:
            source_name = self._source_map[native_name]
        except KeyError as error:
            raise KeyError(
                f"unknown K3 MXFP4 adapter tensor {native_name!r}"
            ) from error
        return [(source_name, tensor)]

    @staticmethod
    def qkv_spec(native_name: str) -> None:
        del native_name
        return None

    @staticmethod
    def tp_spec(native_name: str) -> tuple[int, int]:
        return (1 if ".w2_" in native_name else 0, 1)

    @staticmethod
    def is_expert(native_name: str) -> bool:
        return _MXFP4_ADAPTER_EXPERT.fullmatch(native_name) is not None

    @staticmethod
    def expert_global_id(native_name: str) -> int | None:
        match = _MXFP4_ADAPTER_EXPERT.fullmatch(native_name)
        return int(match.group(2)) if match is not None else None

    @staticmethod
    def expert_local_name(native_name: str, local_idx: int) -> str:
        match = _MXFP4_ADAPTER_EXPERT.fullmatch(native_name)
        if match is None:
            raise ValueError(f"{native_name!r} is not a K3 MXFP4 adapter tensor")
        return f"{match.group(1)}{local_idx}"


def _split_mxfp4_source_for_etp(
    tensor: torch.Tensor,
    source_name: str,
    ps: Any,
) -> torch.Tensor:
    etp_size = int(getattr(ps, "etp_size", 1))
    if etp_size == 1:
        return tensor
    split_dim = 1 if ".w2.weight_" in source_name else 0
    if tensor.size(split_dim) % etp_size:
        raise ValueError(
            f"{source_name!r} dimension {split_dim}={tensor.size(split_dim)} "
            f"is not divisible by ETP={etp_size}"
        )
    return tensor.chunk(etp_size, dim=split_dim)[int(ps.etp_rank)].contiguous()


def _attach_mxfp4_checkpoint_adapter(
    model: nn.Module,
    spec: K3WeightSpec,
    ps: Any,
) -> None:
    """Attach the exact release pairs captured by ``spec.hf_to_native``."""
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

    base_model = unwrap_model(model)
    if hasattr(base_model, "_k3_mxfp4_checkpoint_adapter"):
        raise RuntimeError("K3 model already owns an MXFP4 checkpoint adapter")
    layer_indices = list(
        getattr(base_model, "layer_indices", range(spec.config.num_hidden_layers))
    )
    global_to_local = {
        global_index: local_index
        for local_index, global_index in enumerate(layer_indices)
    }
    adapter = _K3MXFP4CheckpointAdapter(layer_indices)
    experts_per_rank = spec.num_experts // int(getattr(ps, "ep_size", 1))
    expert_start = int(getattr(ps, "ep_rank", 0)) * experts_per_rank
    for native_name, sources in spec._raw_mxfp4_sources.items():
        match = _GROUPED_EXPERT_WEIGHT.fullmatch(native_name)
        if match is None:
            raise AssertionError(f"captured non-expert MXFP4 source {native_name!r}")
        layer_match = re.match(r"layers\.(\d+)\.", native_name)
        if layer_match is None:
            raise AssertionError(f"captured MXFP4 source without layer {native_name!r}")
        global_layer = int(layer_match.group(1))
        global_expert = int(match.group(2))
        if global_layer not in global_to_local:
            continue
        local_expert = global_expert - expert_start
        if not 0 <= local_expert < experts_per_rank:
            continue
        for source_name, tensor in sources:
            source_match = re.search(r"\.(w[123])\.weight_(packed|scale)$", source_name)
            if source_match is None:
                raise AssertionError(f"unexpected K3 MXFP4 source {source_name!r}")
            projection, component = source_match.groups()
            global_name = (
                f"layers.{global_layer}.moe.mxfp4.{projection}_{component}."
                f"weight{global_expert}"
            )
            local_name = (
                f"layers.{global_to_local[global_layer]}.moe.mxfp4."
                f"{projection}_{component}.weight{local_expert}"
            )
            adapter.add_source(
                global_name=global_name,
                local_name=local_name,
                source_name=source_name,
                tensor=_split_mxfp4_source_for_etp(tensor, source_name, ps),
            )
    if adapter.source_map:
        base_model.add_module("_k3_mxfp4_checkpoint_adapter", adapter)


def _k3_local_shape(
    native_name: str,
    config: Any,
    *,
    tp_size: int,
    etp_size: int,
) -> tuple[int, ...]:
    """Resolve a rank-local shape from exact declared state keys."""
    hidden = config.hidden_size
    vocab_divisor = math.lcm(128, tp_size)
    padded_vocab = math.ceil(config.vocab_size / vocab_divisor) * vocab_divisor
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.embedding.weight": (padded_vocab // tp_size, hidden),
        "lm_head.col.linear.weight": (padded_vocab // tp_size, hidden),
        "output_attn_res_norm.weight": (hidden,),
        "output_attn_res_proj.weight": (1, hidden),
        "norm.weight": (hidden,),
    }

    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attention_res_norm.weight",
            "mlp_res_norm.weight",
        ):
            shapes[f"{prefix}.{suffix}"] = (hidden,)
        for suffix in ("self_attention_res_proj.weight", "mlp_res_proj.weight"):
            shapes[f"{prefix}.{suffix}"] = (1, hidden)

        attention = f"{prefix}.self_attention"
        if config.attention_type(layer) == "kda":
            projection = config.kda_num_heads * config.kda_head_dim
            for segment in K3WeightSpec(config).kda_rollout_layout.segments:
                suffix = {
                    "q": "q_proj.linear.weight",
                    "k": "k_proj.linear.weight",
                    "v": "v_proj.linear.weight",
                    "g": "g_proj.linear.weight",
                    "f_a": "f_a_proj.weight",
                    "b": "b_proj.linear.weight",
                }[segment.name]
                rows = segment.local_rows(tp_size)
                shapes[f"{attention}.{suffix}"] = (rows, hidden)
            shapes.update(
                {
                    f"{attention}.f_b_proj.linear.weight": (
                        projection // tp_size,
                        config.kda_head_dim,
                    ),
                    f"{attention}.A_log": (config.kda_num_heads // tp_size,),
                    f"{attention}.dt_bias": (
                        config.kda_num_heads // tp_size,
                        config.kda_head_dim,
                    ),
                    f"{attention}.o_norm.weight": (config.kda_head_dim,),
                    f"{attention}.o_proj.linear.weight": (
                        hidden,
                        projection // tp_size,
                    ),
                }
            )
            for name in ("q", "k", "v"):
                shapes[f"{attention}.{name}_conv1d.weight"] = (
                    projection // tp_size,
                    1,
                    config.kda_short_conv_kernel_size,
                )
        else:
            q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            shapes.update(
                {
                    f"{attention}.linear_q_down_proj.weight": (
                        config.q_lora_rank,
                        hidden,
                    ),
                    f"{attention}.linear_q_up_proj.linear.layer_norm_weight": (
                        config.q_lora_rank,
                    ),
                    f"{attention}.linear_q_up_proj.linear.weight": (
                        config.num_attention_heads * q_head_dim // tp_size,
                        config.q_lora_rank,
                    ),
                    f"{attention}.linear_kv_down_proj.weight": (
                        config.kv_lora_rank + config.qk_rope_head_dim,
                        hidden,
                    ),
                    f"{attention}.linear_kv_up_proj.linear.layer_norm_weight": (
                        config.kv_lora_rank,
                    ),
                    f"{attention}.linear_kv_up_proj.linear.weight": (
                        config.num_attention_heads
                        * (config.qk_nope_head_dim + config.v_head_dim)
                        // tp_size,
                        config.kv_lora_rank,
                    ),
                    f"{attention}.linear_g_proj.linear.weight": (
                        config.num_attention_heads * config.v_head_dim // tp_size,
                        hidden,
                    ),
                    f"{attention}.linear_proj.linear.weight": (
                        hidden,
                        config.num_attention_heads * config.v_head_dim // tp_size,
                    ),
                }
            )

        if layer < config.first_k_dense_replace:
            shapes[f"{prefix}.mlp.gate_up.linear.weight"] = (
                2 * config.intermediate_size // tp_size,
                hidden,
            )
            shapes[f"{prefix}.mlp.down.linear.weight"] = (
                hidden,
                config.intermediate_size // tp_size,
            )
            continue

        moe = f"{prefix}.moe"
        shapes.update(
            {
                f"{moe}.router.gate.weight": (config.num_experts, hidden),
                f"{moe}.router.expert_bias": (config.num_experts,),
                f"{moe}.routed_expert_down_proj.weight": (
                    config.routed_expert_hidden_size,
                    hidden,
                ),
                f"{moe}.routed_expert_norm.weight": (config.routed_expert_hidden_size,),
                f"{moe}.routed_expert_up_proj.weight": (
                    hidden,
                    config.routed_expert_hidden_size,
                ),
                f"{moe}.shared_experts.gate_up.linear.weight": (
                    2 * config.shared_expert_intermediate_size // tp_size,
                    hidden,
                ),
                f"{moe}.shared_experts.down.linear.weight": (
                    hidden,
                    config.shared_expert_intermediate_size // tp_size,
                ),
            }
        )
        for expert in range(config.num_experts):
            shapes[f"{moe}.experts.fc1.weight{expert}"] = (
                2 * config.moe_intermediate_size // etp_size,
                config.routed_expert_hidden_size,
            )
            shapes[f"{moe}.experts.fc2.weight{expert}"] = (
                config.routed_expert_hidden_size,
                config.moe_intermediate_size // etp_size,
            )

    try:
        return shapes[native_name]
    except KeyError as error:
        raise KeyError(f"no K3 dry-run shape rule for {native_name!r}") from error


def plan_k3_rank_weights(
    spec: K3WeightSpec,
    index: Mapping[str, str],
    *,
    layer_indices: list[int],
    has_embed: bool,
    has_head: bool,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
    etp_size: int,
) -> tuple[K3RankWeight, ...]:
    """Select and shape-check one PP/TP/EP/ETP rank without opening shards."""
    config = spec.config
    if not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be within tp_size")
    if config.num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size")
    global_to_local = {
        global_index: local_index
        for local_index, global_index in enumerate(layer_indices)
    }
    experts_per_rank = config.num_experts // ep_size
    expert_start = ep_rank * experts_per_rank
    expert_stop = expert_start + experts_per_rank
    planned = []
    for global_name, hf_names in spec.weight_map().items():
        match = re.match(r"layers\.(\d+)(\..*)", global_name)
        if match is not None:
            global_layer = int(match.group(1))
            if global_layer not in global_to_local:
                continue
            native_name = f"layers.{global_to_local[global_layer]}{match.group(2)}"
        else:
            native_name = global_name
            if native_name.startswith("embed_tokens.") and not has_embed:
                continue
            if (
                native_name.startswith(("output_attn_res_", "norm.", "lm_head."))
                and not has_head
            ):
                continue
        expert_id = spec.expert_global_id(global_name)
        if expert_id is not None and not expert_start <= expert_id < expert_stop:
            continue
        if any(name not in index for name in hf_names):
            missing = next(name for name in hf_names if name not in index)
            raise KeyError(f"rank plan source {missing!r} is absent from index")
        if expert_id is not None:
            native_name = spec.expert_local_name(
                native_name,
                expert_id - expert_start,
            )
        dtype = (
            torch.float32
            if native_name.endswith(
                (
                    ".self_attention.A_log",
                    ".self_attention.dt_bias",
                    ".moe.router.expert_bias",
                )
            )
            else torch.bfloat16
        )
        planned.append(
            K3RankWeight(
                native_name=native_name,
                hf_names=tuple(hf_names),
                shape=_k3_local_shape(
                    global_name,
                    config,
                    tp_size=tp_size,
                    etp_size=etp_size,
                ),
                dtype=dtype,
            )
        )
    names = [item.native_name for item in planned]
    if len(names) != len(set(names)):
        raise RuntimeError("K3 rank plan contains duplicate native state keys")
    return tuple(planned)


def audit_k3_weight_spec_sources(
    spec: K3WeightSpec,
    index: Mapping[str, str],
) -> int:
    """Require every mapped text tensor before opening a release shard."""
    expected_sources = {
        name for source_names in spec.weight_map().values() for name in source_names
    }
    for name in sorted(expected_sources):
        if name in index:
            continue
        if f"{name}_packed" in index and f"{name}_scale" in index:
            continue
        raise ValueError(f"missing mapped K3 tensor {name!r}")

    expected = {
        name.removesuffix("_packed").removesuffix("_scale") for name in expected_sources
    }
    indexed_text = {
        name.removesuffix("_packed").removesuffix("_scale")
        for name in index
        if name.startswith("language_model.")
    }
    unexpected = sorted(indexed_text - expected)
    if unexpected:
        raise ValueError(f"unmapped K3 text tensor {unexpected[0]!r}")
    return len(expected)


def parse_k3_quantization_metadata(
    config: Mapping[str, Any],
) -> K3QuantizationMetadata:
    """Validate K3's mixed BF16/MXFP4 compressed-tensors declaration."""
    text_config = config.get("text_config", config)
    quantization = text_config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("text_config.quantization_config is required")
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("quant_method must be 'compressed-tensors'")
    format_name = quantization.get("format")
    if format_name != "mxfp4-pack-quantized":
        raise ValueError("format must be 'mxfp4-pack-quantized'")

    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("K3 requires exactly one compressed-tensors config group")
    group = next(iter(groups.values()))
    if not isinstance(group, Mapping):
        raise ValueError("K3 compressed-tensors config group must be a mapping")
    if group.get("targets") != ["Linear"]:
        raise ValueError("K3 MXFP4 config group must target Linear modules")
    weights = group.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("K3 MXFP4 weight metadata is required")
    expected = {
        "dynamic": False,
        "group_size": 32,
        "num_bits": 4,
        "scale_dtype": "torch.uint8",
        "symmetric": True,
        "type": "float",
    }
    mismatches = {
        key: (weights.get(key), value)
        for key, value in expected.items()
        if weights.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unexpected K3 MXFP4 weight metadata: {mismatches}")

    ignored = frozenset(quantization.get("ignore", ()))
    if ignored != _EXPECTED_MXFP4_IGNORES:
        missing = sorted(_EXPECTED_MXFP4_IGNORES - ignored)
        unexpected = sorted(ignored - _EXPECTED_MXFP4_IGNORES)
        raise ValueError(
            f"unexpected K3 MXFP4 ignore list: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return K3QuantizationMetadata(
        format=format_name,
        group_size=weights["group_size"],
        num_bits=weights["num_bits"],
        scale_dtype=weights["scale_dtype"],
        ignored_modules=ignored,
    )


def inspect_hf_checkpoint(path: str | Path) -> K3CheckpointManifest:
    """Inspect config and index before opening any 1.56-TB weight shard."""
    import json

    root = Path(path)
    with (root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (root / "model.safetensors.index.json").open(encoding="utf-8") as stream:
        index = json.load(stream)
    text_config = config.get("text_config", config)
    return K3CheckpointManifest(
        quantization=parse_k3_quantization_metadata(config),
        weights=audit_k3_weight_index(
            index,
            num_hidden_layers=int(text_config["num_hidden_layers"]),
            first_k_dense_replace=int(text_config["first_k_dense_replace"]),
            num_experts=int(text_config["num_experts"]),
        ),
    )


def _has(reader: Any, name: str) -> bool:
    index = getattr(reader, "index", None)
    if index is not None:
        return name in index
    try:
        reader.get_tensor(name)
    except (KeyError, FileNotFoundError):
        return False
    return True


def _dequantize_release_mxfp4(reader: Any, name: str) -> torch.Tensor:
    packed_name = f"{name}_packed"
    scale_name = f"{name}_scale"
    if not _has(reader, scale_name):
        raise KeyError(f"MXFP4 tensor {packed_name!r} is missing {scale_name!r}")

    packed = reader.get_tensor(packed_name)
    encoded_scale = reader.get_tensor(scale_name)
    if packed.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"{packed_name} must be uint8/int8, got {packed.dtype}")
    if encoded_scale.dtype != torch.uint8:
        raise TypeError(f"{scale_name} must be uint8 E8M0, got {encoded_scale.dtype}")

    from mlite_k3.primitive.mxfp4 import dequantize_mxfp4

    packed_i8 = packed if packed.dtype == torch.int8 else packed.view(torch.int8)
    scale_e8m0 = encoded_scale.view(torch.float8_e8m0fnu)
    return dequantize_mxfp4(packed_i8, scale_e8m0)


def get_hf_weight(reader: Any, name: str) -> torch.Tensor:
    """Read a BF16 tensor or materialize its public MXFP4 release pair."""
    if _has(reader, name):
        return reader.get_tensor(name)
    if _has(reader, f"{name}_packed"):
        return _dequantize_release_mxfp4(reader, name)
    raise KeyError(f"checkpoint tensor {name!r} was not found")


def _export_mxfp4_weights(
    weights: Iterator[tuple[str, torch.Tensor]],
):
    """Encode only public routed-expert weights from a gathered HF stream."""
    for hf_name, hf_tensor in weights:
        if not _ROUTED_MXFP4_WEIGHT.fullmatch(hf_name):
            yield hf_name, hf_tensor
            continue
        from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

        packed, scale = quantize_mxfp4(hf_tensor)
        yield f"{hf_name}_packed", packed
        yield f"{hf_name}_scale", scale.view(torch.uint8)


def _next_preserved_mxfp4_pair(
    weights: Iterator[tuple[str, torch.Tensor]],
    pending: dict[str, dict[str, torch.Tensor]],
    base_name: str,
) -> QuantizedWeight:
    while True:
        components = pending.get(base_name, {})
        if "packed" in components and "scale" in components:
            del pending[base_name]
            return QuantizedWeight(components["packed"], components["scale"])
        try:
            name, tensor = next(weights)
        except StopIteration as error:
            raise RuntimeError(
                f"model-owned MXFP4 adapter is missing {base_name!r}"
            ) from error
        component = next(
            (suffix for suffix in ("packed", "scale") if name.endswith(f"_{suffix}")),
            None,
        )
        if component is None:
            raise RuntimeError(
                f"model-owned MXFP4 adapter emitted invalid tensor {name!r}"
            )
        component_base = name[: -(len(component) + 1)]
        buffered = pending.setdefault(component_base, {})
        if component in buffered:
            raise RuntimeError(
                f"model-owned MXFP4 adapter emitted duplicate tensor {name!r}"
            )
        buffered[component] = tensor


def _export_with_preserved_mxfp4(
    logical_weights: Iterator[tuple[str, torch.Tensor]],
    preserved_weights: Iterator[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Reuse original pairs iff they still decode to the gathered model value."""
    from mlite_k3.primitive.mxfp4 import dequantize_mxfp4
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for hf_name, hf_tensor in logical_weights:
        if not _ROUTED_MXFP4_WEIGHT.fullmatch(hf_name):
            yield hf_name, hf_tensor
            continue
        preserved = _next_preserved_mxfp4_pair(preserved_weights, pending, hf_name)
        packed_i8 = (
            preserved.packed
            if preserved.packed.dtype == torch.int8
            else preserved.packed.view(torch.int8)
        )
        decoded = dequantize_mxfp4(
            packed_i8,
            preserved.scale.view(torch.float8_e8m0fnu),
        ).to(dtype=hf_tensor.dtype)
        if torch.equal(decoded, hf_tensor):
            packed, scale = preserved.packed, preserved.scale
        else:
            packed, scale = quantize_mxfp4(hf_tensor)
            scale = scale.view(torch.uint8)
        yield f"{hf_name}_packed", packed
        yield f"{hf_name}_scale", scale
    if pending:
        unexpected_name = sorted(pending)[0]
        raise RuntimeError(
            f"model-owned MXFP4 adapter has unexpected buffered tensor {unexpected_name!r}"
        )
    try:
        unexpected_name, _ = next(preserved_weights)
    except StopIteration:
        return
    raise RuntimeError(
        f"model-owned MXFP4 adapter has unexpected trailing tensor {unexpected_name!r}"
    )


def _mxfp4_checkpoint_adapters(
    model: torch.nn.Module | list[torch.nn.Module],
) -> list[_K3MXFP4CheckpointAdapter] | None:
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

    if isinstance(model, nn.ModuleList):
        chunks = list(model)
    elif isinstance(model, list):
        chunks = model
    else:
        chunks = [model]
    adapters = [
        getattr(unwrap_model(chunk), "_k3_mxfp4_checkpoint_adapter", None)
        for chunk in chunks
    ]
    if not any(adapter is not None for adapter in adapters):
        return None
    if any(adapter is None for adapter in adapters):
        raise RuntimeError(
            "K3 MXFP4 checkpoint adapter is missing from one model chunk"
        )
    return list(adapters)


def export_hf_weights(
    model: torch.nn.Module | list[torch.nn.Module],
    config: Any,
    ps: Any,
    *,
    target: str = "bf16",
    **kwargs,
):
    """Export full K3 HF tensors through MLite's shared distributed primitive."""
    from megatron.lite.primitive.ckpt.hf_weights import (
        export_hf_weights as export_with_primitive,
    )

    if target not in ("hf", "bf16", "mxfp4"):
        raise ValueError("target must be 'hf', 'bf16', or 'mxfp4'")
    requested_rank0_only = bool(kwargs.pop("rank0_only", False))
    weights = export_with_primitive(
        model,
        K3WeightSpec(config),
        ps,
        vocab_size=config.vocab_size,
        rank0_only=False,
        **kwargs,
    )
    if target in ("hf", "bf16"):
        for item in weights:
            if (
                not requested_rank0_only
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                yield item
        return
    adapters = _mxfp4_checkpoint_adapters(model)
    if adapters is None:
        yield from _export_mxfp4_weights(weights)
        return
    source_map = {
        native_name: source_name
        for adapter in adapters
        for native_name, source_name in adapter.source_map.items()
    }
    adapter_spec = _K3MXFP4CheckpointAdapterSpec(source_map, config.num_experts)
    adapter_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"cpu", "buffer_max_size_bytes"}
    }
    preserved = export_with_primitive(
        adapters,
        adapter_spec,
        ps,
        **adapter_kwargs,
    )
    exported = _export_with_preserved_mxfp4(iter(weights), iter(preserved))
    for item in exported:
        if (
            not requested_rank0_only
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            yield item


def _checkpoint_tensor(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Detach one export tensor into the contiguous CPU form safetensors needs."""
    return name, tensor.detach().to(device="cpu").contiguous()


def _next_save_group(
    weights: Iterator[tuple[str, torch.Tensor]], target: str
) -> list[tuple[str, torch.Tensor]]:
    """Keep a routed MXFP4 packed/scale pair together in its output shard."""
    name, tensor = next(weights)
    group = [_checkpoint_tensor(name, tensor)]
    if target != "mxfp4" or not name.endswith("_packed"):
        return group

    base = name.removesuffix("_packed")
    try:
        scale_name, scale = next(weights)
    except StopIteration as error:
        raise ValueError(f"MXFP4 tensor {name!r} is missing its scale") from error
    if scale_name != f"{base}_scale":
        raise ValueError(
            f"MXFP4 tensor {name!r} must be followed by {base + '_scale'!r}, "
            f"got {scale_name!r}"
        )
    group.append(_checkpoint_tensor(scale_name, scale))
    return group


def _write_json_atomically(path: Path, contents: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(contents, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_hf_weights(
    model: torch.nn.Module | list[torch.nn.Module],
    path: str | Path,
    config: Any,
    ps: Any,
    *,
    target: str = "bf16",
    max_shard_size_bytes: int = 5 * 1024**3,
) -> WeightIndexAudit:
    """Write a public K3 HF checkpoint as streamed safetensors shards.

    Each shard is atomically published before the index.  The index is itself
    atomically replaced only after every mapped shard exists, so readers never
    observe an index pointing at a partially written checkpoint.
    """
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive")
    if target not in ("bf16", "mxfp4"):
        raise ValueError("target must be 'bf16' or 'mxfp4'")

    from safetensors.torch import save_file

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    root = Path(path)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    total_size = 0
    shard_number = 0
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0

    def flush_shard() -> None:
        nonlocal shard, shard_number, shard_size
        if not shard:
            return
        shard_number += 1
        filename = f"model-{shard_number:05d}.safetensors"
        destination = root / filename
        temporary = root / f".{filename}.tmp"
        save_file(shard, str(temporary))
        os.replace(temporary, destination)
        index.update({name: filename for name in shard})
        shard = {}
        shard_size = 0

    weights = iter(
        export_hf_weights(
            model,
            config,
            ps,
            target=target,
            rank0_only=True,
            cpu=True,
        )
    )
    while True:
        try:
            group = _next_save_group(weights, target)
        except StopIteration:
            break
        group_size = sum(tensor.numel() * tensor.element_size() for _, tensor in group)
        if shard and shard_size + group_size > max_shard_size_bytes:
            flush_shard()
        shard.update(group)
        shard_size += group_size
        total_size += group_size
    flush_shard()

    result = None
    if rank == 0:
        audit_kwargs = (
            {
                "num_hidden_layers": int(config.num_hidden_layers),
                "first_k_dense_replace": int(config.first_k_dense_replace),
                "num_experts": int(config.num_experts),
            }
            if target == "mxfp4"
            else {}
        )
        result = audit_k3_weight_index(
            {"weight_map": index},
            **audit_kwargs,
        )
        _write_json_atomically(
            root / "model.safetensors.index.json",
            {
                "metadata": {"format": target, "total_size": total_size},
                "weight_map": index,
            },
        )
    if torch.distributed.is_initialized():
        payload = [result]
        torch.distributed.broadcast_object_list(payload, src=0)
        result = payload[0]
    assert result is not None
    return result


def load_hf_weights(
    model: torch.nn.Module,
    path: str | Path,
    config: Any,
    ps: Any,
) -> K3CheckpointManifest:
    """Validate metadata, then delegate PP/TP/EP/ETP loading to MLite."""
    manifest = inspect_hf_checkpoint(path)
    from megatron.lite.primitive.ckpt.hf_weights import (
        SafeTensorReader,
        load_hf_weights as load_with_primitive,
    )

    reader = SafeTensorReader(str(path))
    spec = K3WeightSpec(config, manifest=manifest)
    audit_k3_weight_spec_sources(spec, reader.index)
    load_with_primitive(
        model,
        str(path),
        spec,
        ps,
        vocab_size=config.vocab_size,
    )
    _attach_mxfp4_checkpoint_adapter(model, spec, ps)
    return manifest


def audit_k3_weight_index(
    index: Mapping[str, Any] | Mapping[str, str],
    *,
    num_hidden_layers: int | None = None,
    first_k_dense_replace: int | None = None,
    num_experts: int | None = None,
) -> WeightIndexAudit:
    """Validate K3's paired, shard-local routed-expert MXFP4 contract."""
    raw_weight_map = index.get("weight_map", index)
    if not isinstance(raw_weight_map, Mapping):
        raise TypeError("weight index must contain a mapping named 'weight_map'")
    weight_map = dict(raw_weight_map)

    quantized_keys = {
        key
        for key in weight_map
        if key.endswith(".weight_packed") or key.endswith(".weight_scale")
    }
    invalid = sorted(
        key for key in quantized_keys if not _ROUTED_MXFP4_KEY.fullmatch(key)
    )
    if invalid:
        raise ValueError(f"MXFP4 tensor outside routed experts: {invalid[0]!r}")

    packed_keys = sorted(key for key in quantized_keys if key.endswith("_packed"))
    for packed_key in packed_keys:
        scale_key = packed_key.removesuffix("_packed") + "_scale"
        if scale_key not in weight_map:
            raise ValueError(f"{packed_key!r} is missing weight_scale")
        if weight_map[packed_key] != weight_map[scale_key]:
            raise ValueError(
                f"{packed_key!r} and {scale_key!r} are stored in different shards"
            )

    unpaired_scales = sorted(
        scale_key
        for scale_key in quantized_keys
        if scale_key.endswith("_scale")
        and scale_key.removesuffix("_scale") + "_packed" not in weight_map
    )
    if unpaired_scales:
        raise ValueError(f"{unpaired_scales[0]!r} is missing weight_packed")

    coverage_shape = (num_hidden_layers, first_k_dense_replace, num_experts)
    if any(value is not None for value in coverage_shape):
        if not all(value is not None for value in coverage_shape):
            raise ValueError(
                "num_hidden_layers, first_k_dense_replace, and num_experts "
                "must be provided together"
            )
        expected_bases = {
            "language_model.model.layers."
            f"{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
            for layer in range(first_k_dense_replace, num_hidden_layers)
            for expert in range(num_experts)
            for projection in ("w1", "w2", "w3")
        }
        actual_bases = {key.removesuffix("_packed") for key in packed_keys}
        missing = sorted(expected_bases - actual_bases)
        unexpected = sorted(actual_bases - expected_bases)
        if missing or unexpected:
            detail = []
            if missing:
                detail.append(f"missing expected routed weight {missing[0]!r}")
            if unexpected:
                detail.append(f"unexpected routed weight {unexpected[0]!r}")
            raise ValueError("; ".join(detail))

    return WeightIndexAudit(
        quantized_weights=len(packed_keys),
        plain_tensors=len(weight_map) - 2 * len(packed_keys),
        shards=len(set(weight_map.values())),
    )


__all__ = [
    "K3CheckpointManifest",
    "K3QuantizationMetadata",
    "K3RankWeight",
    "K3WeightSpec",
    "audit_k3_weight_spec_sources",
    "WeightIndexAudit",
    "audit_k3_weight_index",
    "export_hf_weights",
    "get_hf_weight",
    "inspect_hf_checkpoint",
    "load_hf_weights",
    "parse_k3_quantization_metadata",
    "plan_k3_rank_weights",
    "save_hf_weights",
]
