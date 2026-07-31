"""Kimi K3 public-checkpoint loading helpers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


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
                mapping[f"{native}.mlp.gate_up.linear.weight"] = [
                    f"{hf}.mlp.gate_proj.weight",
                    f"{hf}.mlp.up_proj.weight",
                ]
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
        for expert in range(self.config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert}"
            mapping[f"{native}.moe.experts.fc1.weight{expert}"] = self._hf_sources(
                f"{expert_prefix}.w1.weight",
                f"{expert_prefix}.w3.weight",
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
        if self._uses_release_mxfp4 and self.is_expert(native_name):
            if len(hf_tensors) % 2:
                raise ValueError(f"{native_name!r} requires MXFP4 packed/scale pairs")
            hf_tensors = [
                self._materialize_mxfp4_pair(packed, scale)
                for packed, scale in zip(
                    hf_tensors[0::2],
                    hf_tensors[1::2],
                    strict=True,
                )
            ]
        if ".gate_up." in native_name or ".fc1." in native_name:
            return torch.cat(hf_tensors, dim=0).contiguous()
        tensor = hf_tensors[0]
        if native_name.endswith(".self_attention.A_log"):
            heads = int(self.config.kda_num_heads)
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
            heads = int(self.config.kda_num_heads)
            head_dim = int(self.config.kda_head_dim)
            if tuple(tensor.shape) != (heads, head_dim):
                raise ValueError(
                    f"{native_name!r} must have shape {(heads, head_dim)}, "
                    f"got {tuple(tensor.shape)}"
                )
            parts = (tensor.reshape(-1).contiguous(),)
        elif ".gate_up." in native_name or ".fc1." in native_name:
            parts = tensor.chunk(2, dim=0)
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


def _k3_local_shape(
    native_name: str,
    config: Any,
    *,
    tp_size: int,
    etp_size: int,
) -> tuple[int, ...]:
    hidden = config.hidden_size
    if native_name in {
        "embed_tokens.embedding.weight",
        "lm_head.col.linear.weight",
    }:
        vocab_divisor = math.lcm(128, tp_size)
        padded_vocab = math.ceil(config.vocab_size / vocab_divisor) * vocab_divisor
        return (padded_vocab // tp_size, hidden)
    if native_name in {
        "output_attn_res_norm.weight",
        "norm.weight",
    } or native_name.endswith(
        (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".self_attention_res_norm.weight",
            ".mlp_res_norm.weight",
        )
    ):
        return (hidden,)
    if native_name == "output_attn_res_proj.weight" or native_name.endswith(
        (".self_attention_res_proj.weight", ".mlp_res_proj.weight")
    ):
        return (1, hidden)

    if ".self_attention." in native_name:
        suffix = native_name.split(".self_attention.", 1)[1]
        kda_projection = config.kda_num_heads * config.kda_head_dim
        if suffix in {
            "q_proj.linear.weight",
            "k_proj.linear.weight",
            "v_proj.linear.weight",
            "g_proj.linear.weight",
        }:
            return (kda_projection // tp_size, hidden)
        if suffix == "f_a_proj.weight":
            return (config.kda_head_dim, hidden)
        if suffix == "f_b_proj.linear.weight":
            return (kda_projection // tp_size, config.kda_head_dim)
        if suffix == "b_proj.linear.weight":
            return (config.kda_num_heads // tp_size, hidden)
        if suffix in {"A_log", "dt_bias"}:
            tail = () if suffix == "A_log" else (config.kda_head_dim,)
            return (config.kda_num_heads // tp_size, *tail)
        if re.fullmatch(r"[qkv]_conv1d\.weight", suffix):
            return (
                kda_projection // tp_size,
                1,
                config.kda_short_conv_kernel_size,
            )
        if suffix == "o_norm.weight":
            return (config.kda_head_dim,)
        if suffix == "o_proj.linear.weight":
            return (hidden, kda_projection // tp_size)

        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        if suffix == "linear_q_down_proj.weight":
            return (config.q_lora_rank, hidden)
        if suffix == "linear_q_up_proj.linear.layer_norm_weight":
            return (config.q_lora_rank,)
        if suffix == "linear_q_up_proj.linear.weight":
            return (
                config.num_attention_heads * q_head_dim // tp_size,
                config.q_lora_rank,
            )
        if suffix == "linear_kv_down_proj.weight":
            return (config.kv_lora_rank + config.qk_rope_head_dim, hidden)
        if suffix == "linear_kv_up_proj.linear.layer_norm_weight":
            return (config.kv_lora_rank,)
        if suffix == "linear_kv_up_proj.linear.weight":
            return (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim)
                // tp_size,
                config.kv_lora_rank,
            )
        if suffix == "linear_g_proj.linear.weight":
            return (
                config.num_attention_heads * config.v_head_dim // tp_size,
                hidden,
            )
        if suffix == "linear_proj.linear.weight":
            return (
                hidden,
                config.num_attention_heads * config.v_head_dim // tp_size,
            )

    if native_name.endswith(".mlp.gate_up.linear.weight"):
        return (2 * config.intermediate_size // tp_size, hidden)
    if native_name.endswith(".mlp.down.linear.weight"):
        return (hidden, config.intermediate_size // tp_size)
    if native_name.endswith(".moe.router.gate.weight"):
        return (config.num_experts, hidden)
    if native_name.endswith(".moe.router.expert_bias"):
        return (config.num_experts,)
    if native_name.endswith(".moe.routed_expert_down_proj.weight"):
        return (config.routed_expert_hidden_size, hidden)
    if native_name.endswith(".moe.routed_expert_norm.weight"):
        return (config.routed_expert_hidden_size,)
    if native_name.endswith(".moe.routed_expert_up_proj.weight"):
        return (hidden, config.routed_expert_hidden_size)
    if native_name.endswith(".moe.shared_experts.gate_up.linear.weight"):
        return (2 * config.shared_expert_intermediate_size // tp_size, hidden)
    if native_name.endswith(".moe.shared_experts.down.linear.weight"):
        return (hidden, config.shared_expert_intermediate_size // tp_size)
    if ".moe.experts.fc1." in native_name:
        return (
            2 * config.moe_intermediate_size // etp_size,
            config.routed_expert_hidden_size,
        )
    if ".moe.experts.fc2." in native_name:
        return (
            config.routed_expert_hidden_size,
            config.moe_intermediate_size // etp_size,
        )
    raise KeyError(f"no K3 dry-run shape rule for {native_name!r}")


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
                    native_name,
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
    weights = export_with_primitive(
        model,
        K3WeightSpec(config),
        ps,
        vocab_size=config.vocab_size,
        **kwargs,
    )
    if target in ("hf", "bf16"):
        yield from weights
        return
    yield from _export_mxfp4_weights(weights)


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
