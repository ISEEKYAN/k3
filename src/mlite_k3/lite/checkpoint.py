"""Kimi K3 public-checkpoint loading helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


class K3WeightSpec:
    """Exact Kimi K3 text-backbone mapping for the public release."""

    def __init__(self, config: Any):
        self.config = config

    def weight_map(self) -> dict[str, list[str]]:
        mapping = {
            "embed_tokens.weight": ["language_model.model.embed_tokens.weight"],
            "output_attn_res_norm.weight": [
                "language_model.model.output_attn_res_norm.weight"
            ],
            "output_attn_res_proj.weight": [
                "language_model.model.output_attn_res_proj.weight"
            ],
            "norm.weight": ["language_model.model.norm.weight"],
            "lm_head.weight": ["language_model.lm_head.weight"],
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
                mapping[f"{native}.mlp.gate_up.weight"] = [
                    f"{hf}.mlp.gate_proj.weight",
                    f"{hf}.mlp.up_proj.weight",
                ]
                mapping[f"{native}.mlp.down.weight"] = [f"{hf}.mlp.down_proj.weight"]
            else:
                self._add_moe(mapping, native, hf)
        return mapping

    @staticmethod
    def _add_kda(mapping: dict[str, list[str]], native: str, hf: str) -> None:
        for name in (
            "A_log",
            "dt_bias",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "b_proj.weight",
            "g_proj.weight",
            "o_norm.weight",
            "o_proj.weight",
        ):
            mapping[f"{native}.self_attention.{name}"] = [f"{hf}.self_attn.{name}"]
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            mapping[f"{native}.self_attention.{name}.conv.weight"] = [
                f"{hf}.self_attn.{name}.weight"
            ]

    @staticmethod
    def _add_mla(mapping: dict[str, list[str]], native: str, hf: str) -> None:
        for name in (
            "q_a_proj.weight",
            "q_a_layernorm.weight",
            "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight",
            "kv_a_layernorm.weight",
            "kv_b_proj.weight",
            "g_proj.weight",
            "o_proj.weight",
        ):
            mapping[f"{native}.self_attention.{name}"] = [f"{hf}.self_attn.{name}"]

    def _add_moe(self, mapping: dict[str, list[str]], native: str, hf: str) -> None:
        prefix = f"{hf}.block_sparse_moe"
        fixed = {
            "expert_bias": "gate.e_score_correction_bias",
            "router.weight": "gate.weight",
            "routed_expert_down_proj.weight": "routed_expert_down_proj.weight",
            "routed_expert_norm.weight": "routed_expert_norm.weight",
            "routed_expert_up_proj.weight": "routed_expert_up_proj.weight",
            "shared_experts.gate_up.weight": (
                "shared_experts.gate_proj.weight",
                "shared_experts.up_proj.weight",
            ),
            "shared_experts.down.weight": "shared_experts.down_proj.weight",
        }
        for native_suffix, hf_suffixes in fixed.items():
            if isinstance(hf_suffixes, str):
                hf_suffixes = (hf_suffixes,)
            mapping[f"{native}.moe.{native_suffix}"] = [
                f"{prefix}.{suffix}" for suffix in hf_suffixes
            ]
        for expert in range(self.config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert}"
            mapping[f"{native}.moe.experts.{expert}.gate_up.weight"] = [
                f"{expert_prefix}.w1.weight",
                f"{expert_prefix}.w3.weight",
            ]
            mapping[f"{native}.moe.experts.{expert}.down.weight"] = [
                f"{expert_prefix}.w2.weight"
            ]

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
        if native_name.endswith(".gate_up.weight"):
            return torch.cat(hf_tensors, dim=0).contiguous()
        tensor = hf_tensors[0]
        if re.search(r"\.[qkv]_conv1d\.conv\.weight$", native_name):
            return tensor.unsqueeze(1)
        return tensor

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        names = self.weight_map().get(native_name)
        if names is None:
            raise KeyError(f"unknown K3 native tensor {native_name!r}")
        if native_name.endswith(".gate_up.weight"):
            parts = tensor.chunk(2, dim=0)
        elif re.search(r"\.[qkv]_conv1d\.conv\.weight$", native_name):
            if tensor.ndim != 3 or tensor.size(1) != 1:
                raise ValueError(
                    f"{native_name!r} must have shape [channels, 1, kernel]"
                )
            parts = (tensor.squeeze(1),)
        else:
            parts = (tensor,)
        return list(zip(names, parts, strict=True))


def audit_k3_weight_spec_sources(
    spec: K3WeightSpec,
    index: Mapping[str, str],
) -> int:
    """Require every mapped text tensor before opening a release shard."""
    expected = {
        name for source_names in spec.weight_map().values() for name in source_names
    }
    for name in sorted(expected):
        if name in index:
            continue
        if f"{name}_packed" in index and f"{name}_scale" in index:
            continue
        raise ValueError(f"missing mapped K3 tensor {name!r}")

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

    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    base = model
    seen: set[int] = set()
    while isinstance(getattr(base, "module", None), torch.nn.Module):
        if id(base) in seen:
            break
        seen.add(id(base))
        base = base.module
    return base


def load_weights_from_reader(
    model: torch.nn.Module,
    reader: Any,
    spec: K3WeightSpec,
) -> int:
    """Stream all native parameters from an already indexed reader."""
    base = _unwrap_model(model)
    mapping = spec.weight_map()
    loaded = 0
    with torch.no_grad():
        for native_name, parameter in base.named_parameters():
            hf_names = mapping.get(native_name)
            if hf_names is None:
                raise KeyError(
                    f"native parameter {native_name!r} has no K3 checkpoint mapping"
                )
            hf_tensors = [get_hf_weight(reader, name) for name in hf_names]
            tensor = spec.hf_to_native(native_name, hf_tensors)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"shape mismatch for {native_name!r}: checkpoint "
                    f"{tuple(tensor.shape)} != model {tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded += 1
            del tensor, hf_tensors
    return loaded


def load_hf_weights(
    model: torch.nn.Module,
    path: str | Path,
    config: Any,
) -> K3CheckpointManifest:
    """Validate metadata first, then stream the public checkpoint tensor-wise."""
    manifest = inspect_hf_checkpoint(path)
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    # Deliberately use the one-shot reader API: each tensor's shard mapping is
    # closed before the next tensor, so a 96-shard release is never retained by
    # one process as a collection of open mmap handles.
    reader = SafeTensorReader(str(path))
    spec = K3WeightSpec(config)
    audit_k3_weight_spec_sources(spec, reader.index)
    load_weights_from_reader(model, reader, spec)
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
    "K3WeightSpec",
    "audit_k3_weight_spec_sources",
    "WeightIndexAudit",
    "audit_k3_weight_index",
    "get_hf_weight",
    "inspect_hf_checkpoint",
    "load_hf_weights",
    "load_weights_from_reader",
    "parse_k3_quantization_metadata",
]
