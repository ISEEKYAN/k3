"""Kimi K3 public-checkpoint loading helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch


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


def audit_k3_weight_index(
    index: Mapping[str, Any] | Mapping[str, str],
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

    return WeightIndexAudit(
        quantized_weights=len(packed_keys),
        plain_tensors=len(weight_map) - 2 * len(packed_keys),
        shards=len(set(weight_map.values())),
    )


__all__ = ["WeightIndexAudit", "audit_k3_weight_index", "get_hf_weight"]
