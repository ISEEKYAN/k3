"""MXFP4 E2M1 + UE8M0 checkpoint dequantization."""

from __future__ import annotations

import torch

MXFP4_BLOCK_SIZE = 32
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize checkpoint-format MXFP4 for CPU validation."""
    if packed.dtype != torch.int8:
        raise TypeError(f"MXFP4 packed tensor must be int8, got {packed.dtype}")
    expected = (*packed.shape[:-1], packed.shape[-1] * 2 // MXFP4_BLOCK_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected {expected}"
        )
    table = torch.tensor(
        (*_E2M1_POSITIVE, *(value * -1.0 for value in _E2M1_POSITIVE)),
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    values = torch.stack(
        (table[(raw & 0x0F).long()], table[(raw >> 4).long()]),
        dim=-1,
    )
    values = values.flatten(-2)
    expanded_scale = scale.float().repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return values * expanded_scale


__all__ = ["MXFP4_BLOCK_SIZE", "dequantize_mxfp4"]
