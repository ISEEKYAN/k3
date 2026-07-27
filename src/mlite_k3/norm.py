"""Small normalization primitives shared by model components."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(dtype)


__all__ = ["RMSNorm"]
