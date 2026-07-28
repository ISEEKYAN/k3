"""Kimi K3 projections around the package-owned KDA operator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlite_k3.primitive.kda import (
    kda as run_kda,
    torch_recurrent_kda as kda_recurrent_reference,
)

from mlite_k3.norm import RMSNorm


class _CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2)
        values = F.pad(values, (self.kernel_size - 1, 0))
        return F.silu(self.conv(values).transpose(1, 2))


class KDA(nn.Module):
    """KDA projections, recurrence, full-rank output gate, and backend routing."""

    def __init__(
        self,
        *,
        hidden_size: int,
        heads: int,
        head_dim: int,
        short_conv_kernel_size: int,
        lower_bound: float,
        norm_eps: float,
        use_full_rank_gate: bool,
    ) -> None:
        super().__init__()
        projection_size = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.lower_bound = lower_bound
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.q_conv1d = _CausalDepthwiseConv1d(projection_size, short_conv_kernel_size)
        self.k_conv1d = _CausalDepthwiseConv1d(projection_size, short_conv_kernel_size)
        self.v_conv1d = _CausalDepthwiseConv1d(projection_size, short_conv_kernel_size)
        self.A_log = nn.Parameter(torch.zeros(heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, heads, bias=False)
        if not use_full_rank_gate:
            raise ValueError("KDA requires a full-rank output gate")
        self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.o_norm = RMSNorm(head_dim, norm_eps)
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.q_conv1d(self.q_proj(x)))
        k = self._heads(self.k_conv1d(self.k_proj(x)))
        v = self._heads(self.v_conv1d(self.v_proj(x)))
        feature_gate = self._heads(self.f_b_proj(self.f_a_proj(x)))
        beta = self.b_proj(x)
        kda_kwargs = {
            "a_log": self.A_log,
            "dt_bias": self.dt_bias.view(self.heads, self.head_dim),
            "lower_bound": self.lower_bound,
            "scale": self.head_dim**-0.5,
        }
        output, _ = run_kda(
            q,
            k,
            v,
            feature_gate,
            beta,
            output_final_state=False,
            backend="auto",
            **kda_kwargs,
        )
        output_gate = torch.sigmoid(self._heads(self.g_proj(x)))
        output = self.o_norm(output) * output_gate
        return self.o_proj(output.flatten(-2))


__all__ = ["KDA", "kda_recurrent_reference"]
