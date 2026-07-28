"""Reusable Kimi Delta Attention primitive.

This module owns KDA math and backend selection. It intentionally accepts
explicit primitive parameters and has no dependency on a model configuration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlite_k3.norm import RMSNorm


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable recurrent KDA reference for small CPU correctness tests."""
    if q.shape != k.shape or q.shape != v.shape or q.shape != gate_logits.shape:
        raise ValueError("q, k, v, and gate_logits must have the same shape")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("beta_logits must have shape [batch, sequence, heads]")
    batch, sequence, heads, head_dim = q.shape
    if a_log.shape != (heads,) or dt_bias.shape != (heads, head_dim):
        raise ValueError("KDA A_log/dt_bias shapes do not match heads and head_dim")

    output_dtype = v.dtype
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    gate = lower_bound * torch.sigmoid(
        a_log.float().exp().view(1, 1, heads, 1)
        * (gate_logits.float() + dt_bias.float().view(1, 1, heads, head_dim))
    )
    beta = torch.sigmoid(beta_logits.float())
    state = (
        q.new_zeros(batch, heads, head_dim, head_dim)
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for index in range(sequence):
        state = state * gate[:, index].exp().unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k[:, index], state)
        delta = (v[:, index] - prediction) * beta[:, index].unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, index], delta)
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, index], state) * scale)
    return torch.stack(outputs, dim=1).to(output_dtype), state


def _fla_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    scale: float,
) -> torch.Tensor:
    """Run the KDA contract through FLA's trainable chunk kernel.

    FLA owns backend selection. In particular, this package never calls the
    forward-only FlashKDA extension directly.
    """
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as error:
        raise ImportError(
            "CUDA KDA requires flash-linear-attention; install the 'kda' extra"
        ) from error

    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate_logits,
        beta=beta_logits,
        scale=scale,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
    )
    return output


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

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        for parameter in (self.A_log, self.dt_bias):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        return module

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
        if x.is_cuda:
            output = _fla_chunk_kda(q, k, v, feature_gate, beta, **kda_kwargs)
        else:
            output, _ = kda_recurrent_reference(
                q, k, v, feature_gate, beta, **kda_kwargs
            )
        output_gate = torch.sigmoid(self._heads(self.g_proj(x)))
        output = self.o_norm(output) * output_gate
        return self.o_proj(output.flatten(-2))


__all__ = ["KDA", "kda_recurrent_reference"]
