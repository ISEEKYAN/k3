"""K3 full-rank KDA composed with MLite's existing GDN CP skeleton."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
)
from megatron.lite.primitive.parallel.cp import get_parameter_local_cp_headwise
from megatron.lite.primitive.utils import ensure_divisible

try:
    from fla.modules.convolution import (
        causal_conv1d as _fla_causal_conv1d,  # pyright: ignore[reportMissingImports]
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


class _K3FullRankDeltaNet(nn.Module):
    """K3-owned full-rank projections around the shared GDN parallel helpers."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        gate_lower_bound: float,
        ps: ParallelState,
        recurrence: Callable,
        deterministic: bool = False,
    ):
        super().__init__()
        self.ps = ps
        self.num_heads = num_heads
        self.num_heads_local = ensure_divisible(num_heads, ps.tp_size)
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.projection_size_local = self.num_heads_local * head_dim
        self.gate_lower_bound = float(gate_lower_bound)
        self.recurrence = recurrence
        self.deterministic = bool(deterministic)

        def column(in_features: int, out_features: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(in_features, out_features, ps, bias=False)

        self.q_proj = column(hidden_size, self.projection_size)
        self.k_proj = column(hidden_size, self.projection_size)
        self.v_proj = column(hidden_size, self.projection_size)
        self.q_conv1d = nn.Conv1d(
            self.projection_size_local,
            self.projection_size_local,
            conv_kernel_size,
            groups=self.projection_size_local,
            bias=False,
            padding=conv_kernel_size - 1,
        )
        self.k_conv1d = nn.Conv1d(
            self.projection_size_local,
            self.projection_size_local,
            conv_kernel_size,
            groups=self.projection_size_local,
            bias=False,
            padding=conv_kernel_size - 1,
        )
        self.v_conv1d = nn.Conv1d(
            self.projection_size_local,
            self.projection_size_local,
            conv_kernel_size,
            groups=self.projection_size_local,
            bias=False,
            padding=conv_kernel_size - 1,
        )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads_local, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(self.num_heads_local, head_dim, dtype=torch.float32)
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = column(head_dim, self.projection_size)
        self.b_proj = column(hidden_size, num_heads)
        self.g_proj = column(hidden_size, self.projection_size)
        self.o_norm = te.RMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.projection_size, hidden_size, ps, bias=False
        )

    def _conv(
        self,
        qkv: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = qkv.shape[1]
        weight = torch.cat(
            (self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight),
            dim=0,
        )
        if _HAS_FLA and (cu_seqlens is not None or not self.deterministic):
            qkv, _ = _fla_causal_conv1d(
                x=qkv,
                weight=weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=cu_seqlens,
            )
            return qkv
        if cu_seqlens is not None:
            raise NotImplementedError("Packed THD K3 convolution requires FLA.")
        output = F.conv1d(
            qkv.transpose(1, 2),
            weight=weight,
            bias=None,
            padding=self.q_conv1d.padding[0],
            groups=weight.shape[0],
        )
        return F.silu(output[:, :, :seq_len].transpose(1, 2))

    def _output_projection(
        self, output: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Restore the model activation dtype after KDA's FP32 math."""
        return self.o_proj(output.to(dtype=reference.dtype))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        del position_ids
        q = self.q_proj(x).transpose(0, 1).contiguous()
        k = self.k_proj(x).transpose(0, 1).contiguous()
        v = self.v_proj(x).transpose(0, 1).contiguous()
        cu_seqlens = (
            GatedDeltaNet._packed_cu_seqlens(packed_seq_params)
            if packed_seq_params is not None
            else None
        )
        q, k, v = self._conv(
            torch.cat((q, k, v), dim=-1),
            cu_seqlens=cu_seqlens,
        ).split(self.projection_size_local, dim=-1)
        shape = (*q.shape[:-1], self.num_heads_local, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        feature_gate = (
            self.f_b_proj(self.f_a_proj(x)).transpose(0, 1).contiguous().view(shape)
        )
        beta = self.b_proj(x).transpose(0, 1).contiguous().view(*q.shape[:-1])
        output, _ = self.recurrence(
            q,
            k,
            v,
            feature_gate,
            beta,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            output_final_state=False,
            scale=self.head_dim**-0.5,
            cu_seqlens=cu_seqlens,
            backend="auto",
        )
        gate = self.g_proj(x).transpose(0, 1).contiguous().view_as(output).sigmoid()
        output = self.o_norm(output) * gate
        output = output.flatten(-2).transpose(0, 1).contiguous()
        return self._output_projection(output, x)


class K3FullRankGatedDeltaNet(_K3FullRankDeltaNet):
    """Connect K3's full-rank KDA math to the shared headwise CP skeleton."""

    def __init__(self, *args, cp_mode: str = "headwise", **kwargs):
        super().__init__(*args, **kwargs)
        if cp_mode != "headwise":
            raise NotImplementedError(
                "K3 KDA currently validates only headwise context parallelism"
            )
        self.cp_mode = cp_mode
        self.conv_dim_local = 3 * self.projection_size_local
        if self.ps.cp_size > 1:
            ensure_divisible(self.num_heads_local, self.ps.cp_size)

    @property
    def conv1d(self):
        """Expose convolution metadata required by the shared GDN helper."""
        return self.q_conv1d

    def _qkvzba_sections(self) -> list[int]:
        projection = self.projection_size_local
        return [
            projection,
            projection,
            projection,
            projection,
            self.num_heads_local,
            projection,
        ]

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        sections = (
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
            self.f_b_proj(self.f_a_proj(x)),
            self.b_proj(x),
            self.g_proj(x),
        )
        return torch.cat(sections, dim=-1).transpose(0, 1).contiguous()

    def _conv_weights(self) -> torch.Tensor:
        return torch.cat(
            tuple(
                get_parameter_local_cp_headwise(
                    weight,
                    dim=0,
                    cp_size=self.ps.cp_size,
                    cp_rank=self.ps.cp_rank,
                )
                for weight in (
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                )
            ),
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        if self.ps.cp_size <= 1:
            return super().forward(
                x,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )
        del position_ids
        if self.ps.cp_group is None:
            raise RuntimeError("CP>1 requires ParallelState.cp_group.")

        is_packed = packed_seq_params is not None
        projected = self._project(x)
        if not is_packed and projected.shape[0] > 1:
            raise ValueError(
                "K3 headwise KDA with dense SBHD input requires micro_batch_size == 1"
            )
        cu_seqlens = (
            GatedDeltaNet._packed_cu_seqlens(packed_seq_params) if is_packed else None
        )
        projected, cu_seqlens = GatedDeltaNet._headwise_cp2hp(
            self,
            projected,
            cu_seqlens,
        )

        cp_div = self.ps.cp_size
        projection = self.projection_size_local // cp_div
        local_heads = self.num_heads_local // cp_div
        q, k, v, feature_gate, beta, output_gate = projected.split(
            [
                projection,
                projection,
                projection,
                projection,
                local_heads,
                projection,
            ],
            dim=-1,
        )
        batch, sequence = q.shape[:2]
        qkv = GatedDeltaNet._causal_conv1d(
            self,
            torch.cat((q, k, v), dim=-1),
            sequence,
            cu_seqlens=cu_seqlens,
            conv_weight=self._conv_weights(),
            cp_div=cp_div,
            cp_context=None,
        )
        q, k, v = qkv.split(projection, dim=-1)
        shape = (batch, sequence, local_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        feature_gate = feature_gate.view(shape)
        beta = beta.view(batch, sequence, local_heads)
        output_gate = output_gate.view(shape)
        a_log = get_parameter_local_cp_headwise(
            self.A_log,
            dim=0,
            cp_size=self.ps.cp_size,
            cp_rank=self.ps.cp_rank,
        )
        dt_bias = get_parameter_local_cp_headwise(
            self.dt_bias,
            dim=0,
            cp_size=self.ps.cp_size,
            cp_rank=self.ps.cp_rank,
        )
        output, _ = self.recurrence(
            q,
            k,
            v,
            feature_gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias,
            lower_bound=self.gate_lower_bound,
            output_final_state=False,
            scale=self.head_dim**-0.5,
            cu_seqlens=cu_seqlens,
            backend="auto",
        )
        output = self.o_norm(output) * output_gate.sigmoid()
        output = GatedDeltaNet._headwise_hp2cp(
            self,
            output.flatten(-2),
            cu_seqlens,
        )
        output = output.transpose(0, 1).contiguous()
        return self._output_projection(output, x)


__all__ = ["K3FullRankGatedDeltaNet"]
