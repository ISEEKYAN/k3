"""K3 full-rank KDA composed with MLite's existing GDN CP skeleton."""

from __future__ import annotations

import torch

from megatron.lite.primitive.modules.gated_delta_net import (
    FullRankGatedDeltaNet,
    GatedDeltaNet,
)
from megatron.lite.primitive.parallel.cp import get_parameter_local_cp_headwise
from megatron.lite.primitive.utils import ensure_divisible


class K3FullRankGatedDeltaNet(FullRankGatedDeltaNet):
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
        return self.o_proj(output.transpose(0, 1).contiguous())


__all__ = ["K3FullRankGatedDeltaNet"]
