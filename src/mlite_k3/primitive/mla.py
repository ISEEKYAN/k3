"""K3 NoPE/output-gated MLA on MLite's shared parallel attention skeleton."""

from __future__ import annotations

import torch

from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    gather_from_sequence_parallel,
)

_PACKED_SEQ_FIELDS = (
    "qkv_format",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
)


class K3MultiLatentAttention(MultiLatentAttention):
    """Add K3's NoPE and output gate without changing the shared primitive."""

    def __init__(
        self,
        *,
        use_nope: bool = False,
        output_gate: bool = False,
        hidden_size: int,
        num_attention_heads: int,
        ps,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ps=ps,
            **kwargs,
        )
        self.use_nope = bool(use_nope)
        self.output_gate = bool(output_gate)
        self.linear_g_proj = (
            ColumnParallelLinear(
                hidden_size,
                num_attention_heads * self.v_head_dim,
                ps,
                bias=False,
            )
            if self.output_gate
            else None
        )

    def forward(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        q_compressed = self.linear_q_down_proj(x)
        kv_combined = self.linear_kv_down_proj(x)
        kv_compressed, k_pos_emb = kv_combined.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.ps.tp_size > 1:
            k_pos_emb = gather_from_sequence_parallel(k_pos_emb, self.ps)

        q_proj = self.linear_q_up_proj(q_compressed)
        q = q_proj.view(
            *q_proj.shape[:-1],
            self.num_heads_local,
            self.q_head_dim,
        )
        kv_proj = self.linear_kv_up_proj(kv_compressed)
        kv = kv_proj.view(
            *kv_proj.shape[:-1],
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        q_nope, q_pos = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        k_nope, value = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_pos = k_pos_emb.unsqueeze(-2)

        is_thd = packed_seq_params is not None
        if is_thd:
            q_nope = q_nope.squeeze(1)
            q_pos = q_pos.squeeze(1)
            k_nope = k_nope.squeeze(1)
            value = value.squeeze(1)
            k_pos = k_pos.squeeze(1)

        if not self.use_nope:
            q_pos, k_pos = self._apply_rope(q_pos, k_pos, packed_seq_params)
        if k_pos.dim() == q_nope.dim():
            k_pos = k_pos.expand(*q_nope.shape[:-1], self.qk_rope_head_dim)
        else:
            k_pos = k_pos.expand(-1, -1, self.num_heads_local, -1)
        query = torch.cat([q_nope, q_pos], dim=-1).contiguous()
        key = torch.cat([k_nope, k_pos], dim=-1).contiguous()
        value = value.contiguous()
        if self._query_scale != 1.0:
            query = query * self._query_scale

        if is_thd:
            if self._use_torch_core:
                out = self._torch_core_attention_thd(
                    query,
                    key,
                    value,
                    packed_seq_params=packed_seq_params,
                ).reshape(query.size(0), 1, -1)
            else:
                psp_kwargs = {
                    name: getattr(packed_seq_params, name)
                    for name in _PACKED_SEQ_FIELDS
                    if getattr(packed_seq_params, name, None) is not None
                }
                assert self.core_attn is not None
                out = self.core_attn(
                    query,
                    key,
                    value,
                    core_attention_bias_type="no_bias",
                    attn_mask_type="padding_causal",
                    **psp_kwargs,
                ).reshape(query.size(0), 1, -1)
        else:
            if self._use_torch_core:
                out = self._torch_core_attention(query, key, value)
            else:
                assert self.core_attn is not None
                out = self.core_attn(
                    query,
                    key,
                    value,
                    core_attention_bias_type="no_bias",
                )
            if out.dim() > x.dim():
                out = out.reshape(
                    *out.shape[:-2],
                    self.num_heads_local * self.v_head_dim,
                )
        if self.linear_g_proj is not None:
            gate = torch.sigmoid(self.linear_g_proj(x))
            if is_thd:
                gate = gate.reshape(out.shape)
            out = out * gate
        return self.linear_proj(out)


__all__ = ["K3MultiLatentAttention"]
