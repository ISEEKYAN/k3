"""Kimi K3 text-backbone model composition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlite_k3.config import K3Config
from mlite_k3.kda import KDA
from mlite_k3.norm import RMSNorm
from mlite_k3.primitives import (
    GatedMultiLatentAttention,
    K3MLP,
    LatentMoE,
)


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.square().mean(dim=-1, keepdim=True)
    keys = values_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(dim=-1).softmax(dim=-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)


class K3DecoderLayer(nn.Module):
    def __init__(self, config: K3Config, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_res_block_size = config.attn_res_block_size
        if config.attention_type(layer_index) == "kda":
            self.self_attention: nn.Module = KDA(
                hidden_size=config.hidden_size,
                heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                short_conv_kernel_size=config.kda_short_conv_kernel_size,
                lower_bound=config.kda_gate_lower_bound,
                norm_eps=config.rms_norm_eps,
                use_full_rank_gate=config.kda_use_full_rank_gate,
            )
        else:
            self.self_attention = GatedMultiLatentAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attention_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.moe: LatentMoE | None = (
            LatentMoE(config) if layer_index >= config.first_k_dense_replace else None
        )
        self.mlp: K3MLP | None = (
            None
            if self.moe is not None
            else K3MLP(config.hidden_size, config.intermediate_size, config)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, hidden = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.size(1):
            hidden_states = _apply_attention_residual(
                prefix_sum.reshape(-1, hidden),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch, sequence, hidden)
        if self.layer_index % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.reshape(-1, hidden).unsqueeze(1)],
                dim=1,
            )
            prefix_sum = None

        attention_output = self.self_attention(self.input_layernorm(hidden_states))
        prefix_sum = (
            attention_output if prefix_sum is None else prefix_sum + attention_output
        )
        mlp_input = _apply_attention_residual(
            prefix_sum.reshape(-1, hidden),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch, sequence, hidden)
        mlp_input = self.post_attention_layernorm(mlp_input)
        mlp_output = (
            self.moe(mlp_input) if self.moe is not None else self.mlp(mlp_input)
        )
        return prefix_sum + mlp_output, block_residual


class K3Model(nn.Module):
    """Text-only Kimi K3 model with explicit hybrid layer composition."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [K3DecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.output_attn_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        images=None,
    ) -> dict[str, torch.Tensor]:
        self.config.ensure_text_only_inputs(
            pixel_values=pixel_values,
            images=images,
        )
        hidden_states = self.embed_tokens(input_ids)
        batch, sequence, hidden = hidden_states.shape
        block_residual = hidden_states.new_zeros(batch * sequence, 0, hidden)
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, block_residual)
        hidden_states = _apply_attention_residual(
            hidden_states.reshape(-1, hidden),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch, sequence, hidden)
        logits = self.lm_head(self.norm(hidden_states))
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return output


__all__ = ["K3DecoderLayer", "K3Model"]
