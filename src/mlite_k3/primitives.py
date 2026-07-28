"""Kimi K3 architecture primitives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlite_k3.config import K3Config
from mlite_k3.norm import RMSNorm


def situ_and_mul(
    gate_up: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activated * up).to(gate_up.dtype)


class K3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, config: K3Config):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(
            situ_and_mul(
                self.gate_up(x),
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        )


class GatedMultiLatentAttention(nn.Module):
    """NoPE MLA with the Kimi K3 per-head output gate."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        if not config.mla_use_nope or not config.mla_use_output_gate:
            raise ValueError("Kimi K3 requires NoPE MLA with output gating")
        self.num_heads = config.num_attention_heads
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        projection_size = self.num_heads * self.v_head_dim
        self.g_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        query = query.view(batch, sequence, self.num_heads, self.q_head_dim)
        compressed = self.kv_a_proj_with_mqa(x)
        kv_latent, key_tail = compressed.split(
            [self.kv_a_layernorm.weight.numel(), self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        kv = kv.view(
            batch,
            sequence,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_tail = key_tail[:, :, None, :].expand(
            batch, sequence, self.num_heads, self.qk_rope_head_dim
        )
        key = torch.cat([key_nope, key_tail], dim=-1)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            scale=self.q_head_dim**-0.5,
        )
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        output = output * torch.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class LatentMoE(nn.Module):
    """K3 latent-space routed experts plus two shared experts."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.scaling_factor = config.routed_scaling_factor
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # The public checkpoint treats e_score_correction_bias as router state,
        # not an optimizer parameter.  Keep it persistent so state_dict and
        # checkpoint streaming retain the exact routing placement.
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts, dtype=torch.float32)
        )
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size,
            config.routed_expert_hidden_size,
            bias=False,
        )
        self.experts = nn.ModuleList(
            [
                K3MLP(
                    config.routed_expert_hidden_size,
                    config.moe_intermediate_size,
                    config,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.routed_expert_norm = RMSNorm(
            config.routed_expert_hidden_size, config.rms_norm_eps
        )
        self.routed_expert_up_proj = nn.Linear(
            config.routed_expert_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.shared_experts = K3MLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        logits = F.linear(flat.float(), self.router.weight.float())
        scores = torch.sigmoid(logits)
        selection = scores + self.expert_bias.to(scores.dtype)
        _, indices = torch.topk(selection, self.top_k, dim=-1, sorted=False)
        topk_scores = scores.gather(-1, indices)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
        topk_scores = topk_scores * self.scaling_factor

        latent = self.routed_expert_down_proj(flat)
        expert_outputs = torch.stack([expert(latent) for expert in self.experts], dim=1)
        routing = torch.zeros_like(scores).scatter(-1, indices, topk_scores)
        routed = torch.einsum(
            "te,ted->td", routing.to(expert_outputs.dtype), expert_outputs
        )
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        shared = self.shared_experts(flat)
        return (routed + shared).view(original_shape)


__all__ = [
    "GatedMultiLatentAttention",
    "K3MLP",
    "LatentMoE",
    "situ_and_mul",
]
