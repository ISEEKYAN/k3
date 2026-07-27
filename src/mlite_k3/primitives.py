"""Kimi K3 architecture primitives.

The CPU KDA recurrence follows the public Moonshot FlashKDA torch reference.
The CUDA production backend can replace this bounded reference without changing
the model composition API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlite_k3.config import K3Config


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
    """Differentiable recurrent KDA reference for small CPU correctness tests.

    Inputs use ``[batch, sequence, heads, head_dim]``. This intentionally favors
    clarity over throughput and must not be used for full-scale training.
    """
    if q.shape != k.shape or q.shape != v.shape or q.shape != gate_logits.shape:
        raise ValueError("q, k, v, and gate_logits must have the same shape")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("beta_logits must have shape [batch, sequence, heads]")
    batch, sequence, heads, head_dim = q.shape
    if a_log.shape != (heads,) or dt_bias.shape != (heads, head_dim):
        raise ValueError("KDA A_log/dt_bias shapes do not match heads and head_dim")

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
    return torch.stack(outputs, dim=1).to(v.dtype), state


class _CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2)
        values = F.pad(values, (self.kernel_size - 1, 0))
        return F.silu(self.conv(values).transpose(1, 2))


class KimiDeltaAttention(nn.Module):
    """KDA projection stack with a CPU reference recurrent backend."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        heads = config.kda_num_heads
        head_dim = config.kda_head_dim
        projection_size = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.lower_bound = config.kda_gate_lower_bound
        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.q_conv1d = _CausalDepthwiseConv1d(
            projection_size, config.kda_short_conv_kernel_size
        )
        self.k_conv1d = _CausalDepthwiseConv1d(
            projection_size, config.kda_short_conv_kernel_size
        )
        self.v_conv1d = _CausalDepthwiseConv1d(
            projection_size, config.kda_short_conv_kernel_size
        )
        self.A_log = nn.Parameter(torch.zeros(heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.f_a_proj = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, heads, bias=False)
        if not config.kda_use_full_rank_gate:
            raise ValueError("Kimi K3 requires a full-rank KDA output gate")
        self.g_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.o_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.q_conv1d(self.q_proj(x)))
        k = self._heads(self.k_conv1d(self.k_proj(x)))
        v = self._heads(self.v_conv1d(self.v_proj(x)))
        feature_gate = self._heads(self.f_b_proj(self.f_a_proj(x)))
        beta = self.b_proj(x)
        output, _ = kda_recurrent_reference(
            q,
            k,
            v,
            feature_gate,
            beta,
            a_log=self.A_log,
            dt_bias=self.dt_bias.view(self.heads, self.head_dim),
            lower_bound=self.lower_bound,
            scale=self.head_dim**-0.5,
        )
        output_gate = torch.sigmoid(self._heads(self.g_proj(x)))
        output = self.o_norm(output) * output_gate
        return self.o_proj(output.flatten(-2))


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
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
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
        logits = self.router(flat.float())
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
    "KimiDeltaAttention",
    "LatentMoE",
    "RMSNorm",
    "kda_recurrent_reference",
    "situ_and_mul",
]
