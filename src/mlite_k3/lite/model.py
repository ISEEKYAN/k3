"""Megatron Lite Kimi K3 model composition."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.attention import MultiLatentAttention
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts, _AllReduceETP
from megatron.lite.primitive.modules.kda import KimiDeltaAttention
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import (
    vocab_parallel_cross_entropy,
)
from megatron.lite.primitive.ops.sp_ops import ScatterToSP
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)

from mlite_k3.config import K3Config


def situ_and_mul(
    gate_up: torch.Tensor,
    probabilities: torch.Tensor | None,
    *,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate_float = gate.float()
    up_float = up.float()
    activated = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    up_float = linear_beta * torch.tanh(up_float / linear_beta)
    output = activated * up_float
    if probabilities is not None:
        output = output * probabilities.float()
    return output.to(gate_up.dtype)


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.square().mean(dim=-1, keepdim=True)
    keys = values_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(dim=-1).softmax(dim=-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)


class K3DenseMLP(nn.Module):
    def __init__(self, config: K3Config, ps: ParallelState) -> None:
        super().__init__()
        self.gate_up = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            ps,
            bias=False,
        )
        self.down = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            ps,
            bias=False,
        )
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(
            situ_and_mul(
                self.gate_up(x),
                None,
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        )


class _LocalLinear(nn.Module):
    """A TP-local projection used by the shared-expert column/row pair."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class K3LatentMoE(nn.Module):
    """K3 latent bottleneck around shared router/dispatcher/GroupedLinear."""

    def __init__(self, config: K3Config, ps: ParallelState) -> None:
        super().__init__()
        self.ps = ps
        self.hidden_size = config.hidden_size
        self.latent_size = config.routed_expert_hidden_size
        router_config = SimpleNamespace(
            hidden_size=config.hidden_size,
            num_experts_per_tok=config.num_experts_per_token,
            n_routed_experts=config.num_experts,
            aux_loss_alpha=0.0,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func=config.moe_router_activation_func,
        )
        self.router = SigmoidTopKRouter(
            router_config,
            ps,
            compute_aux_loss=False,
        )
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size,
            config.routed_expert_hidden_size,
            bias=False,
        )
        expert_config = SimpleNamespace(
            hidden_size=config.routed_expert_hidden_size,
            num_experts=config.num_experts,
            moe_intermediate_size=config.moe_intermediate_size,
            swiglu_limit=0.0,
        )

        def activation(values, probabilities):
            return situ_and_mul(
                values,
                probabilities,
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
            )

        self.experts = Experts(
            expert_config,
            ps,
            use_tp_for_experts=ps.ep_size == 1,
            activation=activation,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.routed_expert_hidden_size,
            ps,
            use_deepep=False,
        )
        self.routed_expert_norm = te.RMSNorm(
            config.routed_expert_hidden_size,
            eps=config.rms_norm_eps,
        )
        self.routed_expert_up_proj = nn.Linear(
            config.routed_expert_hidden_size,
            config.hidden_size,
            bias=False,
        )

        shared_intermediate = config.shared_expert_intermediate_size
        self.shared_gate_up = _LocalLinear(
            config.hidden_size,
            shared_intermediate * 2 // ps.tp_size,
        )
        self.shared_down = _LocalLinear(
            shared_intermediate // ps.tp_size,
            config.hidden_size,
        )
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        partial = self.shared_down(
            situ_and_mul(
                self.shared_gate_up(x),
                None,
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        )
        if self.ps.tp_size > 1:
            partial = _AllReduceETP.apply(partial, self.ps.tp_group)
        return partial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        full_x = gather_from_sequence_parallel(x, self.ps)
        flat_x = full_x.reshape(-1, self.hidden_size)
        scores, indices = self.router(flat_x)
        latent = self.routed_expert_down_proj(flat_x)
        dispatched, tokens_per_expert, probabilities = self.dispatcher.dispatch(
            latent,
            scores,
            indices,
        )
        self.dispatcher.wait_dispatch_event()
        routed = self.experts(
            dispatched,
            tokens_per_expert,
            probabilities,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed = self.dispatcher.combine(routed)
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        output = routed + self._shared_expert(flat_x)
        output = output.view(full_x.shape)
        if self.ps.tp_size > 1:
            # Both expert branches have already reduced their TP-partial
            # intermediate dimension, so each rank holds the complete output.
            # Restoring sequence parallel is a shard, not another sum.
            output = ScatterToSP.apply(
                output,
                self.ps.tp_size,
                self.ps.tp_rank,
                self.ps.tp_group,
            )
        if output.shape != input_shape:
            raise RuntimeError(
                f"K3 MoE restored {tuple(output.shape)}, expected {tuple(input_shape)}"
            )
        return output


class K3ParallelDecoderLayer(nn.Module):
    def __init__(
        self,
        config: K3Config,
        ps: ParallelState,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_res_block_size = config.attn_res_block_size
        if config.attention_type(layer_index) == "kda":
            self.self_attention: nn.Module = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                short_conv_kernel_size=config.kda_short_conv_kernel_size,
                gate_lower_bound=config.kda_gate_lower_bound,
                rms_norm_eps=config.rms_norm_eps,
                ps=ps,
                deterministic=True,
            )
        else:
            self.self_attention = MultiLatentAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                ps=ps,
                rms_norm_eps=config.rms_norm_eps,
                mla_use_nope=config.mla_use_nope,
                mla_use_output_gate=config.mla_use_output_gate,
            )
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attention_res_norm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.moe = (
            K3LatentMoE(config, ps)
            if layer_index >= config.first_k_dense_replace
            else None
        )
        self.mlp = K3DenseMLP(config, ps) if self.moe is None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, batch, hidden = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.size(1):
            hidden_states = _apply_attention_residual(
                prefix_sum.reshape(-1, hidden),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(sequence, batch, hidden)
        if self.layer_index % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    prefix_sum.reshape(-1, hidden).unsqueeze(1),
                ),
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
        ).view(sequence, batch, hidden)
        mlp_input = self.post_attention_layernorm(mlp_input)
        mlp_output = (
            self.moe(mlp_input) if self.moe is not None else self.mlp(mlp_input)
        )
        return prefix_sum + mlp_output, block_residual


class K3ParallelModel(nn.Module):
    """TP-ready K3 composition; other axes remain guarded by the protocol."""

    def __init__(self, config: K3Config, ps: ParallelState) -> None:
        super().__init__()
        self.config = config
        self.ps = ps
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            ps,
        )
        self.layers = nn.ModuleList(
            [
                K3ParallelDecoderLayer(config, ps, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.output_attn_res_norm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = VocabParallelOutput(
            config.vocab_size,
            config.hidden_size,
            ps,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = scatter_to_sequence_parallel(
            self.embed_tokens(input_ids),
            self.ps,
        )
        sequence, batch, hidden = hidden_states.shape
        block_residual = hidden_states.new_zeros(sequence * batch, 0, hidden)
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, block_residual)
        hidden_states = _apply_attention_residual(
            hidden_states.reshape(-1, hidden),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(sequence, batch, hidden)
        local_logits = self.lm_head(self.norm(hidden_states))
        logits = self.lm_head.gather(local_logits).transpose(0, 1).contiguous()
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = vocab_parallel_cross_entropy(
                local_logits,
                labels.transpose(0, 1).contiguous(),
                self.ps.tp_group,
            ).mean()
        return output


__all__ = [
    "K3DenseMLP",
    "K3LatentMoE",
    "K3ParallelDecoderLayer",
    "K3ParallelModel",
    "situ_and_mul",
]
