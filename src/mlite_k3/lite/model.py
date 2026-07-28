"""Megatron-Lite distributed composition for Kimi K3."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp

from mlite_k3.config import K3Config
from mlite_k3.lite.pipeline_state import (
    _pack_pipeline_state,
    _unpack_pipeline_state,
)
from mlite_k3.lite.thd_contract import validate_thd_inputs
from mlite_k3.model import _apply_attention_residual
from mlite_k3.primitive.kda import kda
from mlite_k3.primitive.kda_parallel import K3FullRankGatedDeltaNet
from mlite_k3.primitive.mla import K3MultiLatentAttention


def _situ_with_probs(
    gate_up: torch.Tensor,
    probs: torch.Tensor | None,
    *,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate_float = gate.float()
    up_float = up.float()
    output = (
        beta
        * torch.tanh(gate_float / beta)
        * torch.sigmoid(gate_float)
        * (linear_beta * torch.tanh(up_float / linear_beta))
    )
    if probs is not None:
        output = output * probs.float()
    return output.to(gate_up.dtype)


class ParallelSITUMlp(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: K3Config,
        ps: ParallelState,
    ):
        super().__init__()
        self.gate_up = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            ps,
            bias=False,
        )
        self.down = RowParallelLinear(intermediate_size, hidden_size, ps, bias=False)
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = _situ_with_probs(
            self.gate_up(x),
            None,
            beta=self.beta,
            linear_beta=self.linear_beta,
        )
        return self.down(activated)


class ParallelLatentMoE(nn.Module):
    """K3 latent projection around the shared router/dispatcher/Experts chain."""

    def __init__(
        self, config: K3Config, ps: ParallelState, *, use_deepep: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.router = SigmoidTopKRouter(
            config,
            ps,
            compute_aux_loss=False,
            router_dtype=torch.float32,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.routed_expert_hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size,
            config.routed_expert_hidden_size,
            bias=False,
        )

        def activation(x: torch.Tensor, probs: torch.Tensor | None) -> torch.Tensor:
            return _situ_with_probs(
                x,
                probs,
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
            )

        self.experts = Experts(
            config,
            ps,
            hidden_size=config.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            activation=activation,
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
        self.shared_experts = ParallelSITUMlp(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config,
            ps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.router(x_flat)
        latent = self.routed_expert_down_proj(x_flat)
        dispatched, tokens_per_expert, permuted_probs = self.dispatcher.dispatch(
            latent,
            weights,
            indices,
        )
        self.dispatcher.wait_dispatch_event()
        routed = self.experts(
            dispatched,
            tokens_per_expert,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed = self.dispatcher.combine(routed)
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        shared = self.shared_experts(x).reshape(-1, self.hidden_size)
        return (routed + shared).view(shape)


class K3ParallelDecoderLayer(nn.Module):
    def __init__(
        self,
        config: K3Config,
        layer_index: int,
        ps: ParallelState,
        *,
        use_thd: bool,
        use_deepep: bool,
        deterministic: bool,
        kda_cp_mode: str,
    ):
        super().__init__()
        self.layer_index = layer_index
        attention_type = config.attention_type(layer_index)
        self.attn_res_block_size = config.attn_res_block_size
        self.rms_norm_eps = config.rms_norm_eps
        if attention_type == "kda":
            self.self_attention: nn.Module = K3FullRankGatedDeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                conv_kernel_size=config.kda_short_conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
                gate_lower_bound=config.kda_gate_lower_bound,
                ps=ps,
                recurrence=kda,
                deterministic=deterministic,
                cp_mode=kda_cp_mode,
            )
        else:
            self.self_attention = K3MultiLatentAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                ps=ps,
                rms_norm_eps=config.rms_norm_eps,
                use_thd=use_thd,
                use_nope=config.mla_use_nope,
                output_gate=config.mla_use_output_gate,
            )
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = te.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attention_res_norm = te.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp_res_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.moe = (
            ParallelLatentMoE(config, ps, use_deepep=use_deepep)
            if layer_index >= config.first_k_dense_replace
            else None
        )
        self.mlp = (
            None
            if self.moe is not None
            else ParallelSITUMlp(
                config.hidden_size,
                config.intermediate_size,
                config,
                ps,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        packed_seq_params: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, batch, hidden = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.size(1):
            hidden_states = _apply_attention_residual(
                prefix_sum.reshape(-1, hidden),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                variance_epsilon=self.rms_norm_eps,
            ).view(sequence, batch, hidden)
        if self.layer_index % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.reshape(-1, hidden).unsqueeze(1)),
                dim=1,
            )
            prefix_sum = None

        attention_input = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(
            attention_input,
            packed_seq_params=packed_seq_params,
        )
        prefix_sum = (
            attention_output if prefix_sum is None else prefix_sum + attention_output
        )
        mlp_input = _apply_attention_residual(
            prefix_sum.reshape(-1, hidden),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            variance_epsilon=self.rms_norm_eps,
        ).view(sequence, batch, hidden)
        mlp_input = self.post_attention_layernorm(mlp_input)
        mlp_output = (
            self.moe(mlp_input) if self.moe is not None else self.mlp(mlp_input)
        )
        return prefix_sum + mlp_output, block_residual


class K3ParallelModel(nn.Module):
    def __init__(
        self,
        config: K3Config,
        ps: ParallelState,
        *,
        use_thd: bool = False,
        use_deepep: bool = False,
        deterministic: bool = False,
        kda_cp_mode: str = "headwise",
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.rms_norm_eps = config.rms_norm_eps
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.embed_tokens: VocabParallelEmbedding | None = None
        if self.pre_process:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                ps,
                deterministic=deterministic,
            )
        self.layers = nn.ModuleList(
            [
                K3ParallelDecoderLayer(
                    config,
                    index,
                    ps,
                    use_thd=use_thd,
                    use_deepep=use_deepep,
                    deterministic=deterministic,
                    kda_cp_mode=kda_cp_mode,
                )
                for index in self.layer_indices
            ]
        )
        self.output_attn_res_norm: nn.Module | None = None
        self.output_attn_res_proj: nn.Linear | None = None
        self.norm: nn.Module | None = None
        self.lm_head: VocabParallelOutput | None = None
        if self.post_process:
            self.output_attn_res_norm = te.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = VocabParallelOutput(
                config.vocab_size, config.hidden_size, ps
            )

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("K3ParallelModel expects one folded pipeline tensor")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def _apply(self, fn, recurse: bool = True):
        model = super()._apply(fn, recurse=recurse)
        # The official KDA decay state remains FP32 even when model weights and
        # activations are converted to BF16. Reassert it after the outer model
        # conversion because Torch/TE may replace Parameters during recursion.
        for layer in self.layers:
            attention = layer.self_attention
            for name in ("A_log", "dt_bias"):
                parameter = getattr(attention, name, None)
                if parameter is not None and parameter.dtype != torch.float32:
                    setattr(
                        attention,
                        name,
                        nn.Parameter(
                            parameter.detach().float(),
                            requires_grad=parameter.requires_grad,
                        ),
                    )
        return model

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> dict[str, torch.Tensor]:
        if self.pre_process:
            if input_ids is None or input_ids.ndim != 2:
                raise ValueError(
                    "K3 embedding stage requires dense [batch, sequence] input_ids"
                )
            if packed_seq_params is not None:
                validate_thd_inputs(
                    input_ids,
                    labels,
                    loss_mask,
                    packed_seq_params,
                )
            assert self.embed_tokens is not None
            hidden_states = self.embed_tokens(input_ids)
            if self.ps.cp_size > 1:
                hidden_states = zigzag_slice_for_cp(
                    hidden_states,
                    self.ps.cp_rank,
                    self.ps.cp_size,
                    seq_dim=0,
                )
            hidden_states = scatter_to_sequence_parallel(hidden_states, self.ps)
            sequence, batch, hidden = hidden_states.shape
            block_residual = hidden_states.new_zeros(sequence * batch, 0, hidden)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            if hidden_states is None:
                raise ValueError("K3 non-first pipeline stage requires an input tensor")
            hidden_states, block_residual = _unpack_pipeline_state(
                hidden_states,
                hidden_size=self.config.hidden_size,
            )
            sequence, batch, hidden = hidden_states.shape
        for layer in self.layers:
            hidden_states, block_residual = layer(
                hidden_states,
                block_residual,
                packed_seq_params=packed_seq_params,
            )
        if not self.post_process:
            return {
                "hidden_states": _pack_pipeline_state(hidden_states, block_residual)
            }

        assert self.output_attn_res_proj is not None
        assert self.output_attn_res_norm is not None
        assert self.norm is not None
        assert self.lm_head is not None
        hidden_states = _apply_attention_residual(
            hidden_states.reshape(-1, hidden),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            variance_epsilon=self.rms_norm_eps,
        ).view(sequence, batch, hidden)
        logits = self.lm_head(self.norm(hidden_states))
        output: dict[str, torch.Tensor] = {"hidden_states": hidden_states}
        if labels is None:
            output["logits"] = self.lm_head.gather(logits).transpose(0, 1).contiguous()
        else:
            labels_sb = labels.transpose(0, 1).contiguous()
            if self.ps.cp_size > 1:
                labels_sb = zigzag_slice_for_cp(
                    labels_sb,
                    self.ps.cp_rank,
                    self.ps.cp_size,
                    seq_dim=0,
                )
            loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
            if loss_mask is None:
                output["loss"] = loss.mean()
            else:
                mask_sb = loss_mask.transpose(0, 1).to(dtype=loss.dtype)
                output["loss"] = (loss * mask_sb).sum() / mask_sb.sum().clamp_min(1)
            output["log_probs"] = (-loss).transpose(0, 1).contiguous()
        return output


__all__ = [
    "K3ParallelDecoderLayer",
    "K3ParallelModel",
    "ParallelLatentMoE",
    "ParallelSITUMlp",
]
