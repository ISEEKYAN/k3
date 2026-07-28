from __future__ import annotations

import torch
import torch.nn.functional as F

from mlite_k3.config import K3Config
from mlite_k3.kda import KDA, kda_recurrent_reference
from mlite_k3.primitives import GatedMultiLatentAttention, LatentMoE


def _tiny_config() -> K3Config:
    return K3Config(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        intermediate_size=24,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        kda_head_dim=4,
        kda_num_heads=2,
        kda_short_conv_kernel_size=2,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=6,
        routed_expert_hidden_size=8,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=2,
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + eps
    )
    return weight * normalized.to(x.dtype)


def _causal_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    kernel_size = weight.shape[-1]
    values = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
    values = F.conv1d(values, weight, groups=weight.shape[0])
    return F.silu(values.transpose(1, 2))


def _kda_reference(module: KDA, x: torch.Tensor) -> torch.Tensor:
    batch, sequence, _ = x.shape
    heads, head_dim = module.heads, module.head_dim

    def project(projection, convolution):
        values = F.linear(x, projection.weight)
        values = _causal_depthwise_conv(values, convolution.conv.weight)
        return values.view(batch, sequence, heads, head_dim)

    q = F.normalize(project(module.q_proj, module.q_conv1d).float(), dim=-1)
    k = F.normalize(project(module.k_proj, module.k_conv1d).float(), dim=-1)
    v = project(module.v_proj, module.v_conv1d).float()
    feature_gate = F.linear(
        F.linear(x, module.f_a_proj.weight),
        module.f_b_proj.weight,
    ).view(batch, sequence, heads, head_dim)
    beta = torch.sigmoid(F.linear(x, module.b_proj.weight).float())
    gate = module.lower_bound * torch.sigmoid(
        module.A_log.float().exp().view(1, 1, heads, 1)
        * (feature_gate.float() + module.dt_bias.float().view(1, 1, heads, head_dim))
    )
    state = x.new_zeros(batch, heads, head_dim, head_dim, dtype=torch.float32)
    outputs = []
    for token in range(sequence):
        state = state * gate[:, token].exp().unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
        update = (v[:, token] - prediction) * beta[:, token].unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, token], update)
        outputs.append(
            torch.einsum("bhd,bhdv->bhv", q[:, token], state) * head_dim**-0.5
        )
    output = torch.stack(outputs, dim=1).to(x.dtype)
    output = _rms_norm(
        output,
        module.o_norm.weight,
        module.o_norm.variance_epsilon,
    )
    output_gate = torch.sigmoid(
        F.linear(x, module.g_proj.weight).view(batch, sequence, heads, head_dim)
    )
    return F.linear((output * output_gate).flatten(-2), module.o_proj.weight)


def _mla_reference(
    module: GatedMultiLatentAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, _ = x.shape
    query_latent = _rms_norm(
        F.linear(x, module.q_a_proj.weight),
        module.q_a_layernorm.weight,
        module.q_a_layernorm.variance_epsilon,
    )
    query = F.linear(query_latent, module.q_b_proj.weight)
    query = query.view(batch, sequence, module.num_heads, module.q_head_dim)
    compressed = F.linear(x, module.kv_a_proj_with_mqa.weight)
    kv_latent, key_tail = compressed.split(
        [module.kv_a_layernorm.weight.numel(), module.qk_rope_head_dim],
        dim=-1,
    )
    kv_latent = _rms_norm(
        kv_latent,
        module.kv_a_layernorm.weight,
        module.kv_a_layernorm.variance_epsilon,
    )
    kv = F.linear(kv_latent, module.kv_b_proj.weight)
    kv = kv.view(
        batch,
        sequence,
        module.num_heads,
        module.qk_nope_head_dim + module.v_head_dim,
    )
    key_nope, value = kv.split(
        [module.qk_nope_head_dim, module.v_head_dim],
        dim=-1,
    )
    key_tail = key_tail[:, :, None, :].expand(
        batch,
        sequence,
        module.num_heads,
        module.qk_rope_head_dim,
    )
    key = torch.cat((key_nope, key_tail), dim=-1)
    scores = torch.einsum("bthd,bshd->bhts", query, key)
    scores = scores * module.q_head_dim**-0.5
    causal = torch.ones(sequence, sequence, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(causal, float("-inf"))
    probabilities = scores.float().softmax(dim=-1).to(value.dtype)
    output = torch.einsum("bhts,bshd->bthd", probabilities, value)
    output = output.reshape(batch, sequence, -1)
    output = output * torch.sigmoid(F.linear(x, module.g_proj.weight))
    return F.linear(output, module.o_proj.weight)


def _situ(
    gate_up: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return activated * up


def _mlp_reference(module, x: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, module.gate_up.weight)
    return F.linear(
        _situ(gate_up, beta=module.beta, linear_beta=module.linear_beta),
        module.down.weight,
    )


def _latent_moe_reference(module: LatentMoE, x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    scores = torch.sigmoid(F.linear(flat.float(), module.router.weight.float()))
    selection = scores + module.expert_bias
    indices = torch.topk(selection, module.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    weights = weights * module.scaling_factor

    latent = F.linear(flat, module.routed_expert_down_proj.weight)
    expert_outputs = torch.stack(
        [_mlp_reference(expert, latent) for expert in module.experts],
        dim=1,
    )
    routing = torch.zeros_like(scores).scatter(-1, indices, weights)
    routed = torch.einsum("te,ted->td", routing, expert_outputs)
    routed = _rms_norm(
        routed,
        module.routed_expert_norm.weight,
        module.routed_expert_norm.variance_epsilon,
    )
    routed = F.linear(routed, module.routed_expert_up_proj.weight)
    shared = _mlp_reference(module.shared_experts, flat)
    return (routed + shared).view(original_shape)


def test_tiny_kda_matches_independent_reference_and_preserves_dtype():
    torch.manual_seed(11)
    config = _tiny_config()
    module = KDA(
        hidden_size=config.hidden_size,
        heads=config.kda_num_heads,
        head_dim=config.kda_head_dim,
        short_conv_kernel_size=config.kda_short_conv_kernel_size,
        lower_bound=config.kda_gate_lower_bound,
        norm_eps=config.rms_norm_eps,
        use_full_rank_gate=config.kda_use_full_rank_gate,
    )
    x = torch.randn(2, 4, config.hidden_size)

    actual = module(x)
    expected = _kda_reference(module, x)

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_kda_recurrent_reference_preserves_bfloat16_value_dtype():
    shape = (1, 2, 1, 4)
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    gate = torch.randn(shape, dtype=torch.bfloat16)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16)

    output, state = kda_recurrent_reference(
        q,
        k,
        v,
        gate,
        beta,
        a_log=torch.zeros(1),
        dt_bias=torch.zeros(1, 4),
        lower_bound=-5.0,
    )

    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32


def test_kda_keeps_fla_gate_parameters_in_float32_when_model_is_bfloat16():
    config = _tiny_config()
    module = KDA(
        hidden_size=config.hidden_size,
        heads=config.kda_num_heads,
        head_dim=config.kda_head_dim,
        short_conv_kernel_size=config.kda_short_conv_kernel_size,
        lower_bound=config.kda_gate_lower_bound,
        norm_eps=config.rms_norm_eps,
        use_full_rank_gate=config.kda_use_full_rank_gate,
    ).to(dtype=torch.bfloat16)

    assert module.q_proj.weight.dtype == torch.bfloat16
    assert module.A_log.dtype == torch.float32
    assert module.dt_bias.dtype == torch.float32


def test_tiny_gated_mla_matches_independent_reference():
    torch.manual_seed(13)
    module = GatedMultiLatentAttention(_tiny_config())
    x = torch.randn(2, 4, 16)

    actual = module(x)
    expected = _mla_reference(module, x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_tiny_latent_moe_matches_independent_reference():
    torch.manual_seed(17)
    module = LatentMoE(_tiny_config())
    x = torch.randn(2, 4, 16)

    actual = module(x)
    expected = _latent_moe_reference(module, x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
