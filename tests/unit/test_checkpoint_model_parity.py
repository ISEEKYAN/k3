from __future__ import annotations

import torch
import torch.nn.functional as F

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    iter_hf_weights,
    load_weights_from_reader,
)
from mlite_k3.model import K3Model


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _tiny_quantized_config() -> K3Config:
    return K3Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        kda_head_dim=8,
        kda_num_heads=4,
        kda_short_conv_kernel_size=2,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=32,
        routed_expert_hidden_size=32,
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=2,
    )


def _release_reader(model: K3Model, spec: K3WeightSpec) -> _Reader:
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    release = {}
    for name, tensor in iter_hf_weights(model, spec):
        if ".experts." not in name:
            release[name] = tensor.clone()
            continue
        packed, scale = quantize_mxfp4(tensor)
        release[f"{name}_packed"] = packed
        release[f"{name}_scale"] = scale.view(torch.uint8)
    return _Reader(release)


def _rms_norm(module, x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    values = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + module.variance_epsilon
    )
    return module.weight.float() * values


def _causal_short_conv(module, x: torch.Tensor) -> torch.Tensor:
    values = F.pad(x.transpose(1, 2), (module.kernel_size - 1, 0))
    values = F.conv1d(
        values,
        module.conv.weight,
        bias=None,
        groups=module.conv.groups,
    )
    return F.silu(values.transpose(1, 2))


def _official_kda_reference(module, x: torch.Tensor) -> torch.Tensor:
    """Independent transcription of pinned modeling_kimi_linear.py KDA math."""
    shape = (*x.shape[:-1], module.heads, module.head_dim)
    q = _causal_short_conv(module.q_conv1d, F.linear(x, module.q_proj.weight))
    k = _causal_short_conv(module.k_conv1d, F.linear(x, module.k_proj.weight))
    v = _causal_short_conv(module.v_conv1d, F.linear(x, module.v_proj.weight))
    q = F.normalize(q.view(shape).float(), p=2, dim=-1)
    k = F.normalize(k.view(shape).float(), p=2, dim=-1)
    v = v.view(shape).float()
    gate_logits = F.linear(
        F.linear(x, module.f_a_proj.weight),
        module.f_b_proj.weight,
    ).view(shape)
    beta = torch.sigmoid(F.linear(x, module.b_proj.weight).float())
    gate = module.lower_bound * torch.sigmoid(
        module.A_log.float().exp().view(1, 1, module.heads, 1)
        * (
            gate_logits.float()
            + module.dt_bias.float().view(1, 1, module.heads, module.head_dim)
        )
    )

    state = q.new_zeros(q.size(0), module.heads, module.head_dim, module.head_dim)
    outputs = []
    for index in range(q.size(1)):
        state = state * gate[:, index].exp().unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k[:, index], state)
        delta = (v[:, index] - prediction) * beta[:, index].unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, index], delta)
        outputs.append(
            torch.einsum("bhd,bhdv->bhv", q[:, index], state) * module.head_dim**-0.5
        )
    output = torch.stack(outputs, dim=1)
    output_gate = torch.sigmoid(F.linear(x, module.g_proj.weight).view(shape))
    output = _rms_norm(module.o_norm, output) * output_gate
    return F.linear(output.flatten(-2), module.o_proj.weight)


def _official_mla_reference(module, x: torch.Tensor) -> torch.Tensor:
    """Independent eager-attention form of pinned Kimi MLA."""
    batch, sequence, _ = x.shape
    query = F.linear(
        _rms_norm(module.q_a_layernorm, F.linear(x, module.q_a_proj.weight)),
        module.q_b_proj.weight,
    ).view(batch, sequence, module.num_heads, module.q_head_dim)
    compressed = F.linear(x, module.kv_a_proj_with_mqa.weight)
    kv_latent, key_tail = compressed.split(
        [module.kv_a_layernorm.weight.numel(), module.qk_rope_head_dim],
        dim=-1,
    )
    key_value = F.linear(
        _rms_norm(module.kv_a_layernorm, kv_latent),
        module.kv_b_proj.weight,
    ).view(
        batch,
        sequence,
        module.num_heads,
        module.qk_nope_head_dim + module.v_head_dim,
    )
    key_nope, value = key_value.split(
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
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key)
    scores = scores * module.q_head_dim**-0.5
    causal = torch.ones(sequence, sequence, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probabilities = scores.softmax(dim=-1, dtype=torch.float32)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    output = output.transpose(1, 2).reshape(batch, sequence, -1)
    output = output * torch.sigmoid(F.linear(x, module.g_proj.weight))
    return F.linear(output, module.o_proj.weight)


def _official_mlp_reference(module, x: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x, module.gate_up.weight).chunk(2, dim=-1)
    activated = module.beta * torch.tanh(gate / module.beta) * torch.sigmoid(gate)
    if module.linear_beta is not None:
        up = module.linear_beta * torch.tanh(up / module.linear_beta)
    return F.linear(activated * up, module.down.weight)


def _official_moe_reference(module, x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    flat = x.reshape(-1, x.size(-1))
    scores = torch.sigmoid(F.linear(flat.float(), module.router.weight.float()))
    selection = scores + module.expert_bias.float()
    indices = torch.topk(selection, module.top_k, dim=-1, sorted=False).indices
    topk_scores = scores.gather(-1, indices)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    topk_scores = topk_scores * module.scaling_factor

    latent = F.linear(flat, module.routed_expert_down_proj.weight)
    expert_outputs = torch.stack(
        [_official_mlp_reference(expert, latent) for expert in module.experts],
        dim=1,
    )
    routing = torch.zeros_like(scores).scatter(-1, indices, topk_scores)
    routed = torch.einsum("te,ted->td", routing, expert_outputs)
    routed = F.linear(
        _rms_norm(module.routed_expert_norm, routed),
        module.routed_expert_up_proj.weight,
    )
    shared = _official_mlp_reference(module.shared_experts, flat)
    return (routed + shared).view(original_shape)


def _official_attn_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection,
    norm,
) -> torch.Tensor:
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    keys = values_float * torch.rsqrt(
        values_float.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
    )
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(dim=-1).softmax(dim=-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1)


def _official_layer_reference(
    layer,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, hidden = hidden_states.shape
    prefix_sum = hidden_states
    if block_residual.size(1):
        hidden_states = _official_attn_residual(
            prefix_sum.reshape(-1, hidden),
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view(batch, sequence, hidden)
    if layer.layer_index % layer.attn_res_block_size == 0:
        block_residual = torch.cat(
            (block_residual, prefix_sum.reshape(-1, hidden).unsqueeze(1)),
            dim=1,
        )
        prefix_sum = None

    attention_input = _rms_norm(layer.input_layernorm, hidden_states)
    if layer.layer_index == 0:
        attention_output = _official_kda_reference(
            layer.self_attention,
            attention_input,
        )
    else:
        attention_output = _official_mla_reference(
            layer.self_attention,
            attention_input,
        )
    prefix_sum = (
        attention_output if prefix_sum is None else prefix_sum + attention_output
    )
    mlp_input = _official_attn_residual(
        prefix_sum.reshape(-1, hidden),
        block_residual,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view(batch, sequence, hidden)
    mlp_input = _rms_norm(layer.post_attention_layernorm, mlp_input)
    mlp_output = (
        _official_moe_reference(layer.moe, mlp_input)
        if layer.moe is not None
        else _official_mlp_reference(layer.mlp, mlp_input)
    )
    return prefix_sum + mlp_output, block_residual


def test_real_tiny_model_tensors_have_complete_public_weight_mapping():
    config = _tiny_quantized_config()
    model = K3Model(config)
    mapping = K3WeightSpec(config).weight_map()

    production_bias = "layers.1.moe.router.expert_bias"
    assert set(dict(model.named_parameters())) | set(dict(model.named_buffers())) == (
        set(mapping) - {production_bias}
    )
    assert mapping[production_bias] == mapping["layers.1.moe.expert_bias"]
    assert "layers.1.moe.expert_bias" not in dict(model.named_parameters())
    assert "layers.1.moe.expert_bias" in dict(model.named_buffers())
    assert "layers.1.moe.expert_bias" in model.state_dict()
    assert not any(name.endswith(".conv.bias") for name, _ in model.named_parameters())


def test_mxfp4_loaded_tiny_model_matches_independent_layer_reference():
    # The independent equations above are transcribed from the pinned public
    # modeling_kimi_linear.py SHA-256 9e3564c70ac21854ce5a090cc946c5dc...
    torch.manual_seed(20260727)
    config = _tiny_quantized_config()
    source = K3Model(config)
    spec = K3WeightSpec(config)
    reader = _release_reader(source, spec)
    target = K3Model(config)

    assert sum(name.endswith("_packed") for name in reader.index) == (
        config.num_experts * 3
    )
    assert audit_k3_weight_spec_sources(spec, reader.index) == len(
        {name for names in spec.weight_map().values() for name in names}
    )
    assert load_weights_from_reader(target, reader, spec) == len(target.state_dict())
    plain_hf = dict(iter_hf_weights(target, spec))
    assert all(tensor.dtype == torch.bfloat16 for tensor in plain_hf.values())
    reloaded = K3Model(config)
    assert audit_k3_weight_spec_sources(spec, plain_hf) == len(plain_hf)
    assert load_weights_from_reader(reloaded, _Reader(plain_hf), spec) == len(
        reloaded.state_dict()
    )
    for (target_name, target_parameter), (reloaded_name, reloaded_parameter) in zip(
        target.state_dict().items(),
        reloaded.state_dict().items(),
        strict=True,
    ):
        assert target_name == reloaded_name
        assert torch.equal(target_parameter, reloaded_parameter)

    input_ids = torch.tensor([[1, 5, 7], [3, 2, 9]])
    target_hidden = target.embed_tokens(input_ids)
    reference_hidden = F.embedding(input_ids, target.embed_tokens.weight)
    target_residual = target_hidden.new_zeros(
        target_hidden.size(0) * target_hidden.size(1),
        0,
        target_hidden.size(2),
    )
    reference_residual = target_residual.clone()

    for layer in target.layers:
        reference_hidden, reference_residual = _official_layer_reference(
            layer,
            reference_hidden,
            reference_residual,
        )
        target_hidden, target_residual = layer(target_hidden, target_residual)
        torch.testing.assert_close(
            target_hidden,
            reference_hidden,
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(target_residual, reference_residual)

    hidden = target_hidden.size(-1)
    reference_hidden = _official_attn_residual(
        reference_hidden.reshape(-1, hidden),
        reference_residual,
        target.output_attn_res_proj,
        target.output_attn_res_norm,
    ).view_as(reference_hidden)
    reference_logits = F.linear(
        _rms_norm(target.norm, reference_hidden),
        target.lm_head.weight,
    )
    target_logits = target(input_ids=input_ids)["logits"]
    reloaded_logits = reloaded(input_ids=input_ids)["logits"]
    assert torch.equal(target_logits, reloaded_logits)
    torch.testing.assert_close(target_logits, reference_logits, rtol=0, atol=1e-6)
