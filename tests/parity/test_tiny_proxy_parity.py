from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from mlite_k3.config import K3Config
from mlite_k3.model import K3Model


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


def _linear(x: torch.Tensor, module) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


def _rms_norm(x: torch.Tensor, module) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + module.variance_epsilon)
    return (normalized * module.weight.float()).to(x.dtype)


def _depthwise_conv(x: torch.Tensor, module) -> torch.Tensor:
    values = F.pad(x.transpose(1, 2), (module.kernel_size - 1, 0))
    values = F.conv1d(
        values,
        module.conv.weight,
        module.conv.bias,
        groups=module.conv.groups,
    )
    return F.silu(values.transpose(1, 2))


def _reference_kda(x: torch.Tensor, module) -> torch.Tensor:
    shape = (*x.shape[:2], module.heads, module.head_dim)
    q = _depthwise_conv(_linear(x, module.q_proj), module.q_conv1d).view(shape)
    k = _depthwise_conv(_linear(x, module.k_proj), module.k_conv1d).view(shape)
    v = _depthwise_conv(_linear(x, module.v_proj), module.v_conv1d).view(shape)
    gate_logits = _linear(_linear(x, module.f_a_proj), module.f_b_proj).view(shape)
    beta = _linear(x, module.b_proj).float().sigmoid()
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    gate = module.lower_bound * torch.sigmoid(
        module.A_log.float().exp().view(1, 1, module.heads, 1)
        * (
            gate_logits.float()
            + module.dt_bias.float().view(1, 1, module.heads, module.head_dim)
        )
    )
    state = x.new_zeros(
        x.shape[0],
        module.heads,
        module.head_dim,
        module.head_dim,
        dtype=torch.float32,
    )
    outputs = []
    for token in range(x.shape[1]):
        state = state * gate[:, token].exp().unsqueeze(-1)
        predicted = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
        update = beta[:, token].unsqueeze(-1) * (v[:, token].float() - predicted)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, token], update)
        outputs.append(
            torch.einsum("bhd,bhdv->bhv", q[:, token], state) * module.head_dim**-0.5
        )
    output = torch.stack(outputs, dim=1).to(x.dtype)
    output = _rms_norm(output, module.o_norm)
    output = output * _linear(x, module.g_proj).view(shape).sigmoid()
    return _linear(output.flatten(-2), module.o_proj)


def _reference_mla(x: torch.Tensor, module) -> torch.Tensor:
    batch, sequence, _ = x.shape
    query = _linear(
        _rms_norm(_linear(x, module.q_a_proj), module.q_a_layernorm),
        module.q_b_proj,
    ).view(batch, sequence, module.num_heads, module.q_head_dim)
    compressed = _linear(x, module.kv_a_proj_with_mqa)
    latent, key_tail = compressed.split(
        [module.kv_a_layernorm.weight.numel(), module.qk_rope_head_dim],
        dim=-1,
    )
    kv = _linear(_rms_norm(latent, module.kv_a_layernorm), module.kv_b_proj)
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
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = query @ key.transpose(-1, -2) * module.q_head_dim**-0.5
    causal = torch.ones(sequence, sequence, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    output = scores.softmax(dim=-1) @ value
    output = output.transpose(1, 2).reshape(batch, sequence, -1)
    output = output * _linear(x, module.g_proj).sigmoid()
    return _linear(output, module.o_proj)


def _reference_mlp(x: torch.Tensor, module) -> torch.Tensor:
    gate, up = _linear(x, module.gate_up).chunk(2, dim=-1)
    gate = module.beta * torch.tanh(gate.float() / module.beta) * gate.float().sigmoid()
    if module.linear_beta is not None:
        up = module.linear_beta * torch.tanh(up.float() / module.linear_beta)
    return _linear((gate * up.float()).to(x.dtype), module.down)


def _reference_moe(x: torch.Tensor, module) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    scores = _linear(flat, module.router).float().sigmoid()
    selection = scores + module.expert_bias.to(scores.dtype)
    _, indices = selection.topk(module.top_k, dim=-1, sorted=False)
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    weights = weights * module.scaling_factor
    latent = _linear(flat, module.routed_expert_down_proj)
    expert_outputs = torch.stack(
        [_reference_mlp(latent, expert) for expert in module.experts],
        dim=1,
    )
    routing = torch.zeros_like(scores).scatter(-1, indices, weights)
    routed = torch.einsum(
        "te,ted->td",
        routing.to(expert_outputs.dtype),
        expert_outputs,
    )
    routed = _linear(
        _rms_norm(routed, module.routed_expert_norm),
        module.routed_expert_up_proj,
    )
    return (routed + _reference_mlp(flat, module.shared_experts)).view(shape)


def _attention_residual(prefix, block, projection, norm):
    values = torch.cat((block, prefix.unsqueeze(1)), dim=1)
    normalized = _rms_norm(values, norm)
    scores = _linear(normalized, projection).float().softmax(dim=1)
    return (
        torch.matmul(scores.transpose(1, 2), values.float()).squeeze(1).to(values.dtype)
    )


def _reference_layer(hidden: torch.Tensor, block: torch.Tensor, layer):
    batch, sequence, width = hidden.shape
    prefix = hidden
    if block.shape[1]:
        hidden = _attention_residual(
            prefix.reshape(-1, width),
            block,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view(batch, sequence, width)
    if layer.layer_index % layer.attn_res_block_size == 0:
        block = torch.cat(
            (block, prefix.reshape(-1, width).unsqueeze(1)),
            dim=1,
        )
        prefix = None
    attention_input = _rms_norm(hidden, layer.input_layernorm)
    if layer.layer_index == 0:
        attention = _reference_kda(attention_input, layer.self_attention)
    else:
        attention = _reference_mla(attention_input, layer.self_attention)
    prefix = attention if prefix is None else prefix + attention
    mlp_input = _attention_residual(
        prefix.reshape(-1, width),
        block,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view(batch, sequence, width)
    mlp_input = _rms_norm(mlp_input, layer.post_attention_layernorm)
    mlp = (
        _reference_moe(mlp_input, layer.moe)
        if layer.moe is not None
        else _reference_mlp(mlp_input, layer.mlp)
    )
    return prefix + mlp, block


def _reference_model(model: K3Model, input_ids: torch.Tensor, labels: torch.Tensor):
    hidden = F.embedding(input_ids, model.embed_tokens.weight)
    batch, sequence, width = hidden.shape
    block = hidden.new_zeros(batch * sequence, 0, width)
    layer_outputs = []
    for layer in model.layers:
        hidden, block = _reference_layer(hidden, block, layer)
        layer_outputs.append(hidden)
    hidden = _attention_residual(
        hidden.reshape(-1, width),
        block,
        model.output_attn_res_proj,
        model.output_attn_res_norm,
    ).view(batch, sequence, width)
    logits = _linear(_rms_norm(hidden, model.norm), model.lm_head)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return logits, loss, layer_outputs


def test_tiny_hybrid_proxy_matches_independent_layerwise_reference():
    torch.manual_seed(20260727)
    actual = K3Model(_tiny_config())
    reference = copy.deepcopy(actual)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

    actual_layers = []
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: actual_layers.append(output[0])
        )
        for layer in actual.layers
    ]
    actual_output = actual(input_ids=input_ids, labels=labels)
    for hook in hooks:
        hook.remove()
    reference_logits, reference_loss, reference_layers = _reference_model(
        reference,
        input_ids,
        labels,
    )

    for got, expected in zip(actual_layers, reference_layers, strict=True):
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-6)
    layer_max_abs = [
        (got - expected).abs().max().item()
        for got, expected in zip(actual_layers, reference_layers, strict=True)
    ]
    torch.testing.assert_close(
        actual_output["logits"],
        reference_logits,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        actual_output["loss"],
        reference_loss,
        atol=2e-6,
        rtol=2e-6,
    )

    actual_output["loss"].backward()
    reference_loss.backward()
    parameters = [
        "embed_tokens.weight",
        "layers.0.self_attention.q_proj.weight",
        "layers.1.self_attention.q_a_proj.weight",
        "layers.1.moe.router.weight",
        "layers.1.moe.experts.0.gate_up.weight",
        "lm_head.weight",
    ]
    actual_parameters = dict(actual.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    gradient_max_abs = {}
    for name in parameters:
        gradient_max_abs[name] = (
            (actual_parameters[name].grad - reference_parameters[name].grad)
            .abs()
            .max()
            .item()
        )
        torch.testing.assert_close(
            actual_parameters[name].grad,
            reference_parameters[name].grad,
            atol=3e-6,
            rtol=3e-6,
        )
    print(
        {
            "layer_max_abs": layer_max_abs,
            "logits_max_abs": (actual_output["logits"] - reference_logits)
            .abs()
            .max()
            .item(),
            "loss_abs": (actual_output["loss"] - reference_loss).abs().item(),
            "gradient_max_abs": gradient_max_abs,
        }
    )
