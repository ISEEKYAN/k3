from __future__ import annotations

import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import K3WeightSpec, load_weights_from_reader
from mlite_k3.model import K3Model


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


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


def _deterministic_hf_reader(
    model: K3Model,
    spec: K3WeightSpec,
) -> _Reader:
    tensors: dict[str, torch.Tensor] = {}
    mapping = spec.weight_map()
    named_tensors = list(model.named_parameters()) + list(model.named_buffers())
    for offset, (native_name, parameter) in enumerate(named_tensors):
        source_names = mapping[native_name]
        values = torch.linspace(
            -0.02 + offset * 1e-5,
            0.02 + offset * 1e-5,
            parameter.numel(),
            dtype=torch.float32,
        ).reshape(parameter.shape)
        if native_name.endswith(".gate_up.weight"):
            source_values = values.chunk(2, dim=0)
        elif native_name.endswith(
            (".q_conv1d.conv.weight", ".k_conv1d.conv.weight", ".v_conv1d.conv.weight")
        ):
            source_values = (values.squeeze(1),)
        else:
            source_values = (values,)
        tensors.update(zip(source_names, source_values, strict=True))
    return _Reader(tensors)


def test_complete_tiny_weight_spec_load_runs_hybrid_forward_backward():
    torch.manual_seed(7)
    config = _tiny_config()
    model = K3Model(config)
    spec = K3WeightSpec(config)
    reader = _deterministic_hf_reader(model, spec)

    loaded = load_weights_from_reader(model, reader, spec)

    assert loaded == len(model.state_dict())
    assert reader.index == {
        name for names in spec.weight_map().values() for name in names
    }
    output = model(
        input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        labels=torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]]),
    )
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].self_attention.q_proj.weight.grad is not None
    assert model.layers[1].self_attention.q_a_proj.weight.grad is not None
    assert model.layers[1].moe.router.weight.grad is not None
