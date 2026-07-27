from __future__ import annotations

import torch

from mlite_k3.config import K3Config


def tiny_config() -> K3Config:
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


def test_kda_reference_recurrence_matches_one_token_equation():
    from mlite_k3.primitives import kda_recurrent_reference

    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]]]])
    v = torch.tensor([[[[2.0, -1.0]]]])
    gate_logits = torch.zeros_like(q)
    beta_logits = torch.zeros(1, 1, 1)
    a_log = torch.zeros(1)
    dt_bias = torch.zeros(1, 2)

    out, state = kda_recurrent_reference(
        q,
        k,
        v,
        gate_logits,
        beta_logits,
        a_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )

    expected_state = torch.tensor([[[[1.0, -0.5], [0.0, 0.0]]]])
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(out, torch.tensor([[[[1.0, -0.5]]]]))


def test_hybrid_model_uses_kda_mla_and_latent_moe_in_real_forward_backward():
    from mlite_k3.model import K3Model
    from mlite_k3.primitives import (
        GatedMultiLatentAttention,
        KimiDeltaAttention,
        LatentMoE,
    )

    torch.manual_seed(7)
    model = K3Model(tiny_config())
    assert isinstance(model.layers[0].self_attention, KimiDeltaAttention)
    assert isinstance(model.layers[1].self_attention, GatedMultiLatentAttention)
    assert model.layers[0].moe is None
    assert isinstance(model.layers[1].moe, LatentMoE)

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
    output = model(input_ids=input_ids, labels=labels)

    assert output["logits"].shape == (2, 4, 32)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.layers[0].self_attention.g_proj.weight.grad is not None
    assert model.layers[1].self_attention.g_proj.weight.grad is not None
    assert model.layers[1].moe.router.weight.grad is not None
    assert model.layers[1].moe.routed_expert_down_proj.weight.grad is not None


def test_protocol_is_importable_and_builds_tiny_config_from_public_shape():
    from mlite_k3.lite import protocol

    config = protocol.build_model_config(
        {
            "model_type": "kimi_linear",
            **{
                key: value
                for key, value in tiny_config().__dict__.items()
                if key
                not in {
                    "source_model_type",
                    "full_attention_layers",
                    "kda_layers",
                    "kda_head_dim",
                    "kda_num_heads",
                    "kda_short_conv_kernel_size",
                    "kda_use_full_rank_gate",
                    "kda_gate_lower_bound",
                }
            },
            "linear_attn_config": {
                "full_attn_layers": [2],
                "kda_layers": [1],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 2,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
        }
    )

    assert config.layer_types == ["kda", "mla"]
    assert protocol.vocab_size(config) == 32
