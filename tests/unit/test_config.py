from __future__ import annotations

import pytest

from mlite_k3.config import K3Config


def _tiny_text_config() -> dict:
    return {
        "model_type": "kimi_linear",
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "intermediate_size": 64,
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "gate_lower_bound": -5.0,
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 12,
        "routed_expert_hidden_size": 16,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 2,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
    }


def test_config_unwraps_official_multimodal_wrapper_but_models_text_only():
    config = K3Config._from_hf_dict(
        {
            "model_type": "kimi_k3",
            "text_config": _tiny_text_config(),
            "vision_config": {"vt_hidden_size": 1024},
        }
    )

    assert config.source_model_type == "kimi_k3"
    assert config.attention_type(0) == "kda"
    assert config.attention_type(3) == "mla"
    assert config.layer_types == ["kda", "kda", "kda", "mla"]
    assert config.num_shared_experts == 2
    assert config.routed_expert_hidden_size == 16
    assert config.moe_intermediate_size == 12


def test_config_preserves_official_k3_defaults():
    config = K3Config()

    assert config.num_hidden_layers == 93
    assert config.layer_types.count("kda") == 69
    assert config.layer_types.count("mla") == 24
    assert config.layer_types[-3:] == ["kda", "mla", "mla"]
    assert config.num_experts == 896
    assert config.num_experts_per_token == 16
    assert config.num_shared_experts == 2
    assert config.mla_use_nope
    assert config.mla_use_output_gate
    assert config.kda_use_full_rank_gate
    assert config.kda_gate_lower_bound == -5.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mla_use_nope", False),
        ("mla_use_output_gate", False),
        ("num_shared_experts", 1),
        ("moe_router_activation_func", "softmax"),
        ("moe_renormalize", False),
        ("topk_method", "greedy"),
        ("use_grouped_topk", False),
        ("latent_moe_use_norm", False),
        ("hidden_act", "silu"),
    ],
)
def test_config_rejects_unsupported_architecture_drift(field, value):
    source = _tiny_text_config()
    source[field] = value

    with pytest.raises(ValueError, match=field):
        K3Config._from_hf_dict(source)


def test_config_rejects_incomplete_or_overlapping_attention_schedule():
    source = _tiny_text_config()
    source["linear_attn_config"] = {
        **source["linear_attn_config"],
        "full_attn_layers": [3, 4],
        "kda_layers": [1, 2, 3],
    }

    with pytest.raises(ValueError, match="attention layer schedule"):
        K3Config._from_hf_dict(source)


def test_config_rejects_bare_vision_model_type():
    with pytest.raises(ValueError, match="kimi_k3 or kimi_linear"):
        K3Config._from_hf_dict({"model_type": "moonvit"})
