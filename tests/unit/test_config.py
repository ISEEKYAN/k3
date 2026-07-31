from __future__ import annotations

from types import SimpleNamespace

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
        "moe_layer_freq": 1,
        "moe_intermediate_size": 12,
        "routed_expert_hidden_size": 16,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 2,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
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
    assert config.moe_layer_freq == 1
    assert config.num_expert_group == 1
    assert config.topk_group == 1


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


def test_transformer_config_contract_covers_dense_moe_and_router_fields():
    config = K3Config._from_hf_dict(_tiny_text_config())
    constructor, aliases = config.transformer_config_contract()

    assert constructor == {
        "num_layers": 4,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "num_moe_experts": 4,
        "moe_ffn_hidden_size": 12,
        "moe_latent_size": 16,
        "moe_shared_expert_intermediate_size": 24,
        "moe_layer_freq": [0, 1, 1, 1],
        "moe_router_topk": 2,
        "moe_router_score_function": "sigmoid",
        "moe_router_pre_softmax": True,
        "moe_router_topk_scaling_factor": 1.0,
        "moe_router_num_groups": 1,
        "moe_router_group_topk": 1,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
        "moe_router_dtype": "fp32",
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "alltoall",
    }
    assert aliases == {
        "first_k_dense_replace": 1,
        "moe_layer_freq_source": 1,
        "moe_intermediate_size": 12,
        "routed_expert_hidden_size": 16,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 2,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "use_grouped_topk": True,
        "latent_moe_use_norm": True,
    }

    converted = SimpleNamespace(**constructor, **aliases)
    config.assert_transformer_config_contract(converted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda fields: fields.pop("first_k_dense_replace"), "missing"),
        (lambda fields: fields.update(moe_router_topk=3), "mismatched"),
    ],
)
def test_transformer_config_contract_fails_loudly_on_loss_or_drift(mutation, message):
    config = K3Config._from_hf_dict(_tiny_text_config())
    constructor, aliases = config.transformer_config_contract()
    fields = {**constructor, **aliases}
    mutation(fields)

    with pytest.raises(RuntimeError, match=message):
        config.assert_transformer_config_contract(SimpleNamespace(**fields))


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


def test_config_fails_loudly_on_multimodal_inputs():
    with pytest.raises(NotImplementedError, match="text inputs only"):
        K3Config.ensure_text_only_inputs(pixel_values=object())
    with pytest.raises(NotImplementedError, match="MoonViT-V2"):
        K3Config.ensure_text_only_inputs(images=[object()])
