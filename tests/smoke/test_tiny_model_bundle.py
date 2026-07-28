from __future__ import annotations

import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.protocol import ImplConfig, build_model


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


def test_real_mlite_protocol_builds_and_runs_tiny_hybrid_bundle_on_cpu():
    config = _tiny_config()
    bundle = build_model(
        config,
        impl_cfg=ImplConfig(device="cpu", dtype="float32"),
    )
    output = bundle.chunks[0](
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        labels=torch.tensor([[2, 3, 4, 5]]),
    )

    assert bundle.extras["validated_scope"] == "single_rank_reference"
    assert output["logits"].shape == (1, 4, 32)
    assert torch.isfinite(output["loss"])


def test_bfloat16_bundle_runs_with_float32_kda_gates_and_router_math():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(device="cpu", dtype="bfloat16"),
    )

    output = bundle.chunks[0](
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        labels=torch.tensor([[2, 3, 4, 5]]),
    )

    kda = bundle.chunks[0].layers[0].self_attention
    assert kda.A_log.dtype == torch.float32
    assert kda.dt_bias.dtype == torch.float32
    assert output["logits"].dtype == torch.bfloat16
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["loss"])
