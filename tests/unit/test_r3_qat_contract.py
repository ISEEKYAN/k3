from __future__ import annotations

import pytest

from mlite_k3.config import K3Config
from mlite_k3.lite import protocol
from mlite_k3.lite.protocol import ImplConfig, build_model


def _tiny_config() -> K3Config:
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


def test_protocol_exports_shared_zigzag_r3_contract():
    from megatron.lite.model import protocol_utils

    assert protocol.router_replay_roots is protocol_utils.router_replay_roots
    assert protocol.pack_routed_experts is protocol_utils.pack_routed_experts
    assert protocol.pack_r3_replay_mask is protocol_utils.pack_r3_replay_mask
    assert (
        protocol.unpack_thd_forward_output is protocol_utils.unpack_thd_forward_output
    )


def test_k3_parallel_kda_imports_against_latest_mlite():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src/mlite_k3/primitive/kda_parallel.py"
    ).read_text()

    assert "FullRankGatedDeltaNet," not in source
    assert "class _K3FullRankDeltaNet(nn.Module):" in source
    assert "GatedDeltaNet._headwise_cp2hp" in source


def test_latest_mlite_constructs_the_distributed_k3_model():
    pytest.importorskip("transformer_engine")
    from megatron.lite.primitive.parallel import ParallelState

    from mlite_k3.lite.model import K3ParallelModel

    model = K3ParallelModel(
        _tiny_config(),
        ParallelState(),
        deterministic=True,
    )

    assert len(model.layers) == 2
    assert model.layers[0].self_attention.__class__.__name__ == (
        "K3FullRankGatedDeltaNet"
    )


def test_mxfp4_qat_only_parametrizes_routed_expert_linears():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(
            device="cpu",
            dtype="float32",
            qat={"enabled": True, "format": "mxfp4", "ignore_patterns": ()},
        ),
    )
    names = {
        name
        for name, _ in bundle.chunks[0].named_parameters()
        if name.endswith(".parametrizations.weight.original")
    }

    assert names
    assert all(
        any(
            component in name.split(".")
            for component in (
                "routed_expert_down_proj",
                "experts",
                "routed_expert_up_proj",
            )
        )
        for name in names
    ), sorted(names)
    assert not any("shared_experts" in name for name in names)
    assert not any("self_attention" in name for name in names)
    assert not any(".mlp." in name for name in names)
    assert bundle.extras["qat"]["quantized_modules"] == len(names)


def test_disabled_qat_is_inert():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(device="cpu", dtype="float32"),
    )

    assert not any(
        "parametrizations" in name for name, _ in bundle.chunks[0].named_parameters()
    )
    assert bundle.extras["qat"]["quantized_modules"] == 0
