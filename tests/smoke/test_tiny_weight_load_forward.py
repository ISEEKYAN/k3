from __future__ import annotations

import torch

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


def test_complete_tiny_model_runs_hybrid_forward_backward():
    torch.manual_seed(7)
    config = _tiny_config()
    model = K3Model(config)
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
