from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from tools.build_proxy_checkpoint import build_proxy, keep_weight, proxy_config


def test_proxy_keeps_twelve_layers_and_one_sixteenth_of_experts():
    assert keep_weight(
        "language_model.model.layers.11.block_sparse_moe.experts.55.w1.weight_packed",
        layers=12,
        experts=56,
    )
    assert not keep_weight(
        "language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_packed",
        layers=12,
        experts=56,
    )
    assert not keep_weight(
        "language_model.model.layers.1.block_sparse_moe.experts.56.w1.weight_packed",
        layers=12,
        experts=56,
    )
    assert keep_weight(
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
        layers=12,
        experts=56,
    )
    assert keep_weight(
        "language_model.model.embed_tokens.weight", layers=12, experts=56
    )
    assert not keep_weight("vision_tower.blocks.0.weight", layers=12, experts=56)


def test_proxy_config_preserves_hybrid_schedule_boundary():
    source = {
        "model_type": "kimi_k3",
        "text_config": {
            "num_hidden_layers": 93,
            "num_experts": 896,
            "attn_res_block_size": 12,
            "num_experts_per_token": 16,
            "linear_attn_config": {
                "full_attn_layers": [4, 8, 12, 16],
                "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13],
            },
        },
    }

    result = proxy_config(source, layers=12, experts=56)

    assert result["text_config"]["num_hidden_layers"] == 12
    assert result["text_config"]["num_experts"] == 56
    assert result["text_config"]["num_experts_per_token"] == 16
    assert result["text_config"]["attn_res_block_size"] == 12
    assert result["text_config"]["linear_attn_config"]["full_attn_layers"] == [
        4,
        8,
        12,
    ]
    assert result["text_config"]["linear_attn_config"]["kda_layers"] == [
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
    ]
    assert source["text_config"]["num_hidden_layers"] == 93


def test_build_proxy_rewrites_index_and_drops_out_of_scope_tensors(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = {
        "model_type": "kimi_k3",
        "text_config": {
            "num_hidden_layers": 93,
            "num_experts": 896,
            "linear_attn_config": {
                "full_attn_layers": [4, 8, 12, 16],
                "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13],
            },
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    names = {
        "language_model.model.embed_tokens.weight": torch.ones(2, 2),
        "language_model.model.layers.11.block_sparse_moe.experts.55.w1.weight": (
            torch.full((2, 2), 55)
        ),
        "language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight": (
            torch.full((2, 2), 12)
        ),
        "language_model.model.layers.1.block_sparse_moe.experts.56.w1.weight": (
            torch.full((2, 2), 56)
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(names, source / shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard for name in names}})
    )

    build_proxy(source, output, layers=12, experts=56)

    result_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert result_index["metadata"]["proxy"] == {"layers": 12, "experts": 56}
    assert set(result_index["weight_map"]) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.11.block_sparse_moe.experts.55.w1.weight",
    }
    output_shard = next(output.glob("*.safetensors"))
    assert set(load_file(output_shard)) == set(result_index["weight_map"])
