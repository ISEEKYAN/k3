from __future__ import annotations

import pytest
import torch

from mlite_k3.lite.checkpoint import (
    audit_k3_weight_index,
    get_hf_weight,
    parse_k3_quantization_metadata,
)


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _quantization_config() -> dict:
    return {
        "text_config": {
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "format": "mxfp4-pack-quantized",
                        "targets": ["Linear"],
                        "weights": {
                            "dynamic": False,
                            "group_size": 32,
                            "num_bits": 4,
                            "scale_dtype": "torch.uint8",
                            "symmetric": True,
                            "type": "float",
                        },
                    }
                },
                "format": "mxfp4-pack-quantized",
                "ignore": [
                    r"re:.*self_attn.*",
                    r"re:.*shared_experts.*",
                    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
                    r"re:.*lm_head.*",
                    r"re:.*vision_tower.*",
                    r"re:.*mm_projector.*",
                ],
                "quant_method": "compressed-tensors",
            }
        }
    }


def _independent_mxfp4_reference(
    packed: torch.Tensor, encoded_scale: torch.Tensor
) -> torch.Tensor:
    """Decode the public compressed-tensors contract without MLite helpers."""
    values = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
    )
    low = values[(packed & 0x0F).long()]
    high = values[(packed >> 4).long()]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    scale = torch.exp2(encoded_scale.to(torch.int32) - 127).float()
    return unpacked * scale.repeat_interleave(32, dim=-1)


def test_release_mxfp4_pair_matches_independent_compressed_tensors_formula():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[0::2] | (codes[1::2] << 4)).repeat(2, 1)
    scale = torch.tensor([[127], [129]], dtype=torch.uint8)
    reader = _Reader(
        {
            "experts.0.w1.weight_packed": packed,
            "experts.0.w1.weight_scale": scale,
        }
    )

    got = get_hf_weight(reader, "experts.0.w1.weight")
    expected = _independent_mxfp4_reference(packed, scale)

    assert got.dtype == torch.float32
    assert torch.equal(got, expected)


def test_public_compressed_tensors_metadata_is_frozen():
    metadata = parse_k3_quantization_metadata(_quantization_config())

    assert metadata.format == "mxfp4-pack-quantized"
    assert metadata.group_size == 32
    assert metadata.num_bits == 4
    assert metadata.scale_dtype == "torch.uint8"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda config: config["text_config"]["quantization_config"].update(
                {"quant_method": "other"}
            ),
            "quant_method",
        ),
        (
            lambda config: config["text_config"]["quantization_config"][
                "config_groups"
            ]["group_0"]["weights"].update({"group_size": 128}),
            "weight metadata",
        ),
        (
            lambda config: config["text_config"]["quantization_config"][
                "ignore"
            ].remove(r"re:.*shared_experts.*"),
            "ignore list",
        ),
    ],
)
def test_compressed_tensors_metadata_drift_fails_loudly(mutation, message):
    config = _quantization_config()
    mutation(config)

    with pytest.raises(ValueError, match=message):
        parse_k3_quantization_metadata(config)


def test_plain_bf16_weight_is_not_reinterpreted():
    weight = torch.randn(3, 5, dtype=torch.bfloat16)
    reader = _Reader({"shared_experts.w1.weight": weight})

    got = get_hf_weight(reader, "shared_experts.w1.weight")

    assert got is weight


def test_packed_weight_requires_its_scale():
    reader = _Reader(
        {"experts.0.w1.weight_packed": torch.zeros(2, 16, dtype=torch.uint8)}
    )

    with pytest.raises(KeyError, match="weight_scale"):
        get_hf_weight(reader, "experts.0.w1.weight")


def test_weight_index_audit_requires_complete_colocated_expert_pairs():
    weight_map = {
        "language_model.model.layers.1.self_attn.q_proj.weight": "b.safetensors",
    }
    for projection in ("w1", "w2", "w3"):
        base = (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            f"{projection}.weight"
        )
        weight_map[f"{base}_packed"] = "a.safetensors"
        weight_map[f"{base}_scale"] = "a.safetensors"

    summary = audit_k3_weight_index(
        {"weight_map": weight_map},
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_experts=1,
    )

    assert summary.quantized_weights == 3
    assert summary.plain_tensors == 1
    assert summary.shards == 2


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda weights: weights.pop(
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
            ),
            "missing weight_scale",
        ),
        (
            lambda weights: weights.__setitem__(
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",
                "other.safetensors",
            ),
            "different shards",
        ),
        (
            lambda weights: weights.__setitem__(
                "language_model.model.layers.1.self_attn.q_proj.weight_packed",
                "a.safetensors",
            ),
            "outside routed experts",
        ),
    ],
)
def test_weight_index_audit_fails_loudly_on_incomplete_or_misrouted_pairs(
    mutate, message
):
    weight_map = {
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": "a.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale": "a.safetensors",
    }
    mutate(weight_map)

    with pytest.raises(ValueError, match=message):
        audit_k3_weight_index({"weight_map": weight_map})


def test_weight_index_audit_checks_every_layer_expert_and_projection():
    weight_map = {}
    for projection in ("w1", "w2"):
        base = (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            f"{projection}.weight"
        )
        weight_map[f"{base}_packed"] = "a.safetensors"
        weight_map[f"{base}_scale"] = "a.safetensors"

    with pytest.raises(ValueError, match="missing expected routed weight.*w3"):
        audit_k3_weight_index(
            {"weight_map": weight_map},
            num_hidden_layers=2,
            first_k_dense_replace=1,
            num_experts=1,
        )
