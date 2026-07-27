from __future__ import annotations

import pytest
import torch

from mlite_k3.lite.checkpoint import (
    audit_k3_weight_index,
    get_hf_weight,
)


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


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
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed": "a.safetensors",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale": "a.safetensors",
        "language_model.model.layers.1.self_attn.q_proj.weight": "b.safetensors",
    }

    summary = audit_k3_weight_index({"weight_map": weight_map})

    assert summary.quantized_weights == 1
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
