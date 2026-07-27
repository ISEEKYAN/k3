from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    audit_k3_weight_index,
    get_hf_weight,
    iter_hf_weights,
    load_weights_from_reader,
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


class _TinyConfig:
    num_hidden_layers = 2
    first_k_dense_replace = 1
    num_experts = 1

    @staticmethod
    def attention_type(layer_index: int) -> str:
        return ("kda", "mla")[layer_index]


def test_k3_weight_spec_covers_text_backbone_with_k3_specific_expert_names():
    mapping = K3WeightSpec(_TinyConfig()).weight_map()

    assert mapping["embed_tokens.weight"] == [
        "language_model.model.embed_tokens.weight"
    ]
    assert mapping["layers.0.self_attention.q_conv1d.conv.weight"] == [
        "language_model.model.layers.0.self_attn.q_conv1d.weight"
    ]
    assert mapping["layers.1.self_attention.q_a_proj.weight"] == [
        "language_model.model.layers.1.self_attn.q_a_proj.weight"
    ]
    assert mapping["layers.1.moe.experts.0.gate_up.weight"] == [
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight",
    ]
    assert mapping["layers.1.moe.experts.0.down.weight"] == [
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
    ]
    assert not any("vision" in name for names in mapping.values() for name in names)


def test_k3_weight_spec_applies_only_the_two_required_layout_transforms():
    spec = K3WeightSpec(_TinyConfig())
    gate = torch.randn(3, 4)
    up = torch.randn(3, 4)
    conv = torch.randn(4, 3)

    fused = spec.hf_to_native("layers.1.moe.experts.0.gate_up.weight", [gate, up])
    expanded = spec.hf_to_native("layers.0.self_attention.q_conv1d.conv.weight", [conv])

    assert torch.equal(fused, torch.cat((gate, up), dim=0))
    assert expanded.shape == (4, 1, 3)
    assert torch.equal(expanded.squeeze(1), conv)


def test_k3_weight_spec_source_audit_accepts_plain_and_paired_weights():
    spec = K3WeightSpec(_TinyConfig())
    release_index = {}
    for source_names in spec.weight_map().values():
        for source_name in source_names:
            if ".experts." in source_name:
                release_index[f"{source_name}_packed"] = "a.safetensors"
                release_index[f"{source_name}_scale"] = "a.safetensors"
            else:
                release_index[source_name] = "a.safetensors"

    assert audit_k3_weight_spec_sources(spec, release_index) == len(
        {name for names in spec.weight_map().values() for name in names}
    )

    release_index.pop("language_model.model.layers.0.self_attn.A_log")
    with pytest.raises(ValueError, match="missing mapped K3 tensor.*A_log"):
        audit_k3_weight_spec_sources(spec, release_index)


def test_k3_weight_spec_roundtrips_dequantized_expert_layout():
    spec = K3WeightSpec(_TinyConfig())
    gate = torch.randn(3, 32, dtype=torch.bfloat16)
    up = torch.randn(3, 32, dtype=torch.bfloat16)
    native_name = "layers.1.moe.experts.0.gate_up.weight"

    native = spec.hf_to_native(native_name, [gate, up])
    restored = dict(spec.native_to_hf(native_name, native))

    assert torch.equal(
        restored["language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"],
        gate,
    )
    assert torch.equal(
        restored["language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight"],
        up,
    )


class _TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Linear(32, 6, bias=False, dtype=torch.bfloat16)


class _TinyMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_TinyExpert()])


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _TinyMoe()


class _TinyExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer()])


class _ExpertConfig:
    num_hidden_layers = 1
    first_k_dense_replace = 0
    num_experts = 1

    @staticmethod
    def attention_type(layer_index: int) -> str:
        assert layer_index == 0
        return "kda"


def test_streaming_loader_dequantizes_and_copies_one_tiny_expert():
    low_codes = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.uint8)
    high_codes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.uint8)
    packed_row = (low_codes | (high_codes << 4)).repeat(2)
    packed = packed_row.repeat(3, 1)
    scale = torch.full((3, 1), 127, dtype=torch.uint8)
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"
    reader = _Reader(
        {
            f"{prefix}.w1.weight_packed": packed,
            f"{prefix}.w1.weight_scale": scale,
            f"{prefix}.w3.weight_packed": packed ^ 0x88,
            f"{prefix}.w3.weight_scale": scale,
        }
    )
    model = _TinyExpertModel()

    loaded = load_weights_from_reader(model, reader, K3WeightSpec(_ExpertConfig()))

    expected_gate = _independent_mxfp4_reference(packed, scale)
    expected_up = _independent_mxfp4_reference(packed ^ 0x88, scale)
    assert loaded == 1
    assert torch.equal(
        model.layers[0].moe.experts[0].gate_up.weight.float(),
        torch.cat((expected_gate, expected_up), dim=0),
    )


def test_tiny_mxfp4_load_plain_bf16_export_and_reload_is_bitwise():
    low_codes = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.uint8)
    high_codes = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.uint8)
    packed = (low_codes | (high_codes << 4)).repeat(3, 2)
    scale = torch.full((3, 1), 127, dtype=torch.uint8)
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"
    quantized = _Reader(
        {
            f"{prefix}.w1.weight_packed": packed,
            f"{prefix}.w1.weight_scale": scale,
            f"{prefix}.w3.weight_packed": packed ^ 0x88,
            f"{prefix}.w3.weight_scale": scale,
        }
    )
    spec = K3WeightSpec(_ExpertConfig())
    first = _TinyExpertModel()
    second = _TinyExpertModel()

    load_weights_from_reader(first, quantized, spec)
    plain_bf16 = dict(iter_hf_weights(first, spec))
    load_weights_from_reader(second, _Reader(plain_bf16), spec)

    assert set(plain_bf16) == {f"{prefix}.w1.weight", f"{prefix}.w3.weight"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in plain_bf16.values())
    assert torch.equal(
        first.layers[0].moe.experts[0].gate_up.weight,
        second.layers[0].moe.experts[0].gate_up.weight,
    )


def test_streaming_loader_fails_on_unmapped_native_parameter():
    model = nn.Module()
    model.register_parameter("unexpected", nn.Parameter(torch.zeros(1)))

    with pytest.raises(KeyError, match="no K3 checkpoint mapping"):
        load_weights_from_reader(model, _Reader({}), K3WeightSpec(_ExpertConfig()))


def test_streaming_loader_fails_on_wrong_checkpoint_shape():
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"
    reader = _Reader(
        {
            f"{prefix}.w1.weight": torch.zeros(3, 16),
            f"{prefix}.w3.weight": torch.zeros(3, 16),
        }
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        load_weights_from_reader(
            _TinyExpertModel(), reader, K3WeightSpec(_ExpertConfig())
        )
