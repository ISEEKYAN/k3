from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import mlite_k3.lite.checkpoint as checkpoint
from mlite_k3.lite.checkpoint import (
    K3CheckpointManifest,
    K3QuantizationMetadata,
    K3WeightSpec,
    WeightIndexAudit,
    audit_k3_weight_spec_sources,
    audit_k3_weight_index,
    export_hf_weights,
    get_hf_weight,
    save_hf_weights,
    parse_k3_quantization_metadata,
)


def _single_rank_parallel_state():
    return SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
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
    vocab_size = 64
    kda_num_heads = 3
    kda_head_dim = 4
    intermediate_size = 3
    shared_expert_intermediate_size = 3
    moe_intermediate_size = 3
    routed_expert_hidden_size = 4

    @staticmethod
    def attention_type(layer_index: int) -> str:
        return ("kda", "mla")[layer_index]


def test_k3_weight_spec_covers_text_backbone_with_k3_specific_expert_names():
    mapping = K3WeightSpec(_TinyConfig()).weight_map()

    assert mapping["embed_tokens.embedding.weight"] == [
        "language_model.model.embed_tokens.weight"
    ]
    assert mapping["lm_head.col.linear.weight"] == ["language_model.lm_head.weight"]
    assert mapping["layers.0.self_attention.q_proj.linear.weight"] == [
        "language_model.model.layers.0.self_attn.q_proj.weight"
    ]
    assert mapping["layers.0.self_attention.q_conv1d.weight"] == [
        "language_model.model.layers.0.self_attn.q_conv1d.weight"
    ]
    assert mapping["layers.1.self_attention.linear_q_down_proj.weight"] == [
        "language_model.model.layers.1.self_attn.q_a_proj.weight"
    ]
    assert mapping["layers.1.self_attention.linear_q_up_proj.linear.weight"] == [
        "language_model.model.layers.1.self_attn.q_b_proj.weight"
    ]
    assert mapping["layers.1.moe.experts.fc1.weight0"] == [
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight",
    ]
    assert mapping["layers.1.moe.experts.fc2.weight0"] == [
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
    ]
    assert not any("vision" in name for names in mapping.values() for name in names)


def test_k3_weight_spec_implements_hf_weights_parallel_contract():
    from megatron.lite.primitive.ckpt.hf_weights import HFWeights

    spec = K3WeightSpec(_TinyConfig())

    assert isinstance(spec, HFWeights)
    assert spec.num_experts == 1
    assert spec.qkv_spec("anything") is None
    assert spec.tp_spec("embed_tokens.embedding.weight") == (0, 0)
    assert spec.tp_spec("lm_head.col.linear.weight") == (0, 0)
    for suffix in (
        "q_proj.linear.weight",
        "k_proj.linear.weight",
        "v_proj.linear.weight",
        "f_b_proj.linear.weight",
        "b_proj.linear.weight",
        "g_proj.linear.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
        "dt_bias",
    ):
        assert spec.tp_spec(f"layers.0.self_attention.{suffix}") == (0, 0)
    assert spec.tp_spec("layers.0.self_attention.o_proj.linear.weight") == (1, 0)
    assert spec.tp_spec("layers.1.self_attention.linear_q_up_proj.linear.weight") == (
        0,
        0,
    )
    assert spec.tp_spec("layers.1.self_attention.linear_kv_up_proj.linear.weight") == (
        0,
        0,
    )
    assert spec.tp_spec("layers.1.self_attention.linear_g_proj.linear.weight") == (
        0,
        0,
    )
    assert spec.tp_spec("layers.1.self_attention.linear_proj.linear.weight") == (
        1,
        0,
    )
    assert spec.tp_spec("layers.1.moe.experts.fc1.weight0") == (0, 1)
    assert spec.tp_spec("layers.1.moe.experts.fc2.weight0") == (1, 1)
    assert spec.tp_spec("layers.1.moe.router.gate.weight") is None

    expert = "layers.1.moe.experts.fc1.weight0"
    assert spec.is_expert(expert)
    assert spec.expert_global_id(expert) == 0
    assert spec.expert_local_name(expert, 3) == "layers.1.moe.experts.fc1.weight3"
    assert spec.weight_map()["layers.1.moe.router.expert_bias"] == [
        "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
    ]


def test_k3_rollout_layout_exports_every_segment_bit_exactly() -> None:
    class TinyExportConfig(_TinyConfig):
        num_hidden_layers = 1
        kda_num_heads = 2
        kda_head_dim = 8

    layout = K3WeightSpec(TinyExportConfig()).kda_rollout_layout
    source = {
        segment.name: torch.full((segment.rows, 3), index, dtype=torch.int32)
        for index, segment in enumerate(layout.segments, start=1)
    }

    fused = layout.fuse_ordered(
        tuple((segment.name, source[segment.name]) for segment in layout.segments)
    )
    restored = layout.split(fused)

    assert tuple(restored) == ("q", "k", "v", "g", "f_a", "b")
    for segment in layout.segments:
        assert torch.equal(restored[segment.name], source[segment.name])


def test_k3_declares_the_rollout_kda_fusion_geometry() -> None:
    layout = K3WeightSpec(_TinyConfig()).kda_rollout_layout

    assert tuple(segment.name for segment in layout.segments) == (
        "q",
        "k",
        "v",
        "g",
        "f_a",
        "b",
    )
    assert tuple(segment.rows for segment in layout.segments) == (12, 12, 12, 12, 4, 3)
    assert tuple(segment.replicated for segment in layout.segments) == (
        False,
        False,
        False,
        False,
        True,
        False,
    )


def test_k3_rollout_layout_rejects_wrong_order_and_head_count() -> None:
    layout = K3WeightSpec(_TinyConfig()).kda_rollout_layout
    source = {segment.name: torch.zeros(segment.rows, 2) for segment in layout.segments}
    wrong_order = tuple(
        (name, source[name]) for name in ("q", "v", "k", "g", "f_a", "b")
    )

    with pytest.raises(ValueError, match=r"segment order mismatch.*k.*v"):
        layout.fuse_ordered(wrong_order)

    source["q"] = torch.zeros(source["q"].size(0) + layout.segments[0].head_dim, 2)
    with pytest.raises(ValueError, match=r"q.*12 rows.*got 16"):
        layout.fuse_ordered(
            tuple((segment.name, source[segment.name]) for segment in layout.segments)
        )


def test_k3_rollout_layout_splits_each_mxfp4_scale_bit_exactly() -> None:
    layout = K3WeightSpec(_TinyConfig()).kda_rollout_layout
    packed = {
        segment.name: torch.full((segment.rows, 2), index, dtype=torch.uint8)
        for index, segment in enumerate(layout.segments, start=1)
    }
    scales = {
        segment.name: torch.full((segment.rows, 1), index + 10, dtype=torch.uint8)
        for index, segment in enumerate(layout.segments, start=1)
    }

    restored = layout.split_quantized(layout.fuse(packed), layout.fuse(scales))

    for segment in layout.segments:
        assert torch.equal(restored[segment.name].packed, packed[segment.name])
        assert torch.equal(restored[segment.name].scale, scales[segment.name])


def test_k3_weight_spec_materializes_mxfp4_sources_from_manifest():
    manifest = K3CheckpointManifest(
        quantization=K3QuantizationMetadata(
            format="mxfp4-pack-quantized",
            group_size=32,
            num_bits=4,
            scale_dtype="torch.uint8",
            ignored_modules=frozenset(),
        ),
        weights=WeightIndexAudit(quantized_weights=3, plain_tensors=0, shards=1),
    )
    spec = K3WeightSpec(_TinyConfig(), manifest=manifest)
    native = "layers.1.moe.experts.fc1.weight0"
    sources = spec.weight_map()[native]

    assert sources == [
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_packed",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight_scale",
    ]
    assert spec.raw_hf_source(native, 0, sources[0]) is True
    assert spec.raw_hf_source(native, 1, sources[1]) is True

    packed = torch.zeros(3, 16, dtype=torch.uint8)
    scale = torch.full((3, 1), 127, dtype=torch.uint8)
    materialized = spec.hf_to_native(native, [packed, scale, packed, scale])

    assert materialized.shape == (6, 32)
    assert materialized.dtype == torch.float32


def test_k3_load_delegates_to_shared_hfweights_primitive(monkeypatch):
    manifest = K3CheckpointManifest(
        quantization=K3QuantizationMetadata(
            format="mxfp4-pack-quantized",
            group_size=32,
            num_bits=4,
            scale_dtype="torch.uint8",
            ignored_modules=frozenset(),
        ),
        weights=WeightIndexAudit(quantized_weights=3, plain_tensors=0, shards=1),
    )
    calls = []

    class Reader:
        def __init__(self, path):
            assert path == "checkpoint"
            self.index = {}

    monkeypatch.setattr(checkpoint, "inspect_hf_checkpoint", lambda path: manifest)
    monkeypatch.setattr(
        checkpoint, "audit_k3_weight_spec_sources", lambda spec, index: 0
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader",
        Reader,
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.load_hf_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    model = nn.Module()
    ps = object()

    result = checkpoint.load_hf_weights(model, "checkpoint", _TinyConfig(), ps)

    assert result is manifest
    args, kwargs = calls.pop()
    assert args[0] is model
    assert args[1] == "checkpoint"
    assert isinstance(args[2], K3WeightSpec)
    assert args[2].manifest is manifest
    assert args[3] is ps
    assert kwargs == {"vocab_size": 64}


def test_k3_export_delegates_to_shared_hfweights_primitive(monkeypatch):
    calls = []
    sentinel = torch.tensor([1.0])

    def fake_export(*args, **kwargs):
        calls.append((args, kwargs))
        yield "language_model.model.norm.weight", sentinel

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.export_hf_weights",
        fake_export,
    )
    model = nn.Module()
    ps = object()

    exported = list(export_hf_weights(model, _TinyConfig(), ps))

    assert exported == [("language_model.model.norm.weight", sentinel)]
    args, kwargs = calls.pop()
    assert args[0] is model
    assert isinstance(args[1], K3WeightSpec)
    assert args[2] is ps
    assert kwargs == {"vocab_size": 64, "rank0_only": False}


def _model_with_preserved_mxfp4_encoding():
    manifest = K3CheckpointManifest(
        quantization=K3QuantizationMetadata(
            format="mxfp4-pack-quantized",
            group_size=32,
            num_bits=4,
            scale_dtype="torch.uint8",
            ignored_modules=frozenset(),
        ),
        weights=WeightIndexAudit(quantized_weights=3, plain_tensors=0, shards=1),
    )
    spec = K3WeightSpec(_TinyConfig(), manifest=manifest)
    native_name = "layers.1.moe.experts.fc2.weight0"
    packed = torch.arange(32 * 16, dtype=torch.int32).to(torch.uint8).view(32, 16)
    scale = torch.full((32, 1), 121, dtype=torch.uint8)
    logical = spec.hf_to_native(native_name, [packed, scale]).to(torch.bfloat16)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_indices = [1]
            layer = nn.Module()
            layer.moe = nn.Module()
            layer.moe.experts = nn.Module()
            layer.moe.experts.fc2 = _TinyGroupedLinear(logical.shape)
            self.layers = nn.ModuleList([layer])
            layer.moe.experts.fc2.weight0.data.copy_(logical)

    model = TinyModel()
    checkpoint._attach_mxfp4_checkpoint_adapter(
        model,
        spec,
        _single_rank_parallel_state(),
    )

    return model, packed, scale


def test_mxfp4_export_reuses_model_owned_release_encoding_bit_exactly():
    model, packed, scale = _model_with_preserved_mxfp4_encoding()
    exported = dict(
        export_hf_weights(
            model,
            _TinyConfig(),
            _single_rank_parallel_state(),
            target="mxfp4",
            cpu=True,
        )
    )
    prefix = "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
    assert torch.equal(exported[f"{prefix}_packed"], packed)
    assert torch.equal(exported[f"{prefix}_scale"], scale)


def test_mxfp4_export_requantizes_after_model_weight_changes():
    model, packed, scale = _model_with_preserved_mxfp4_encoding()
    model.layers[0].moe.experts.fc2.weight0.data.fill_(1.0)

    exported = dict(
        export_hf_weights(
            model,
            _TinyConfig(),
            _single_rank_parallel_state(),
            target="mxfp4",
            cpu=True,
        )
    )
    prefix = "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
    assert not (
        torch.equal(exported[f"{prefix}_packed"], packed)
        and torch.equal(exported[f"{prefix}_scale"], scale)
    )


def test_k3_weight_spec_removes_release_a_log_zero_padding():
    spec = K3WeightSpec(_TinyConfig())
    active = torch.arange(spec.config.kda_num_heads, dtype=torch.float32)
    padded = torch.cat((active, torch.zeros(5)))

    native = spec.hf_to_native(
        "layers.0.self_attention.A_log",
        [padded],
    )

    assert torch.equal(native, active)


def test_k3_weight_spec_rejects_nonzero_a_log_padding():
    spec = K3WeightSpec(_TinyConfig())
    padded = torch.cat(
        (
            torch.zeros(spec.config.kda_num_heads),
            torch.tensor([0.0, 1.0]),
        )
    )

    with pytest.raises(ValueError, match="A_log padding must be exactly zero"):
        spec.hf_to_native("layers.0.self_attention.A_log", [padded])


def test_k3_weight_spec_reshapes_release_dt_bias():
    spec = K3WeightSpec(_TinyConfig())
    flattened = torch.arange(
        spec.config.kda_num_heads * spec.config.kda_head_dim,
        dtype=torch.float32,
    )

    native = spec.hf_to_native(
        "layers.0.self_attention.dt_bias",
        [flattened],
    )

    assert native.shape == (
        spec.config.kda_num_heads,
        spec.config.kda_head_dim,
    )
    assert torch.equal(native.flatten(), flattened)


def test_k3_weight_spec_restores_release_kda_layouts_on_export():
    spec = K3WeightSpec(_TinyConfig())
    a_log = torch.arange(spec.config.kda_num_heads, dtype=torch.float32)
    dt_bias = torch.arange(
        spec.config.kda_num_heads * spec.config.kda_head_dim,
        dtype=torch.float32,
    ).reshape(spec.config.kda_num_heads, spec.config.kda_head_dim)

    [(a_log_name, exported_a_log)] = spec.native_to_hf(
        "layers.0.self_attention.A_log",
        a_log,
    )
    [(dt_bias_name, exported_dt_bias)] = spec.native_to_hf(
        "layers.0.self_attention.dt_bias",
        dt_bias,
    )

    assert a_log_name.endswith(".self_attn.A_log")
    assert exported_a_log.shape == (128,)
    assert torch.equal(exported_a_log[: spec.config.kda_num_heads], a_log)
    assert torch.count_nonzero(exported_a_log[spec.config.kda_num_heads :]) == 0
    assert dt_bias_name.endswith(".self_attn.dt_bias")
    assert exported_dt_bias.shape == (
        spec.config.kda_num_heads * spec.config.kda_head_dim,
    )
    assert torch.equal(exported_dt_bias, dt_bias.flatten())


def test_k3_weight_spec_rejects_wrong_dt_bias_size():
    spec = K3WeightSpec(_TinyConfig())
    wrong = torch.zeros(spec.config.kda_num_heads * spec.config.kda_head_dim + 1)

    with pytest.raises(ValueError, match="dt_bias must contain exactly 12 values"):
        spec.hf_to_native("layers.0.self_attention.dt_bias", [wrong])


def test_k3_weight_spec_applies_required_layout_transforms():
    spec = K3WeightSpec(_TinyConfig())
    gate = torch.randn(3, 4)
    up = torch.randn(3, 4)
    conv = torch.randn(4, 1, 3)

    fused = spec.hf_to_native("layers.1.moe.experts.fc1.weight0", [gate, up])
    preserved = spec.hf_to_native("layers.0.self_attention.q_conv1d.weight", [conv])

    assert torch.equal(fused, torch.cat((gate, up), dim=0))
    assert preserved.shape == (4, 1, 3)
    assert torch.equal(preserved, conv)


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
    native_name = "layers.1.moe.experts.fc1.weight0"

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


class _TinyGroupedLinear(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_parameter(
            "weight0",
            nn.Parameter(torch.randn(*shape, dtype=torch.bfloat16)),
        )


class _TinyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = _TinyGroupedLinear((6, 32))
        self.fc2 = _TinyGroupedLinear((32, 32))


class _TinyRouter(nn.Module):
    pass


class _TinyMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _TinyExperts()
        self.router = _TinyRouter()


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
    shared_expert_intermediate_size = 3
    moe_intermediate_size = 3

    @staticmethod
    def attention_type(layer_index: int) -> str:
        assert layer_index == 0
        return "kda"


def test_mxfp4_resync_export_only_packs_routed_expert_weights():
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"
    weights = [
        (f"{prefix}.w1.weight", torch.randn(3, 32, dtype=torch.bfloat16)),
        (f"{prefix}.w3.weight", torch.randn(3, 32, dtype=torch.bfloat16)),
    ]

    exported = dict(checkpoint._export_mxfp4_weights(iter(weights)))

    assert set(exported) == {
        f"{prefix}.w1.weight_packed",
        f"{prefix}.w1.weight_scale",
        f"{prefix}.w3.weight_packed",
        f"{prefix}.w3.weight_scale",
    }
    assert exported[f"{prefix}.w1.weight_packed"].dtype == torch.int8
    assert exported[f"{prefix}.w1.weight_scale"].dtype == torch.uint8


@pytest.mark.parametrize("target", ("bf16", "mxfp4"))
def test_save_hf_weights_writes_shards_and_roundtrips_real_files(
    tmp_path, target, monkeypatch
):
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    source = _TinyExpertModel()
    source.layers[0].moe.router.register_buffer(
        "expert_bias", torch.tensor([0.25], dtype=torch.float32)
    )
    spec = K3WeightSpec(_ExpertConfig())
    fc1 = source.layers[0].moe.experts.fc1.weight0.detach().cpu()
    fc2 = source.layers[0].moe.experts.fc2.weight0.detach().cpu()
    exported = spec.native_to_hf("layers.0.moe.experts.fc1.weight0", fc1)
    exported.extend(spec.native_to_hf("layers.0.moe.experts.fc2.weight0", fc2))
    exported.append(
        (
            "language_model.model.layers.0.block_sparse_moe.gate."
            "e_score_correction_bias",
            source.layers[0].moe.router.expert_bias,
        )
    )

    def fake_export(*_args, **_kwargs):
        weights = iter(exported)
        yield from (
            checkpoint._export_mxfp4_weights(weights) if target == "mxfp4" else weights
        )

    monkeypatch.setattr(checkpoint, "export_hf_weights", fake_export)
    summary = save_hf_weights(
        source,
        tmp_path,
        _ExpertConfig(),
        _single_rank_parallel_state(),
        target=target,
        max_shard_size_bytes=64,
    )

    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    assert summary.shards > 1
    assert index["metadata"]["format"] == target
    assert index["metadata"]["total_size"] > 0
    assert not list(tmp_path.glob("*.tmp"))
    assert all(
        (tmp_path / filename).is_file()
        for filename in set(index["weight_map"].values())
    )
    if target == "mxfp4":
        for key, filename in index["weight_map"].items():
            if key.endswith("_packed"):
                assert (
                    index["weight_map"][key.removesuffix("_packed") + "_scale"]
                    == filename
                )

    reader = SafeTensorReader(str(tmp_path))
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"
    if target == "bf16":
        native = spec.hf_to_native(
            "layers.0.moe.experts.fc1.weight0",
            [
                reader.get_tensor(f"{prefix}.w1.weight"),
                reader.get_tensor(f"{prefix}.w3.weight"),
            ],
        )
        assert torch.equal(native, source.layers[0].moe.experts.fc1.weight0)
    assert torch.equal(
        reader.get_tensor(
            "language_model.model.layers.0.block_sparse_moe.gate."
            "e_score_correction_bias"
        ),
        source.layers[0].moe.router.expert_bias.to(torch.bfloat16),
    )


def test_mxfp4_save_rejects_incomplete_routed_grid_before_publishing_index(
    tmp_path, monkeypatch
):
    prefix = "language_model.model.layers.0.block_sparse_moe.experts.0"

    def incomplete_export(*_args, **_kwargs):
        yield f"{prefix}.w1.weight_packed", torch.zeros(1, 16, dtype=torch.int8)
        yield f"{prefix}.w1.weight_scale", torch.zeros(1, 1, dtype=torch.uint8)

    monkeypatch.setattr(checkpoint, "export_hf_weights", incomplete_export)

    with pytest.raises(ValueError, match="missing expected routed weight"):
        save_hf_weights(
            _TinyExpertModel(),
            tmp_path,
            _ExpertConfig(),
            _single_rank_parallel_state(),
            target="mxfp4",
        )

    assert not (tmp_path / "model.safetensors.index.json").exists()
