from __future__ import annotations

import json

import pytest
import torch

from mlite_k3.checkpoint_validation import (
    build_capability_matrix,
    build_structural_samples,
    validate_reader_roundtrip,
    write_validation_report,
)
from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import K3WeightSpec


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _TinyConfig:
    num_hidden_layers = 3
    first_k_dense_replace = 1
    num_experts = 2

    @staticmethod
    def attention_type(layer_index: int) -> str:
        return ("kda", "mla", "kda")[layer_index]


def _plain_checkpoint(spec: K3WeightSpec) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for source_names in spec.weight_map().values():
        for source_name in source_names:
            if source_name in tensors:
                continue
            if source_name.endswith(
                ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")
            ):
                tensor = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
            elif source_name.endswith((".w1.weight", ".w3.weight")):
                tensor = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
            elif source_name.endswith(".w2.weight"):
                tensor = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
            else:
                tensor = torch.tensor([len(tensors)], dtype=torch.bfloat16)
            tensors[source_name] = tensor
    return tensors


def test_real_shape_defaults_have_reproducible_structural_samples():
    samples = build_structural_samples(K3Config())

    assert samples["rule"] == (
        "all MLA layers plus adjacent KDA boundaries; first/last layer; "
        "expert positions 0, 1/4, 1/2, 3/4, and last"
    )
    assert samples["layer_count"] == 93
    assert samples["expert_count"] == 896
    assert samples["layers"][0] == {"index": 0, "attention": "kda", "ffn": "dense"}
    assert samples["layers"][-1] == {"index": 92, "attention": "mla", "ffn": "moe"}
    assert {item["attention"] for item in samples["layers"]} == {"kda", "mla"}
    assert {item["ffn"] for item in samples["layers"]} == {"dense", "moe"}
    assert samples["experts"] == [0, 223, 447, 671, 895]


def test_capability_matrix_has_every_required_cell_and_explicit_mtp_scope():
    matrix = build_capability_matrix()

    assert matrix["columns"] == [
        "load",
        "save",
        "export_bf16",
        "export_mxfp4",
        "qat_canonical",
        "shard_rules",
    ]
    assert [row["structure"] for row in matrix["rows"]] == [
        "dense",
        "moe",
        "mla",
        "kda",
        "shared_expert",
        "router_expert_bias",
        "mtp",
    ]
    for row in matrix["rows"]:
        assert set(row["cells"]) == set(matrix["columns"])
        assert all(
            cell["status"] in {"covered", "excluded_by_contract", "out_of_scope"}
            for cell in row["cells"].values()
        )
    assert {cell["status"] for cell in matrix["rows"][-1]["cells"].values()} == {
        "out_of_scope"
    }


def test_reader_roundtrip_checks_every_mapped_source_shape_dtype_and_bits():
    spec = K3WeightSpec(_TinyConfig())
    tensors = _plain_checkpoint(spec)

    summary = validate_reader_roundtrip(_Reader(tensors), spec)

    assert summary["bitwise_equal"] is True
    assert summary["native_tensors"] == len(spec.weight_map())
    assert summary["source_tensors"] == len(tensors)
    assert len(summary["source_key_sha256"]) == 64
    assert summary["dtypes"] == {"torch.bfloat16": len(tensors)}


def test_report_is_not_published_when_any_tensor_is_not_bitwise_equal(tmp_path):
    class _CorruptingSpec(K3WeightSpec):
        def native_to_hf(self, native_name, tensor):
            restored = super().native_to_hf(native_name, tensor)
            if native_name == "embed_tokens.weight":
                name, value = restored[0]
                value = value.clone()
                value.flatten()[0] = -0.0
                return [(name, value)]
            return restored

    spec = _CorruptingSpec(_TinyConfig())
    output = tmp_path / "summary.json"

    with pytest.raises(AssertionError, match="bitwise mismatch"):
        write_validation_report(
            _Reader(_plain_checkpoint(spec)),
            spec,
            _TinyConfig(),
            output,
            revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        )

    assert not output.exists()


def test_successful_report_is_machine_readable_and_complete(tmp_path):
    spec = K3WeightSpec(_TinyConfig())
    output = tmp_path / "summary.json"

    report = write_validation_report(
        _Reader(_plain_checkpoint(spec)),
        spec,
        _TinyConfig(),
        output,
        revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    )

    assert json.loads(output.read_text()) == report
    assert report["checkpoint"]["bitwise_equal"] is True
    assert report["coverage"]["samples"]["layer_count"] == 3
    assert report["coverage"]["matrix"]["rows"][-1]["structure"] == "mtp"
