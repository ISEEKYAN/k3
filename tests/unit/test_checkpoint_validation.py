from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mlite_k3 import checkpoint_validation
from mlite_k3.checkpoint_validation import (
    build_capability_matrix,
    build_structural_samples,
    validate_checkpoint,
    validate_reader_roundtrip,
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
                tensor = torch.arange(12, dtype=torch.bfloat16).reshape(4, 1, 3)
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


def test_capability_matrix_defaults_to_not_covered_without_execution_evidence():
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
    ]
    for row in matrix["rows"]:
        assert set(row["cells"]) == set(matrix["columns"])
    for row in matrix["rows"]:
        assert {cell["status"] for cell in row["cells"].values()} == {"not-covered"}
        assert all(cell["evidence"] == [] for cell in row["cells"].values())


def test_capability_matrix_derives_only_the_explicitly_evidenced_cell():
    matrix = build_capability_matrix({"router_expert_bias.load": ("job:12345",)})
    rows = {row["structure"]: row["cells"] for row in matrix["rows"]}

    assert rows["router_expert_bias"]["load"] == {
        "status": "covered",
        "evidence": ["job:12345"],
    }
    assert rows["router_expert_bias"]["save"] == {
        "status": "not-covered",
        "evidence": [],
    }


@pytest.mark.parametrize(
    "evidence",
    (
        {"unknown.load": ("test:tests/gpu/test.py::test_case",)},
        {"dense.unknown": ("test:tests/gpu/test.py::test_case",)},
        {"dense.load": ()},
        {"dense.load": ("looks convincing but is not an execution id",)},
    ),
)
def test_capability_matrix_rejects_invalid_or_untraceable_evidence(evidence):
    with pytest.raises(RuntimeError, match="invalid K3 capability evidence"):
        build_capability_matrix(evidence)


def test_reader_roundtrip_checks_every_mapped_source_shape_dtype_and_bits():
    spec = K3WeightSpec(_TinyConfig())
    tensors = _plain_checkpoint(spec)

    summary = validate_reader_roundtrip(_Reader(tensors), spec)

    assert summary["bitwise_equal"] is True
    assert summary["native_tensors"] == len(spec.weight_map())
    assert summary["source_tensors"] == len(tensors)
    assert len(summary["source_key_sha256"]) == 64
    assert summary["dtypes"] == {"torch.bfloat16": len(tensors)}


def _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch):
    tensors = _plain_checkpoint(spec)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard_name))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: shard_name for name in tensors},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "KIMI_K3_CONFIG_SHA256",
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "KIMI_K3_INDEX_SHA256",
        hashlib.sha256(index_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint_validation.K3Config,
        "_from_hf_dict",
        lambda _config: _TinyConfig(),
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "inspect_hf_checkpoint",
        lambda _root: SimpleNamespace(
            weights=SimpleNamespace(
                quantized_weights=0,
                plain_tensors=len(tensors),
                shards=1,
            )
        ),
    )


def test_report_is_not_published_when_any_tensor_is_not_bitwise_equal(
    tmp_path, monkeypatch
):
    class _CorruptingSpec(K3WeightSpec):
        def native_to_hf(self, native_name, tensor):
            restored = super().native_to_hf(native_name, tensor)
            if native_name == "embed_tokens.embedding.weight":
                name, value = restored[0]
                value = value.clone()
                value.flatten()[0] = -0.0
                return [(name, value)]
            return restored

    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    monkeypatch.setattr(checkpoint_validation, "K3WeightSpec", _CorruptingSpec)
    output = tmp_path / "summary.json"

    with pytest.raises(AssertionError, match="bitwise mismatch"):
        validate_checkpoint(tmp_path, output)

    assert not output.exists()


def test_successful_report_uses_real_safetensors_and_has_no_unevidenced_coverage(
    tmp_path, monkeypatch
):
    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    output = tmp_path / "summary.json"

    report = validate_checkpoint(tmp_path, output)

    assert json.loads(output.read_text()) == report
    assert report["checkpoint"]["bitwise_equal"] is True
    assert report["coverage"]["samples"]["layer_count"] == 3
    assert all(
        cell["status"] == "not-covered"
        for row in report["coverage"]["matrix"]["rows"]
        for cell in row["cells"].values()
    )


def test_public_report_writer_cannot_bypass_checkpoint_audit():
    assert not hasattr(checkpoint_validation, "write_validation_report")
