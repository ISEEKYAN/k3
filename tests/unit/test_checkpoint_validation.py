from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mlite_k3 import checkpoint_validation
from mlite_k3.checkpoint_validation import (
    build_capability_matrix,
    build_structural_samples,
    validate_checkpoint,
    validate_reader_metadata,
)
from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import K3WeightSpec
from mlite_k3.lite.checkpoint import (
    K3CheckpointManifest,
    K3QuantizationMetadata,
    WeightIndexAudit,
)
from mlite_k3.validation_harness import (
    finalize_evidence_bundle,
    write_run_record,
)


class _Reader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def tensor_metadata(self, name: str):
        tensor = self._tensors[name]
        dtype = {
            torch.bfloat16: "BF16",
            torch.float32: "F32",
            torch.int8: "I8",
            torch.uint8: "U8",
        }[tensor.dtype]
        return tuple(tensor.shape), dtype


def test_checkpoint_metadata_validation_never_materializes_tensor_payloads():
    class MetadataOnlyReader:
        index = {"weight": "model.safetensors"}

        @staticmethod
        def tensor_metadata(name):
            assert name == "weight"
            return (2, 3), "BF16"

        @staticmethod
        def get_tensor(name):
            raise AssertionError(f"payload read is forbidden: {name}")

    class Spec:
        @staticmethod
        def weight_map():
            return {"native.weight": ["weight"]}

    summary = checkpoint_validation.validate_reader_metadata(
        MetadataOnlyReader(),
        Spec(),
    )

    assert summary["metadata_only"] is True


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
                tensor = torch.arange(64, dtype=torch.bfloat16).reshape(2, 32)
            elif source_name.endswith(".w2.weight"):
                tensor = torch.arange(1024, dtype=torch.bfloat16).reshape(32, 32)
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
    source = f"job:12345:assertion:expert_bias_is_finite_fp32#sha256:{'a' * 64}"
    matrix = build_capability_matrix({"router_expert_bias.load": (source,)})
    rows = {row["structure"]: row["cells"] for row in matrix["rows"]}

    assert rows["router_expert_bias"]["load"] == {
        "status": "covered",
        "evidence": [source],
    }
    assert rows["router_expert_bias"]["save"] == {
        "status": "not-covered",
        "evidence": [],
    }


def test_router_expert_bias_smoke_requires_a_real_bias_and_reports_its_dtype():
    source = (
        Path(__file__).parents[2] / "tests/gpu/test_checkpoint_load_smoke.py"
    ).read_text(encoding="utf-8")

    assert (
        'raise RuntimeError("K3 checkpoint contains no router expert_bias")' in source
    )
    assert '"expert_bias_dtype": str(expert_bias[0].dtype)' in source
    assert '"expert_bias_dtype": "torch.float32"' not in source


@pytest.mark.parametrize(
    "evidence",
    (
        {"unknown.load": ("test:tests/gpu/test.py::test_case",)},
        {"dense.unknown": ("test:tests/gpu/test.py::test_case",)},
        {"dense.load": ()},
        {"dense.load": ("looks convincing but is not an execution id",)},
        {"dense.load": ("test:i_promise_this_ran",)},
        {"dense.load": ("job:12345:assertion:bad#sha256:not-a-real-digest",)},
    ),
)
def test_capability_matrix_rejects_invalid_or_untraceable_evidence(evidence):
    with pytest.raises(RuntimeError, match="invalid K3 capability evidence"):
        build_capability_matrix(evidence)


def test_reader_metadata_checks_every_mapped_source_shape_and_dtype():
    spec = K3WeightSpec(_TinyConfig())
    tensors = _plain_checkpoint(spec)

    summary = validate_reader_metadata(_Reader(tensors), spec)

    assert summary["metadata_only"] is True
    assert summary["native_tensors"] == len(spec.weight_map())
    assert summary["source_tensors"] == len(tensors)
    assert len(summary["source_key_sha256"]) == 64
    assert summary["dtypes"] == {"BF16": len(tensors)}


def test_reader_metadata_uses_manifest_aware_mxfp4_mapping():
    from mlite_k3.lite import checkpoint

    plain_spec = K3WeightSpec(_TinyConfig())
    physical = dict(
        checkpoint._export_mxfp4_weights(iter(_plain_checkpoint(plain_spec).items()))
    )
    manifest = K3CheckpointManifest(
        quantization=K3QuantizationMetadata(
            format="mxfp4-pack-quantized",
            group_size=32,
            num_bits=4,
            scale_dtype="torch.uint8",
            ignored_modules=frozenset(),
        ),
        weights=WeightIndexAudit(quantized_weights=18, plain_tensors=0, shards=1),
    )
    spec = K3WeightSpec(_TinyConfig(), manifest=manifest)

    summary = validate_reader_metadata(_Reader(physical), spec)

    assert summary["source_tensors"] == len(physical)
    assert "I8" in summary["dtypes"]
    assert "U8" in summary["dtypes"]


def test_manifest_validation_catches_physical_layout_plain_spec_misses():
    plain_spec = K3WeightSpec(_TinyConfig())
    logical = _plain_checkpoint(plain_spec)
    manifest = K3CheckpointManifest(
        quantization=K3QuantizationMetadata(
            format="mxfp4-pack-quantized",
            group_size=32,
            num_bits=4,
            scale_dtype="torch.uint8",
            ignored_modules=frozenset(),
        ),
        weights=WeightIndexAudit(quantized_weights=18, plain_tensors=0, shards=1),
    )
    manifest_spec = K3WeightSpec(_TinyConfig(), manifest=manifest)
    physical = {
        source_name: torch.zeros(1, dtype=torch.uint8)
        for source_names in manifest_spec.weight_map().values()
        for source_name in source_names
        if source_name.endswith(("_packed", "_scale"))
    }
    first_expert = next(
        native_name
        for native_name in sorted(manifest_spec.weight_map())
        if manifest_spec.is_expert(native_name)
    )
    packed_name = manifest_spec.weight_map()[first_expert][0]
    physical[packed_name] = physical[packed_name].to(torch.float32)
    tensors = logical | physical

    plain_summary = validate_reader_metadata(_Reader(tensors), plain_spec)

    assert plain_summary["metadata_only"] is True
    with pytest.raises(TypeError, match="MXFP4 packed tensor.*must be I8/U8"):
        validate_reader_metadata(_Reader(tensors), manifest_spec)


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
            quantization=SimpleNamespace(format="plain"),
            weights=SimpleNamespace(
                quantized_weights=0,
                plain_tensors=len(tensors),
                shards=1,
            ),
        ),
    )


def test_report_is_not_published_when_header_layout_is_invalid(tmp_path, monkeypatch):
    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    tensors = _plain_checkpoint(spec)
    conv_name = next(name for name in tensors if name.endswith("q_conv1d.weight"))
    tensors[conv_name] = torch.zeros((4, 3), dtype=torch.bfloat16)
    save_file(tensors, str(shard))
    output = tmp_path / "summary.json"

    with pytest.raises(ValueError, match=r"shape \[channels, 1, kernel\]"):
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
    assert report["checkpoint"]["metadata_only"] is True
    assert report["coverage"]["samples"]["layer_count"] == 3
    assert all(
        cell["status"] == "not-covered"
        for row in report["coverage"]["matrix"]["rows"]
        for cell in row["cells"].values()
    )


def test_cli_refuses_to_publish_when_capability_docs_drift(tmp_path, monkeypatch):
    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    drifted_doc = tmp_path / "validation.md"
    canonical_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )
    drifted_doc.write_text(
        canonical_doc.replace('"load",', '"imaginary",', 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_CAPABILITY_DOC_PATH",
        drifted_doc,
        raising=False,
    )
    output = tmp_path / "summary.json"

    with pytest.raises(RuntimeError, match="capability schema drift"):
        checkpoint_validation.main([str(tmp_path), "--output", str(output)])

    assert not output.exists()


def test_validator_constructs_the_same_manifest_aware_spec_as_production_load(
    tmp_path, monkeypatch
):
    original_spec = K3WeightSpec
    spec = original_spec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    constructed_manifests = []

    class _RecordingSpec(original_spec):
        def __init__(self, config, *, manifest=None):
            constructed_manifests.append(manifest)
            super().__init__(config, manifest=manifest)

    monkeypatch.setattr(checkpoint_validation, "K3WeightSpec", _RecordingSpec)

    validate_checkpoint(tmp_path, tmp_path / "summary.json")

    assert len(constructed_manifests) == 1
    assert constructed_manifests[0] is not None


def test_report_accepts_only_harness_verified_execution_evidence(tmp_path, monkeypatch):
    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    runs = []
    for tier, job_id, nodes in (
        ("checkpoint_gather_1n", "12345", 1),
        ("checkpoint_gather_2n", "12346", 2),
    ):
        run_dir = tmp_path / tier
        run_dir.mkdir()
        (run_dir / "stdout.log").write_text(
            "K3_CHECKPOINT_LOAD_SMOKE="
            + json.dumps(
                {
                    "world_size": 8,
                    "assertions": [
                        {
                            "cell": "moe.export_bf16",
                            "assertion": "export_matches_baseline",
                        }
                    ],
                    "axes": ["tp", "ep", "pp"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        write_run_record(
            run_dir,
            tier=tier,
            command=[
                "torchrun",
                "--nproc-per-node=8",
                "tests/gpu/test_checkpoint_load_smoke.py",
            ],
            returncode=0,
            duration_seconds=3.0,
            git_commit="verified-commit",
            slurm_job_id=job_id,
            slurm_nodes=nodes,
            slurm_partition="interactive",
        )
        runs.append(run_dir)
    bundle_path = tmp_path / "evidence.json"
    finalize_evidence_bundle(
        runs,
        bundle_path,
        sacct_query=lambda job_id: f"{job_id}|COMPLETED|0:0|4|interactive\n",
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_current_git_commit",
        lambda: "verified-commit",
    )

    report = validate_checkpoint(
        tmp_path,
        tmp_path / "summary.json",
        evidence_bundle=bundle_path,
    )

    rows = {
        row["structure"]: row["cells"] for row in report["coverage"]["matrix"]["rows"]
    }
    assert rows["moe"]["export_bf16"]["status"] == "covered"
    assert all(
        source.startswith(("test:", "job:"))
        for source in rows["moe"]["export_bf16"]["evidence"]
    )
    assert set(report["coverage"]["axes"]) == {"tp", "ep", "pp"}
    assert [run["job_id"] for run in report["coverage"]["runs"]] == [
        "12345",
        "12346",
    ]


def test_report_rejects_evidence_missing_a_blocking_tier(tmp_path, monkeypatch):
    spec = K3WeightSpec(_TinyConfig())
    _write_real_safetensors_checkpoint(tmp_path, spec, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(
        "K3_CHECKPOINT_LOAD_SMOKE="
        + json.dumps(
            {
                "world_size": 8,
                "assertions": [
                    {
                        "cell": "moe.export_bf16",
                        "assertion": "export_matches_baseline",
                    }
                ],
                "axes": ["tp", "ep", "pp"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_record(
        run_dir,
        tier="checkpoint_gather_1n",
        command=[
            "torchrun",
            "--nproc-per-node=8",
            "tests/gpu/test_checkpoint_load_smoke.py",
        ],
        returncode=0,
        duration_seconds=3.0,
        git_commit="verified-commit",
        slurm_job_id="12345",
        slurm_nodes=1,
        slurm_partition="interactive",
    )
    bundle_path = tmp_path / "evidence.json"
    finalize_evidence_bundle(
        [run_dir],
        bundle_path,
        sacct_query=lambda _job_id: "12345|COMPLETED|0:0|4|interactive\n",
    )
    monkeypatch.setattr(
        checkpoint_validation,
        "_current_git_commit",
        lambda: "verified-commit",
    )

    with pytest.raises(RuntimeError, match="missing blocking validation tiers"):
        validate_checkpoint(
            tmp_path,
            tmp_path / "summary.json",
            evidence_bundle=bundle_path,
        )


def test_public_report_writer_cannot_bypass_checkpoint_audit():
    assert not hasattr(checkpoint_validation, "write_validation_report")


def test_capability_schema_cannot_drift_from_documentation():
    validation_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )

    checkpoint_validation._assert_capability_doc_contract(validation_doc)


def test_capability_schema_drift_fails_loudly():
    validation_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )
    drifted = validation_doc.replace('"load",', '"imaginary",', 1)

    with pytest.raises(RuntimeError, match="capability schema drift"):
        checkpoint_validation._assert_capability_doc_contract(drifted)


def test_capability_table_drift_fails_loudly():
    validation_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )
    drifted = validation_doc.replace(
        "| dense | not-covered",
        "| dense | covered",
        1,
    )

    with pytest.raises(RuntimeError, match="capability table drift"):
        checkpoint_validation._assert_capability_doc_contract(drifted)
