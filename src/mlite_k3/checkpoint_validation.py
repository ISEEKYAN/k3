"""Fail-closed validation for the pinned public Kimi K3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    get_hf_weight,
    inspect_hf_checkpoint,
)


KIMI_K3_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
KIMI_K3_CONFIG_SHA256 = (
    "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
)
KIMI_K3_INDEX_SHA256 = (
    "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
)

_CAPABILITIES = (
    "load",
    "save",
    "export_bf16",
    "export_mxfp4",
    "qat_canonical",
    "shard_rules",
)
_STRUCTURES = (
    "dense",
    "moe",
    "mla",
    "kda",
    "shared_expert",
    "router_expert_bias",
)


def _normalize_capability_evidence(
    evidence: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    normalized = {
        str(key): tuple(str(source) for source in sources)
        for key, sources in (evidence or {}).items()
    }
    valid_keys = {
        f"{structure}.{capability}"
        for structure in _STRUCTURES
        for capability in _CAPABILITIES
    }
    unknown = sorted(set(normalized) - valid_keys)
    missing_sources = sorted(key for key, sources in normalized.items() if not sources)
    invalid_sources = sorted(
        f"{key}:{source}"
        for key, sources in normalized.items()
        for source in sources
        if not source.startswith(("test:", "job:"))
    )
    if unknown or missing_sources or invalid_sources:
        raise RuntimeError(
            "invalid K3 capability evidence: "
            f"unknown_cells={unknown}, missing_sources={missing_sources}, "
            f"invalid_sources={invalid_sources}"
        )
    return normalized


def build_capability_matrix(
    evidence: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Derive coverage cells exclusively from traceable execution evidence."""
    normalized = _normalize_capability_evidence(evidence)
    rows = []
    for structure in _STRUCTURES:
        cells = {}
        for capability in _CAPABILITIES:
            sources = normalized.get(f"{structure}.{capability}", ())
            cells[capability] = {
                "status": "covered" if sources else "not-covered",
                "evidence": list(sources),
            }
        rows.append({"structure": structure, "cells": cells})

    return {"columns": list(_CAPABILITIES), "rows": rows}


def build_structural_samples(config: Any) -> dict[str, Any]:
    """Select deterministic boundary samples from the full layer/expert space."""
    layer_count = int(config.num_hidden_layers)
    expert_count = int(config.num_experts)
    mla_layers = {
        index for index in range(layer_count) if config.attention_type(index) == "mla"
    }
    sampled_layers = {0, layer_count - 1}
    for index in mla_layers:
        sampled_layers.update(
            candidate
            for candidate in (index - 1, index, index + 1)
            if 0 <= candidate < layer_count
        )
    experts = sorted({(expert_count - 1) * quartile // 4 for quartile in range(5)})
    layers = [
        {
            "index": index,
            "attention": config.attention_type(index),
            "ffn": ("dense" if index < int(config.first_k_dense_replace) else "moe"),
        }
        for index in sorted(sampled_layers)
    ]
    return {
        "rule": (
            "all MLA layers plus adjacent KDA boundaries; first/last layer; "
            "expert positions 0, 1/4, 1/2, 3/4, and last"
        ),
        "layer_count": layer_count,
        "expert_count": expert_count,
        "layers": layers,
        "experts": experts,
    }


def _source_key_digest(source_keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in source_keys:
        digest.update(key.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def validate_reader_roundtrip(reader: Any, spec: K3WeightSpec) -> dict[str, Any]:
    """Round-trip every logical source tensor and require exact equality."""
    dtypes: Counter[str] = Counter()
    source_keys: set[str] = set()
    mapping = spec.weight_map()

    for native_name in sorted(mapping):
        hf_names = mapping[native_name]
        expected = [get_hf_weight(reader, name) for name in hf_names]
        native = spec.hf_to_native(native_name, expected)
        restored = spec.native_to_hf(native_name, native)
        if [name for name, _ in restored] != hf_names:
            raise AssertionError(
                f"name mismatch for {native_name!r}: "
                f"{[name for name, _ in restored]!r} != {hf_names!r}"
            )
        for source_name, source, (_, actual) in zip(
            hf_names, expected, restored, strict=True
        ):
            if source.shape != actual.shape:
                raise AssertionError(
                    f"shape mismatch for {source_name!r}: "
                    f"{tuple(actual.shape)} != {tuple(source.shape)}"
                )
            if source.dtype != actual.dtype:
                raise AssertionError(
                    f"dtype mismatch for {source_name!r}: "
                    f"{actual.dtype} != {source.dtype}"
                )
            if not _bitwise_equal(source, actual):
                raise AssertionError(f"bitwise mismatch for {source_name!r}")
            if source_name not in source_keys:
                source_keys.add(source_name)
                dtypes[str(source.dtype)] += 1
        del actual, expected, native, restored, source

    ordered_keys = sorted(source_keys)
    return {
        "bitwise_equal": True,
        "native_tensors": len(mapping),
        "source_tensors": len(ordered_keys),
        "source_key_sha256": _source_key_digest(ordered_keys),
        "dtypes": dict(sorted(dtypes.items())),
    }


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    path: str | Path,
    output: str | Path,
    *,
    capability_evidence: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Validate the pinned complete release without retaining multiple shards."""
    root = Path(path)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    if config_sha256 != KIMI_K3_CONFIG_SHA256:
        raise ValueError(
            f"config.json SHA-256 is {config_sha256}, expected {KIMI_K3_CONFIG_SHA256}"
        )
    if index_sha256 != KIMI_K3_INDEX_SHA256:
        raise ValueError(
            f"model.safetensors.index.json SHA-256 is {index_sha256}, "
            f"expected {KIMI_K3_INDEX_SHA256}"
        )

    with config_path.open(encoding="utf-8") as stream:
        hf_config = json.load(stream)
    config = K3Config._from_hf_dict(hf_config)
    manifest = inspect_hf_checkpoint(root)

    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    reader = SafeTensorReader(str(root))
    spec = K3WeightSpec(config)
    mapped_sources = audit_k3_weight_spec_sources(spec, reader.index)
    checkpoint = validate_reader_roundtrip(reader, spec)
    checkpoint.update(
        {
            "revision": KIMI_K3_REVISION,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "mapped_sources": mapped_sources,
            "physical_index_tensors": len(reader.index),
            "quantized_weights": manifest.weights.quantized_weights,
            "plain_tensors": manifest.weights.plain_tensors,
            "shards": manifest.weights.shards,
        }
    )
    report = {
        "checkpoint": checkpoint,
        "coverage": {
            "samples": build_structural_samples(config),
            "matrix": build_capability_matrix(capability_evidence),
        },
    }
    _write_json_atomically(Path(output), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate every mapped tensor in moonshotai/Kimi-K3."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        help="JSON mapping of structure.capability to test:/job: execution IDs",
    )
    args = parser.parse_args(argv)
    evidence = None
    if args.evidence_manifest is not None:
        with args.evidence_manifest.open(encoding="utf-8") as stream:
            evidence = json.load(stream)
        if not isinstance(evidence, Mapping):
            parser.error("--evidence-manifest must contain a JSON object")
    validate_checkpoint(
        args.checkpoint,
        args.output,
        capability_evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
