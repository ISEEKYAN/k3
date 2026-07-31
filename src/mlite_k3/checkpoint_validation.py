"""Fail-closed validation for the pinned public Kimi K3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safetensors import safe_open

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    inspect_hf_checkpoint,
)
from mlite_k3.validation_schema import (
    CAPABILITIES,
    STRUCTURES,
    is_verified_evidence_source,
)


KIMI_K3_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
KIMI_K3_CONFIG_SHA256 = (
    "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
)
KIMI_K3_INDEX_SHA256 = (
    "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
)

_CAPABILITY_DOC_CONTRACT = re.compile(
    r"<!-- K3_CAPABILITY_SCHEMA_BEGIN -->\s*"
    r"```json\s*(\{.*?\})\s*```\s*"
    r"<!-- K3_CAPABILITY_SCHEMA_END -->",
    re.DOTALL,
)
_CAPABILITY_DOC_PATH = Path(__file__).resolve().parents[2] / "docs/validation.md"


def _normalize_capability_evidence(
    evidence: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    normalized = {
        str(key): tuple(str(source) for source in sources)
        for key, sources in (evidence or {}).items()
    }
    valid_keys = {
        f"{structure}.{capability}"
        for structure in STRUCTURES
        for capability in CAPABILITIES
    }
    unknown = sorted(set(normalized) - valid_keys)
    missing_sources = sorted(key for key, sources in normalized.items() if not sources)
    invalid_sources = sorted(
        f"{key}:{source}"
        for key, sources in normalized.items()
        for source in sources
        if not is_verified_evidence_source(source)
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
    for structure in STRUCTURES:
        cells = {}
        for capability in CAPABILITIES:
            sources = normalized.get(f"{structure}.{capability}", ())
            cells[capability] = {
                "status": "covered" if sources else "not-covered",
                "evidence": list(sources),
            }
        rows.append({"structure": structure, "cells": cells})

    return {"columns": list(CAPABILITIES), "rows": rows}


def _assert_capability_doc_contract(contents: str) -> None:
    match = _CAPABILITY_DOC_CONTRACT.search(contents)
    if match is None:
        raise RuntimeError("docs/validation.md is missing the capability schema")
    documented = json.loads(match.group(1))
    expected = {
        "capabilities": list(CAPABILITIES),
        "structures": list(STRUCTURES),
    }
    if documented != expected:
        raise RuntimeError(
            f"capability schema drift: runtime={expected}, docs={documented}"
        )
    header = (
        "| Structure | Load | Save | BF16 export | MXFP4 export | "
        "Canonical QAT | Shard rules |"
    )
    lines = contents.splitlines()
    try:
        start = lines.index(header)
    except ValueError as error:
        raise RuntimeError(
            "docs/validation.md is missing the capability table"
        ) from error
    table_rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
    labels = {
        "dense": "dense",
        "routed MoE": "moe",
        "MLA": "mla",
        "KDA": "kda",
        "shared expert": "shared_expert",
        "router + expert bias": "router_expert_bias",
    }
    documented_rows = {
        labels[row[0]]: row[1:] for row in table_rows if row and row[0] in labels
    }
    runtime_rows = {
        row["structure"]: [
            row["cells"][capability]["status"] for capability in CAPABILITIES
        ]
        for row in build_capability_matrix()["rows"]
    }
    if documented_rows != runtime_rows:
        raise RuntimeError(
            f"capability table drift: runtime={runtime_rows}, docs={documented_rows}"
        )


def _assert_repository_capability_doc_contract() -> None:
    try:
        contents = _CAPABILITY_DOC_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"cannot read capability contract {_CAPABILITY_DOC_PATH}"
        ) from error
    _assert_capability_doc_contract(contents)


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


def _read_tensor_metadata(
    reader: Any,
    source_names: Sequence[str],
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read safetensors header metadata without materializing payload bytes."""
    accessor = getattr(reader, "tensor_metadata", None)
    if callable(accessor):
        return {
            name: (tuple(shape), str(dtype))
            for name in source_names
            for shape, dtype in (accessor(name),)
        }

    root = Path(reader.path)
    index = dict(reader.index)
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for name in source_names:
        shard = index.get(name, "model.safetensors")
        by_shard[root / shard].append(name)

    metadata: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard, names in by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(names) - available)
            if missing:
                raise KeyError(
                    f"safetensors shard {shard.name!r} is missing {missing[:8]!r}"
                )
            for name in names:
                tensor_slice = handle.get_slice(name)
                metadata[name] = (
                    tuple(int(dim) for dim in tensor_slice.get_shape()),
                    str(tensor_slice.get_dtype()),
                )
    return metadata


def _validate_header_layout(
    spec: K3WeightSpec,
    native_name: str,
    hf_names: Sequence[str],
    metadata: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    if getattr(spec, "_uses_release_mxfp4", False) and spec.is_expert(native_name):
        if len(hf_names) % 2:
            raise ValueError(f"{native_name!r} requires MXFP4 packed/scale pairs")
        for packed_name, scale_name in zip(
            hf_names[0::2],
            hf_names[1::2],
            strict=True,
        ):
            packed_shape, packed_dtype = metadata[packed_name]
            scale_shape, scale_dtype = metadata[scale_name]
            if packed_dtype not in {"I8", "U8"}:
                raise TypeError(
                    f"MXFP4 packed tensor {packed_name!r} must be I8/U8, "
                    f"got {packed_dtype}"
                )
            if scale_dtype != "U8":
                raise TypeError(
                    f"MXFP4 scale tensor {scale_name!r} must be U8, got {scale_dtype}"
                )
            if not packed_shape or not scale_shape:
                raise ValueError(
                    f"MXFP4 pair {packed_name!r}/{scale_name!r} has an empty shape"
                )
        return

    if re.search(r"\.[qkv]_conv1d\.weight$", native_name):
        shape, _dtype = metadata[hf_names[0]]
        if len(shape) != 3 or shape[1] != 1:
            raise ValueError(
                f"{native_name!r} must have shape [channels, 1, kernel], got {shape}"
            )

    if (".gate_up." in native_name or ".fc1." in native_name) and len(hf_names) == 2:
        left_shape, _ = metadata[hf_names[0]]
        right_shape, _ = metadata[hf_names[1]]
        if len(left_shape) != len(right_shape) or left_shape[1:] != right_shape[1:]:
            raise ValueError(
                f"{native_name!r} fused sources have incompatible shapes: "
                f"{left_shape} and {right_shape}"
            )


def validate_reader_metadata(reader: Any, spec: K3WeightSpec) -> dict[str, Any]:
    """Audit every mapped source using only safetensors index and headers."""
    mapping = spec.weight_map()
    source_keys = sorted(
        {source_name for hf_names in mapping.values() for source_name in hf_names}
    )
    metadata = _read_tensor_metadata(reader, source_keys)
    for native_name, hf_names in sorted(mapping.items()):
        _validate_header_layout(spec, native_name, hf_names, metadata)
    dtypes = Counter(dtype for _shape, dtype in metadata.values())
    ordered_keys = sorted(source_keys)
    return {
        "metadata_only": True,
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


def _current_git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot verify evidence bundle commit outside a git checkout"
        ) from error


def validate_checkpoint(
    path: str | Path,
    output: str | Path,
    *,
    evidence_bundle: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the pinned complete release without retaining multiple shards."""
    _assert_repository_capability_doc_contract()
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
    spec = K3WeightSpec(config, manifest=manifest)
    mapped_sources = audit_k3_weight_spec_sources(spec, reader.index)
    verified_evidence = None
    if evidence_bundle is not None:
        from mlite_k3.validation_harness import load_evidence_bundle

        verified_evidence = load_evidence_bundle(evidence_bundle)
        if verified_evidence.missing_blocking_tiers:
            raise RuntimeError(
                "missing blocking validation tiers: "
                f"{list(verified_evidence.missing_blocking_tiers)}"
            )
        current_commit = _current_git_commit()
        if verified_evidence.git_commit != current_commit:
            raise RuntimeError(
                "evidence bundle commit does not match validator commit: "
                f"evidence={verified_evidence.git_commit}, current={current_commit}"
            )
    checkpoint = validate_reader_metadata(reader, spec)
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
            "matrix": build_capability_matrix(
                None if verified_evidence is None else verified_evidence.capabilities
            ),
            "axes": ({} if verified_evidence is None else verified_evidence.axes),
            "runs": ([] if verified_evidence is None else verified_evidence.runs),
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
        "--evidence-bundle",
        type=Path,
        help="fingerprinted bundle generated by mlite_k3.validation_harness",
    )
    args = parser.parse_args(argv)
    validate_checkpoint(
        args.checkpoint,
        args.output,
        evidence_bundle=args.evidence_bundle,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
