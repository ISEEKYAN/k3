"""Fail-closed validation for the pinned public Kimi K3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    get_hf_weight,
    inspect_hf_checkpoint,
    load_weights_from_reader,
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


def _cell(status: str, evidence: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence}


def build_capability_matrix() -> dict[str, Any]:
    """Return the explicit structure-by-checkpoint-capability contract."""
    rows = []
    for structure in _STRUCTURES:
        cells = {
            "load": _cell(
                "covered",
                "K3WeightSpec + load_weights_from_reader",
            ),
            "save": _cell(
                "covered",
                "save_hf_weights streams the mapped public names",
            ),
            "export_bf16": _cell(
                "covered",
                "iter_hf_weights target=bf16",
            ),
            "export_mxfp4": _cell(
                "covered",
                (
                    "routed w1/w2/w3 use packed+scale pairs"
                    if structure == "moe"
                    else "non-routed tensors remain plain under target=mxfp4"
                ),
            ),
            "qat_canonical": _cell(
                ("covered" if structure == "moe" else "excluded_by_contract"),
                (
                    "only routed expert linears are parametrized"
                    if structure == "moe"
                    else "K3 QAT ignore contract preserves this structure"
                ),
            ),
            "shard_rules": _cell(
                "covered",
                (
                    "each packed/scale pair is co-located"
                    if structure == "moe"
                    else "plain tensors are indexed exactly once"
                ),
            ),
        }
        rows.append({"structure": structure, "cells": cells})

    rows.append(
        {
            "structure": "mtp",
            "cells": {
                capability: _cell(
                    "out_of_scope",
                    "MTP is explicitly outside this checkpoint-validation task",
                )
                for capability in _CAPABILITIES
            },
        }
    )
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


class _LogicalReader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors
        self.index = set(tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _SingleTensorSpec:
    def __init__(self, source: K3WeightSpec, native_name: str):
        self.source = source
        self.native_name = native_name

    def weight_map(self) -> dict[str, list[str]]:
        return {self.native_name: self.source.weight_map()[self.native_name]}

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        return self.source.hf_to_native(native_name, hf_tensors)

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        return self.source.native_to_hf(native_name, tensor)


class _SingleTensorModel(torch.nn.Module):
    def __init__(self, native_name: str, expected: torch.Tensor):
        super().__init__()
        self.native_name = native_name
        self.value = torch.nn.Parameter(
            torch.empty_like(expected),
            requires_grad=False,
        )

    def named_parameters(self, *args, **kwargs):
        del args, kwargs
        yield self.native_name, self.value

    def named_buffers(self, *args, **kwargs):
        del args, kwargs
        return iter(())

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return {self.native_name: self.value}


def validate_reader_roundtrip(reader: Any, spec: K3WeightSpec) -> dict[str, Any]:
    """Round-trip every logical source tensor and require exact equality."""
    dtypes: Counter[str] = Counter()
    source_keys: set[str] = set()
    mapping = spec.weight_map()

    for native_name in sorted(mapping):
        hf_names = mapping[native_name]
        expected = [get_hf_weight(reader, name) for name in hf_names]
        native = spec.hf_to_native(native_name, expected)
        model = _SingleTensorModel(native_name, native)
        logical_reader = _LogicalReader(dict(zip(hf_names, expected, strict=True)))
        loaded = load_weights_from_reader(
            model,
            logical_reader,
            _SingleTensorSpec(spec, native_name),
        )
        if loaded != 1:
            raise AssertionError(f"loader did not restore {native_name!r}")
        if not _bitwise_equal(native, model.value):
            raise AssertionError(f"native load mismatch for {native_name!r}")
        restored = spec.native_to_hf(native_name, model.value)
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
        del actual, expected, logical_reader, model, native, restored, source

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


def write_validation_report(
    reader: Any,
    spec: K3WeightSpec,
    config: Any,
    output: str | Path,
    *,
    revision: str,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish JSON only after every tensor and coverage invariant succeeds."""
    checkpoint = validate_reader_roundtrip(reader, spec)
    checkpoint["revision"] = revision
    if checkpoint_metadata:
        checkpoint.update(checkpoint_metadata)
    report = {
        "checkpoint": checkpoint,
        "coverage": {
            "samples": build_structural_samples(config),
            "matrix": build_capability_matrix(),
        },
    }
    _write_json_atomically(Path(output), report)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: str | Path, output: str | Path) -> dict[str, Any]:
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
    return write_validation_report(
        reader,
        spec,
        config,
        output,
        revision=KIMI_K3_REVISION,
        checkpoint_metadata={
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "mapped_sources": mapped_sources,
            "physical_index_tensors": len(reader.index),
            "quantized_weights": manifest.weights.quantized_weights,
            "plain_tensors": manifest.weights.plain_tensors,
            "shards": manifest.weights.shards,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate every mapped tensor in moonshotai/Kimi-K3."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_checkpoint(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
