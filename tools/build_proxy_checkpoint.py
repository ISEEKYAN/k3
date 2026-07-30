#!/usr/bin/env python3
"""Slice the public Kimi K3 checkpoint into a text proxy without reinitializing it."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


_LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_EXPERT = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.")
_ROUTER_EXPERT_AXIS = re.compile(
    r"\.block_sparse_moe\.gate\.(?:weight|e_score_correction_bias)$"
)
_COPY_METADATA = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_COPY_REMOTE_CODE = (
    "tokenization_kimi.py",
    "tiktoken.model",
    "encoding_k3.py",
    "media_utils.py",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
)


def auto_map_code_files(*documents: dict) -> set[Path]:
    """Return local Python files referenced by nested Hugging Face auto maps."""
    references: set[str] = set()

    def collect_reference(value) -> None:
        if isinstance(value, str):
            references.add(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_reference(item)

    def visit(value) -> None:
        if isinstance(value, dict):
            auto_map = value.get("auto_map")
            if isinstance(auto_map, dict):
                for reference in auto_map.values():
                    collect_reference(reference)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for document in documents:
        visit(document)

    files = set()
    for reference in references:
        if "." not in reference:
            continue
        module = reference.rsplit(".", 1)[0].split("--")[-1]
        files.add(Path(*module.split(".")).with_suffix(".py"))
    return files


def keep_weight(name: str, *, layers: int, experts: int) -> bool:
    layer = _LAYER.search(name)
    if layer is not None and int(layer.group(1)) >= layers:
        return False
    expert = _EXPERT.search(name)
    if expert is not None and int(expert.group(1)) >= experts:
        return False
    return not name.startswith("vision_tower.")


def proxy_config(source: dict, *, layers: int, experts: int) -> dict:
    result = json.loads(json.dumps(source))
    text = result["text_config"]
    text["num_hidden_layers"] = layers
    text["num_experts"] = experts
    linear = text["linear_attn_config"]
    linear["full_attn_layers"] = [
        layer for layer in linear["full_attn_layers"] if layer <= layers
    ]
    linear["kda_layers"] = [layer for layer in linear["kda_layers"] if layer <= layers]
    return result


def slice_proxy_weight(name: str, tensor, *, experts: int):
    if _ROUTER_EXPERT_AXIS.search(name) is None:
        return tensor
    if tensor.ndim < 1 or tensor.shape[0] < experts:
        raise ValueError(
            f"router expert axis for {name!r} is {tuple(tensor.shape)}, "
            f"cannot keep {experts} experts"
        )
    if tensor.shape[0] == experts:
        return tensor
    return tensor.narrow(0, 0, experts).contiguous()


def build_proxy(source: Path, output: Path, *, layers: int, experts: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_config = json.loads((source / "config.json").read_text())
    config = proxy_config(source_config, layers=layers, experts=experts)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in _COPY_METADATA:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    for name in _COPY_REMOTE_CODE:
        source_path = source / name
        if not source_path.is_file():
            raise RuntimeError(f"required K3 remote-code file is missing: {source_path}")
        shutil.copy2(source_path, output / name)
    metadata_documents = [source_config]
    for name in _COPY_METADATA:
        if not name.endswith(".json"):
            continue
        path = source / name
        if path.is_file():
            metadata_documents.append(json.loads(path.read_text()))
    for relative_path in sorted(auto_map_code_files(*metadata_documents)):
        source_path = source / relative_path
        if not source_path.is_file():
            raise RuntimeError(
                f"auto_map references missing local code file: {source_path}"
            )
        output_path = output / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)

    index = json.loads((source / "model.safetensors.index.json").read_text())
    selected = {
        name: shard
        for name, shard in index["weight_map"].items()
        if keep_weight(name, layers=layers, experts=experts)
    }
    shards: dict[str, list[str]] = defaultdict(list)
    for name, shard in selected.items():
        shards[shard].append(name)

    output_map: dict[str, str] = {}
    total_size = 0
    output_shards = sorted(shards)
    total_shards = len(output_shards)
    for output_index, source_shard in enumerate(output_shards, start=1):
        tensors = {}
        with safe_open(source / source_shard, framework="pt", device="cpu") as handle:
            for name in sorted(shards[source_shard]):
                tensor = handle.get_tensor(name)
                tensor = slice_proxy_weight(name, tensor, experts=experts)
                tensors[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
        output_name = f"model-{output_index:05d}-of-{total_shards:05d}.safetensors"
        save_file(tensors, output / output_name)
        output_map.update({name: output_name for name in tensors})

    output_index = {
        "metadata": {
            "total_size": total_size,
            "proxy": {"layers": layers, "experts": experts},
        },
        "weight_map": output_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"K3_PROXY_CHECKPOINT_OK layers={layers} experts={experts} "
        f"tensors={len(output_map)} shards={total_shards} bytes={total_size}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--experts", type=int, default=56)
    args = parser.parse_args()
    if args.layers <= 0 or args.experts <= 0:
        parser.error("--layers and --experts must be positive")
    build_proxy(args.source, args.output, layers=args.layers, experts=args.experts)


if __name__ == "__main__":
    main()
