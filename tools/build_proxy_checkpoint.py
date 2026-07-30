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
_COPY_METADATA = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


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
