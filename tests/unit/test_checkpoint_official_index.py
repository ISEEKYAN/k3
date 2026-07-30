from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    K3WeightSpec,
    audit_k3_weight_spec_sources,
    inspect_hf_checkpoint,
    plan_k3_rank_weights,
)


_OFFICIAL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_OFFICIAL_INDEX_SHA256 = (
    "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
)


def test_official_index_plans_every_tp2_ep2_pp2_rank_without_opening_shards():
    root_text = os.environ.get("K3_OFFICIAL_INDEX_DIR")
    if not root_text:
        pytest.skip(
            "set K3_OFFICIAL_INDEX_DIR to config.json and "
            "model.safetensors.index.json from moonshotai/Kimi-K3@"
            f"{_OFFICIAL_REVISION}"
        )
    root = Path(root_text)
    index_path = root / "model.safetensors.index.json"
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == (
        _OFFICIAL_INDEX_SHA256
    )

    config = K3Config.from_hf(root)
    manifest = inspect_hf_checkpoint(root)
    index = json.loads(index_path.read_text())["weight_map"]
    spec = K3WeightSpec(config, manifest=manifest)
    assert audit_k3_weight_spec_sources(spec, index) == 249756
    expert_bias_sources = {
        sources[0]
        for native_name, sources in spec.weight_map().items()
        if native_name.endswith(".moe.router.expert_bias")
    }
    assert len(expert_bias_sources) == 92
    assert expert_bias_sources <= index.keys()

    pp_layers = (list(range(47)), list(range(47, 93)))
    plans = {}
    for pp_rank, layer_indices in enumerate(pp_layers):
        for tp_rank in range(2):
            for ep_rank in range(2):
                plans[pp_rank, tp_rank, ep_rank] = plan_k3_rank_weights(
                    spec,
                    index,
                    layer_indices=layer_indices,
                    has_embed=pp_rank == 0,
                    has_head=pp_rank == 1,
                    tp_size=2,
                    tp_rank=tp_rank,
                    ep_size=2,
                    ep_rank=ep_rank,
                    etp_size=1,
                )

    assert len(plans) == 8
    for pp_rank in range(2):
        for ep_rank in range(2):
            assert plans[pp_rank, 0, ep_rank] == plans[pp_rank, 1, ep_rank]

    covered_sources = {
        source
        for pp_rank in range(2)
        for ep_rank in range(2)
        for item in plans[pp_rank, 0, ep_rank]
        for source in item.hf_names
    }
    expected_sources = {
        source for sources in spec.weight_map().values() for source in sources
    }
    assert covered_sources == expected_sources
    assert expected_sources == {
        source for source in index if source.startswith("language_model.")
    }

    first = plans[0, 0, 0]
    last = plans[1, 1, 1]
    first_by_name = {item.native_name: item for item in first}
    last_by_name = {item.native_name: item for item in last}
    assert first_by_name["embed_tokens.embedding.weight"].shape == (81920, 7168)
    assert not any(item.native_name.startswith("lm_head.") for item in first)
    assert last_by_name["lm_head.col.linear.weight"].shape == (81920, 7168)
    assert not any(item.native_name.startswith("embed_tokens.") for item in last)
    assert first_by_name["layers.0.self_attention.q_proj.linear.weight"].shape == (
        6144,
        7168,
    )
    assert first_by_name["layers.0.self_attention.o_proj.linear.weight"].shape == (
        7168,
        6144,
    )
    assert first_by_name["layers.0.self_attention.k_conv1d.weight"].shape == (
        6144,
        1,
        4,
    )
    assert first_by_name["layers.1.moe.experts.fc1.weight0"].shape == (6144, 3584)
    assert first_by_name["layers.1.moe.experts.fc2.weight0"].shape == (3584, 3072)
    assert first_by_name["layers.1.moe.router.expert_bias"].dtype == torch.float32
