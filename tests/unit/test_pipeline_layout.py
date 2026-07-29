from __future__ import annotations

import pytest

from mlite_k3.config import K3Config
from mlite_k3.lite.pipeline_layout import (
    _attn_res_decoder_layer_groups,
    validate_attn_res_pipeline_split,
)


def test_k3_attn_res_groups_follow_the_configured_block_boundaries():
    groups = _attn_res_decoder_layer_groups(K3Config())

    assert [(group[0], group[-1], len(group)) for group in groups] == [
        (0, 11, 12),
        (12, 23, 12),
        (24, 35, 12),
        (36, 47, 12),
        (48, 59, 12),
        (60, 71, 12),
        (72, 83, 12),
        (84, 92, 9),
    ]


def test_k3_attn_res_split_guard_accepts_whole_blocks_only():
    config = K3Config()
    groups = _attn_res_decoder_layer_groups(config)

    for group in groups:
        validate_attn_res_pipeline_split(
            group,
            num_hidden_layers=config.num_hidden_layers,
            block_size=config.attn_res_block_size,
        )

    with pytest.raises(ValueError, match="cannot cross pipeline stages"):
        validate_attn_res_pipeline_split(
            list(range(6)),
            num_hidden_layers=config.num_hidden_layers,
            block_size=config.attn_res_block_size,
        )


def test_k3_attn_res_split_guard_rejects_a_decoder_empty_stage():
    with pytest.raises(ValueError, match="at least one complete AttnRes block"):
        validate_attn_res_pipeline_split(
            [],
            num_hidden_layers=93,
            block_size=12,
        )


def test_k3_grouped_layout_reuses_mlite_primitive_for_pp8():
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import (
        ParallelState,
        build_pipeline_chunk_layout,
    )

    config = K3Config()
    groups = _attn_res_decoder_layer_groups(config)
    stages: list[list[int]] = []
    for rank in range(8):
        ps = ParallelState(
            pp_size=8,
            pp_rank=rank,
            pp_is_first=rank == 0,
            pp_is_last=rank == 7,
        )
        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers,
            ps,
            decoder_layer_groups=groups,
        )
        validate_attn_res_pipeline_split(
            layout.layer_indices,
            num_hidden_layers=config.num_hidden_layers,
            block_size=config.attn_res_block_size,
        )
        stages.append(layout.layer_indices)

    assert [len(stage) for stage in stages] == [12, 12, 12, 12, 12, 12, 12, 9]
    assert [layer for stage in stages for layer in stage] == list(range(93))


def test_k3_grouped_layout_fails_loudly_above_the_eight_block_limit():
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import (
        ParallelState,
        build_pipeline_chunk_layout,
    )

    config = K3Config()
    groups = _attn_res_decoder_layer_groups(config)
    with pytest.raises(ValueError, match="at least one complete AttnRes block"):
        for rank in range(9):
            ps = ParallelState(
                pp_size=9,
                pp_rank=rank,
                pp_is_first=rank == 0,
                pp_is_last=rank == 8,
            )
            layout = build_pipeline_chunk_layout(
                config.num_hidden_layers,
                ps,
                decoder_layer_groups=groups,
            )
            validate_attn_res_pipeline_split(
                layout.layer_indices,
                num_hidden_layers=config.num_hidden_layers,
                block_size=config.attn_res_block_size,
            )
