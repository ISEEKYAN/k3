from __future__ import annotations

import pytest

from mlite_k3.config import K3Config
from mlite_k3.lite.pipeline_layout import (
    _attn_res_decoder_layer_groups,
    build_k3_pipeline_layout,
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
    validate_attn_res_pipeline_split(
        list(range(32)),
        num_hidden_layers=config.num_hidden_layers,
        block_size=config.attn_res_block_size,
        allow_split_attn_res_block=True,
    )


def test_k3_attn_res_split_guard_rejects_a_decoder_empty_stage():
    with pytest.raises(ValueError, match="at least one decoder layer"):
        validate_attn_res_pipeline_split(
            [],
            num_hidden_layers=93,
            block_size=12,
        )


def test_k3_grouped_layout_reuses_mlite_primitive_for_pp8():
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import ParallelState

    config = K3Config()
    stages: list[list[int]] = []
    for rank in range(8):
        ps = ParallelState(
            pp_size=8,
            pp_rank=rank,
            pp_is_first=rank == 0,
            pp_is_last=rank == 7,
        )
        layout = build_k3_pipeline_layout(config, ps)
        stages.append(layout.layer_indices)

    assert [len(stage) for stage in stages] == [12, 12, 12, 12, 12, 12, 12, 9]
    assert [layer for stage in stages for layer in stage] == list(range(93))


@pytest.mark.parametrize(
    ("pp_size", "expected_sizes"),
    [
        (2, [48, 45]),
        (3, [24, 36, 33]),
        (5, [12, 12, 24, 24, 21]),
        (6, [12, 12, 12, 12, 24, 21]),
        (7, [12, 12, 12, 12, 12, 12, 21]),
    ],
)
def test_k3_grouped_layout_supports_nondivisor_pp_sizes(pp_size, expected_sizes):
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import ParallelState

    config = K3Config()
    stages = []
    for rank in range(pp_size):
        ps = ParallelState(
            pp_size=pp_size,
            pp_rank=rank,
            pp_is_first=rank == 0,
            pp_is_last=rank == pp_size - 1,
        )
        layout = build_k3_pipeline_layout(config, ps)
        stages.append(layout.layer_indices)

    assert [len(stage) for stage in stages] == expected_sizes
    assert all(stages)
    assert [layer for stage in stages for layer in stage] == list(range(93))


@pytest.mark.parametrize(
    ("sizes", "expected_ranges"),
    [
        ([32, 32, 29], [(0, 31), (32, 63), (64, 92)]),
        ([36, 36, 21], [(0, 35), (36, 71), (72, 92)]),
    ],
)
def test_k3_explicit_pipeline_layout_can_relax_attn_res_alignment(
    sizes,
    expected_ranges,
):
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import ParallelState

    rows = [
        ["embedding", *(["decoder"] * sizes[0])],
        ["decoder"] * sizes[1],
        [*(["decoder"] * sizes[2]), "loss"],
    ]
    stages = []
    for rank in range(3):
        ps = ParallelState(
            pp_size=3,
            pp_rank=rank,
            pp_is_first=rank == 0,
            pp_is_last=rank == 2,
            pp_layout=rows,
        )
        with pytest.warns(
            UserWarning, match="disables default AttnRes-block alignment"
        ):
            layout = build_k3_pipeline_layout(K3Config(), ps)
        stages.append(layout.layer_indices)

    assert [(stage[0], stage[-1]) for stage in stages] == expected_ranges
    assert [layer for stage in stages for layer in stage] == list(range(93))


def test_k3_pipeline_layout_rejects_more_stages_than_layers():
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import ParallelState

    with pytest.raises(ValueError, match="cannot exceed num_hidden_layers"):
        build_k3_pipeline_layout(
            K3Config(num_hidden_layers=2, full_attention_layers=(2,), kda_layers=(1,)),
            ParallelState(pp_size=3),
        )


def test_k3_explicit_pipeline_layout_rejects_missing_layers_and_empty_stage():
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    from megatron.lite.primitive.parallel import ParallelState

    missing_layer = [
        ["embedding", *(["decoder"] * 32)],
        ["decoder"] * 32,
        [*(["decoder"] * 28), "loss"],
    ]
    with pytest.warns(UserWarning):
        with pytest.raises(AssertionError, match="decoder layers 92"):
            build_k3_pipeline_layout(
                K3Config(),
                ParallelState(pp_size=3, pp_rank=1, pp_layout=missing_layer),
            )

    empty_stage = [
        ["embedding", *(["decoder"] * 48)],
        [],
        [*(["decoder"] * 45), "loss"],
    ]
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="at least one decoder layer"):
            build_k3_pipeline_layout(
                K3Config(),
                ParallelState(pp_size=3, pp_rank=1, pp_layout=empty_stage),
            )
