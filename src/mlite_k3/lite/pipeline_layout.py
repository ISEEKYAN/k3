"""K3-specific pipeline layout constraints for cumulative AttnRes blocks."""

from __future__ import annotations

import warnings
from typing import Any

from mlite_k3.config import K3Config


def _attn_res_decoder_layer_groups(config: K3Config) -> list[list[int]]:
    """Return contiguous decoder groups that must remain on one PP stage."""
    return [
        list(
            range(
                start,
                min(start + config.attn_res_block_size, config.num_hidden_layers),
            )
        )
        for start in range(0, config.num_hidden_layers, config.attn_res_block_size)
    ]


def validate_attn_res_pipeline_split(
    layer_indices: list[int],
    *,
    num_hidden_layers: int,
    block_size: int,
    allow_split_attn_res_block: bool = False,
) -> None:
    """Validate local layer ownership and optionally require block alignment."""
    if not layer_indices:
        raise ValueError("Each K3 pipeline stage must own at least one decoder layer.")
    if layer_indices != list(
        range(layer_indices[0], layer_indices[0] + len(layer_indices))
    ):
        raise ValueError(
            f"K3 pipeline stage layers must be contiguous, got {layer_indices}."
        )
    if layer_indices[0] < 0 or layer_indices[-1] >= num_hidden_layers:
        raise ValueError(f"K3 pipeline stage layers are out of range: {layer_indices}.")

    local_layers = set(layer_indices)
    for start in range(0, num_hidden_layers, block_size):
        block = set(range(start, min(start + block_size, num_hidden_layers)))
        local_block = local_layers & block
        if not allow_split_attn_res_block and local_block and local_block != block:
            raise ValueError(
                "K3 AttnRes block cannot cross pipeline stages: "
                f"stage owns layers {layer_indices}, but block "
                f"{min(block)}..{max(block)} must stay together."
            )


def build_k3_pipeline_layout(config: K3Config, ps: Any):
    """Build the default aligned layout or honor an explicit user layout."""
    from megatron.lite.primitive.parallel import build_pipeline_chunk_layout

    if ps.pp_size > config.num_hidden_layers:
        raise ValueError(
            "K3 pipeline parallel size cannot exceed num_hidden_layers: "
            f"pp_size={ps.pp_size}, num_hidden_layers={config.num_hidden_layers}."
        )

    explicit_layout = getattr(ps, "pp_layout", None) is not None
    if explicit_layout:
        warnings.warn(
            "Explicit K3 pp_layout disables default AttnRes-block alignment. "
            "AttnRes snapshots stay in the existing BF16 folded P2P payload "
            "with no extra P2P calls; boundary placement can change payload "
            "width and pipeline bubble, so benchmark the selected layout.",
            UserWarning,
            stacklevel=2,
        )
    layout = build_pipeline_chunk_layout(
        config.num_hidden_layers,
        ps,
        decoder_layer_groups=(
            None if explicit_layout else _attn_res_decoder_layer_groups(config)
        ),
    )
    validate_attn_res_pipeline_split(
        layout.layer_indices,
        num_hidden_layers=config.num_hidden_layers,
        block_size=config.attn_res_block_size,
        allow_split_attn_res_block=explicit_layout,
    )
    return layout


__all__ = [
    "_attn_res_decoder_layer_groups",
    "build_k3_pipeline_layout",
    "validate_attn_res_pipeline_split",
]
