"""K3-specific pipeline layout constraints for cumulative AttnRes blocks."""

from __future__ import annotations

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
) -> None:
    """Fail fast when a PP stage owns only part of an AttnRes block."""
    if not layer_indices:
        raise ValueError(
            "Each K3 pipeline stage must own at least one complete AttnRes block."
        )
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
        if local_block and local_block != block:
            raise ValueError(
                "K3 AttnRes block cannot cross pipeline stages: "
                f"stage owns layers {layer_indices}, but block "
                f"{min(block)}..{max(block)} must stay together."
            )


__all__ = [
    "_attn_res_decoder_layer_groups",
    "validate_attn_res_pipeline_split",
]
