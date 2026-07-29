"""K3 pipeline payload helpers.

MLite's pipeline runtime intentionally transports one rank-3 tensor. K3 folds its
second AttnRes activation stream into that tensor's hidden dimension so the
existing dynamic-shape P2P and 1F1B schedule remain unchanged.
"""

from __future__ import annotations

import torch


def _pack_pipeline_state(
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> torch.Tensor:
    """Fold ``[S,B,H]`` and ``[S*B,K,H]`` into one ``[S,B,(K+1)H]`` tensor."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [sequence, batch, hidden]")
    if block_residual.ndim != 3:
        raise ValueError("block_residual must have shape [sequence*batch, K, hidden]")
    sequence, batch, hidden = hidden_states.shape
    if block_residual.shape[0] != sequence * batch or block_residual.shape[2] != hidden:
        raise ValueError(
            "block_residual does not match hidden_states sequence/batch/hidden"
        )
    residual = block_residual.view(sequence, batch, -1)
    return torch.cat((hidden_states, residual), dim=-1)


def _unpack_pipeline_state(
    pipeline_state: torch.Tensor,
    *,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore the two K3 activation streams from MLite's single P2P tensor."""
    if pipeline_state.ndim != 3:
        raise ValueError(
            "pipeline state must have shape [sequence, batch, folded_hidden]"
        )
    if pipeline_state.shape[-1] % hidden_size:
        raise ValueError(
            "pipeline state hidden dimension must be a multiple of hidden_size"
        )
    sequence, batch, folded_hidden = pipeline_state.shape
    snapshots = folded_hidden // hidden_size - 1
    if snapshots < 0:
        raise ValueError("pipeline state is smaller than one hidden state")
    hidden_states = pipeline_state[..., :hidden_size]
    block_residual = pipeline_state[..., hidden_size:].view(
        sequence * batch,
        snapshots,
        hidden_size,
    )
    return hidden_states, block_residual


def _pipeline_boundary_bytes(
    *,
    sequence_length: int,
    micro_batch_size: int,
    hidden_size: int,
    residual_snapshots: int,
    dtype: torch.dtype,
) -> dict[str, int]:
    """Return the standard and K3-specific forward payload sizes at a PP boundary."""
    element_size = torch.empty((), dtype=dtype).element_size()
    hidden_bytes = sequence_length * micro_batch_size * hidden_size * element_size
    residual_bytes = hidden_bytes * residual_snapshots
    return {
        "hidden_states": hidden_bytes,
        "block_residual": residual_bytes,
        "total": hidden_bytes + residual_bytes,
    }


__all__ = [
    "_pack_pipeline_state",
    "_pipeline_boundary_bytes",
    "_unpack_pipeline_state",
]
