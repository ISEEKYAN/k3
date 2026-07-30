from __future__ import annotations

import torch

from mlite_k3.lite.pipeline_state import (
    _pack_pipeline_state,
    _unpack_pipeline_state,
)
from mlite_k3.model import _apply_attention_residual


class _Norm(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-5


def _short_attn_res_forward(boundaries: tuple[int, ...]) -> torch.Tensor:
    hidden_states = torch.arange(16, dtype=torch.float32).view(2, 1, 8) / 16
    block_residual = hidden_states.new_zeros(2, 0, 8)
    projection = torch.nn.Linear(8, 1, bias=False)
    projection.weight.data.fill_(0.125)
    norm = _Norm(8)

    for layer_index in range(93):
        prefix_sum = hidden_states
        if block_residual.size(1):
            hidden_states = _apply_attention_residual(
                prefix_sum.reshape(-1, 8),
                block_residual,
                projection,
                norm,
            ).view_as(hidden_states)
        if layer_index % 12 == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.reshape(-1, 8).unsqueeze(1)),
                dim=1,
            )
            prefix_sum = None

        attention_output = hidden_states * 0.01 + (layer_index + 1) * 1e-4
        prefix_sum = (
            attention_output if prefix_sum is None else prefix_sum + attention_output
        )
        mlp_input = _apply_attention_residual(
            prefix_sum.reshape(-1, 8),
            block_residual,
            projection,
            norm,
        ).view_as(hidden_states)
        hidden_states = prefix_sum + mlp_input * 0.01

        if layer_index + 1 in boundaries:
            packed = _pack_pipeline_state(hidden_states, block_residual)
            hidden_states, block_residual = _unpack_pipeline_state(
                packed,
                hidden_size=8,
            )

    return hidden_states


def test_split_and_aligned_pp3_layouts_are_numerically_equivalent():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            split_output = _short_attn_res_forward((32, 64))
            aligned_output = _short_attn_res_forward((36, 72))
    finally:
        torch.set_num_threads(previous_threads)

    assert torch.equal(split_output, aligned_output)
