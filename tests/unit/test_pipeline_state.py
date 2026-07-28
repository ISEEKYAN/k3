from __future__ import annotations

import pytest
import torch

from mlite_k3.lite.pipeline_state import (
    _pack_pipeline_state,
    _pipeline_boundary_bytes,
    _unpack_pipeline_state,
)


def test_pipeline_state_roundtrip_preserves_hidden_and_attention_residual():
    hidden = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    block = torch.randn(10, 3, 8, dtype=torch.bfloat16)

    packed = _pack_pipeline_state(hidden, block)
    restored_hidden, restored_block = _unpack_pipeline_state(packed, hidden_size=8)

    assert packed.shape == (5, 2, 32)
    assert torch.equal(restored_hidden, hidden)
    assert torch.equal(restored_block, block)


def test_pipeline_state_roundtrip_supports_empty_attention_residual():
    hidden = torch.randn(4, 1, 16)
    block = hidden.new_zeros(4, 0, 16)

    packed = _pack_pipeline_state(hidden, block)
    restored_hidden, restored_block = _unpack_pipeline_state(packed, hidden_size=16)

    assert packed.shape == hidden.shape
    assert torch.equal(restored_hidden, hidden)
    assert restored_block.shape == block.shape


def test_pipeline_state_rejects_malformed_hidden_multiple():
    with pytest.raises(ValueError, match="multiple"):
        _unpack_pipeline_state(torch.empty(4, 1, 17), hidden_size=8)


def test_pipeline_boundary_bytes_separates_standard_and_k3_specific_payload():
    sizes = _pipeline_boundary_bytes(
        sequence_length=64,
        micro_batch_size=1,
        hidden_size=256,
        residual_snapshots=4,
        dtype=torch.bfloat16,
    )

    assert sizes == {
        "hidden_states": 32_768,
        "block_residual": 131_072,
        "total": 163_840,
    }
