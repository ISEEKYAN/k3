from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.parallel import split_packed_to_cp_local
from mlite_k3.lite.thd_contract import (
    thd_requires_context_parallel_slice,
    validate_thd_inputs,
)


def _packed(total_tokens: int = 8, *, local_cp_size: int | None = 1):
    return SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, total_tokens], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, total_tokens], dtype=torch.int32),
        total_tokens=total_tokens,
        local_cp_size=local_cp_size,
    )


def test_thd_contract_accepts_canonical_single_packed_row():
    tokens = torch.arange(8).view(1, 8)

    validate_thd_inputs(tokens, tokens.clone(), torch.ones_like(tokens), _packed())


@pytest.mark.parametrize("shape", [(8,), (2, 4)])
def test_thd_contract_rejects_noncanonical_token_shape(shape):
    tokens = torch.zeros(shape, dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[1, total_tokens\]"):
        validate_thd_inputs(tokens, None, None, _packed())


def test_thd_contract_requires_matching_labels_and_explicit_total():
    tokens = torch.zeros(1, 8, dtype=torch.long)

    with pytest.raises(ValueError, match="labels"):
        validate_thd_inputs(
            tokens, torch.zeros(1, 7, dtype=torch.long), None, _packed()
        )
    with pytest.raises(ValueError, match="total_tokens"):
        validate_thd_inputs(tokens, None, None, _packed(total_tokens=7))


def test_plain_thd_requires_exactly_one_model_context_parallel_slice():
    packed = _packed()

    assert thd_requires_context_parallel_slice(packed, cp_size=2)


def test_thd_rejects_missing_or_zero_context_parallel_metadata():
    with pytest.raises(ValueError, match="local_cp_size must be explicitly set"):
        thd_requires_context_parallel_slice(_packed(local_cp_size=None), cp_size=2)
    with pytest.raises(ValueError, match="local_cp_size must be positive"):
        thd_requires_context_parallel_slice(_packed(local_cp_size=0), cp_size=2)


def test_single_rank_thd_with_explicit_metadata_needs_no_slice():
    assert not thd_requires_context_parallel_slice(_packed(), cp_size=1)


def test_protocol_local_thd_must_not_be_split_again():
    packed = _packed()
    packed.local_cp_size = 2

    assert not thd_requires_context_parallel_slice(packed, cp_size=2)


def test_thd_rejects_local_context_parallel_metadata_for_another_topology():
    packed = _packed()
    packed.local_cp_size = 4

    with pytest.raises(ValueError, match="local_cp_size=4.*cp_size=2"):
        thd_requires_context_parallel_slice(packed, cp_size=2)


def test_thd_contract_accepts_cp_local_width_against_global_total():
    tokens = torch.arange(4).view(1, 4)

    validate_thd_inputs(
        tokens,
        tokens.clone(),
        torch.ones_like(tokens),
        _packed(total_tokens=8, local_cp_size=2),
    )


@pytest.mark.parametrize(
    ("cu_seqlens", "message"),
    [
        ([1, 3, 8], "start at 0"),
        ([0, 5, 4], "nondecreasing"),
        ([0, 3, 7], "end at total_tokens"),
    ],
)
def test_thd_contract_rejects_malformed_cu_seqlens(cu_seqlens, message):
    packed = _packed()
    packed.cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32)
    packed.cu_seqlens_kv = packed.cu_seqlens_q.clone()

    with pytest.raises(ValueError, match=message):
        validate_thd_inputs(torch.arange(8).view(1, 8), None, None, packed)


def test_thd_contract_rejects_different_q_and_kv_segment_boundaries():
    tokens = torch.zeros(1, 10, dtype=torch.long)
    packed = _packed(total_tokens=10)
    packed.cu_seqlens_q = torch.tensor([0, 5, 10], dtype=torch.int32)
    packed.cu_seqlens_kv = torch.tensor([0, 7, 10], dtype=torch.int32)

    with pytest.raises(ValueError, match="identical q and kv segment boundaries"):
        validate_thd_inputs(tokens, None, None, packed)


def test_thd_contract_rejects_sequence_lengths_that_cannot_be_zigzag_split():
    packed = _packed()
    packed.cu_seqlens_q = torch.tensor([0, 5, 8], dtype=torch.int32)
    packed.cu_seqlens_kv = packed.cu_seqlens_q.clone()

    with pytest.raises(ValueError, match=r"divisible by 2 \* cp_size=4"):
        validate_thd_inputs(
            torch.arange(8).view(1, 8),
            None,
            None,
            packed,
            cp_size=2,
        )


def _expected_zigzag_tokens(
    full_tokens: torch.Tensor,
    lengths: list[int],
    *,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    pieces = []
    offset = 0
    for length in lengths:
        row = full_tokens[offset : offset + length]
        offset += length
        chunks = row.view(2 * cp_size, length // (2 * cp_size))
        pieces.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    return torch.cat(pieces)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_packed_thd_cp_shards_preserve_each_token_in_exact_order(cp_size):
    lengths = [8, 16]
    full_tokens = torch.arange(sum(lengths))
    cu_seqlens = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(lengths).cumsum(0))
    )
    shards = []

    for cp_rank in range(cp_size):
        shard = split_packed_to_cp_local(
            full_tokens,
            cu_seqlens_padded=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            dim=0,
        )
        expected = (
            full_tokens
            if cp_size == 1
            else _expected_zigzag_tokens(
                full_tokens,
                lengths,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
        )
        assert torch.equal(shard, expected)
        assert shard.numel() == full_tokens.numel() // cp_size
        shards.append(shard)

    assert torch.equal(
        torch.sort(torch.cat(shards)).values,
        full_tokens,
    )
