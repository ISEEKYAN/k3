from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mlite_k3.lite.thd_contract import (
    thd_requires_context_parallel_slice,
    validate_thd_inputs,
)


def _packed(total_tokens: int = 8):
    return SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, total_tokens], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, total_tokens], dtype=torch.int32),
        total_tokens=total_tokens,
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


def test_protocol_local_thd_must_not_be_split_again():
    packed = _packed()
    packed.local_cp_size = 2

    assert not thd_requires_context_parallel_slice(packed, cp_size=2)


def test_thd_rejects_local_context_parallel_metadata_for_another_topology():
    packed = _packed()
    packed.local_cp_size = 4

    with pytest.raises(ValueError, match="local_cp_size=4.*cp_size=2"):
        thd_requires_context_parallel_slice(packed, cp_size=2)
