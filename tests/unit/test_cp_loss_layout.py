from __future__ import annotations

import pytest
import torch

from mlite_k3.lite.loss_layout import prepare_labels_and_loss_mask


class _ParallelState:
    cp_rank = 1
    cp_size = 2


def test_labels_and_loss_mask_use_the_same_cp_zigzag_permutation():
    labels = torch.arange(8).view(1, 8)
    loss_mask = (100 + torch.arange(8)).view(1, 8)

    labels_sb, mask_sb = prepare_labels_and_loss_mask(
        labels,
        loss_mask,
        _ParallelState(),
    )

    assert labels_sb[:, 0].tolist() == [2, 3, 4, 5]
    assert mask_sb[:, 0].tolist() == [102, 103, 104, 105]


def test_cp_local_thd_labels_and_mask_are_not_sliced_twice():
    labels = torch.arange(4).view(1, 4)
    loss_mask = (100 + torch.arange(4)).view(1, 4)

    labels_sb, mask_sb = prepare_labels_and_loss_mask(
        labels,
        loss_mask,
        _ParallelState(),
        slice_for_cp=False,
    )

    assert labels_sb[:, 0].tolist() == [0, 1, 2, 3]
    assert mask_sb[:, 0].tolist() == [100, 101, 102, 103]


def test_labels_and_loss_mask_reject_mismatched_pre_permutation_shapes():
    labels = torch.arange(8).view(1, 8)
    loss_mask = torch.ones(1, 7)

    with pytest.raises(ValueError, match="same shape before CP permutation"):
        prepare_labels_and_loss_mask(labels, loss_mask, _ParallelState())
