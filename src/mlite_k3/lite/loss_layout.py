"""Shared sequence layout for labels and loss masks."""

from __future__ import annotations

from typing import Any

import torch

from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp


def prepare_labels_and_loss_mask(
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None,
    ps: Any,
    *,
    slice_for_cp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply one CP sequence-layout contract to labels and their mask."""
    labels_sb = labels.transpose(0, 1).contiguous()
    mask_sb = None if loss_mask is None else loss_mask.transpose(0, 1).contiguous()
    if mask_sb is not None and mask_sb.shape != labels_sb.shape:
        raise ValueError(
            "labels and loss_mask must have the same shape before CP permutation"
        )

    if slice_for_cp and ps.cp_size > 1:
        labels_sb = zigzag_slice_for_cp(
            labels_sb,
            ps.cp_rank,
            ps.cp_size,
            seq_dim=0,
        )
        if mask_sb is not None:
            mask_sb = zigzag_slice_for_cp(
                mask_sb,
                ps.cp_rank,
                ps.cp_size,
                seq_dim=0,
            )

    if mask_sb is not None and mask_sb.shape != labels_sb.shape:
        raise RuntimeError(
            "labels and loss_mask must have the same shape after CP permutation"
        )
    return labels_sb, mask_sb


__all__ = ["prepare_labels_and_loss_mask"]
