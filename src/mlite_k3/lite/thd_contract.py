"""Structural contract for K3 packed-sequence execution."""

from __future__ import annotations

from typing import Any

import torch


def validate_thd_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    packed_seq_params: Any,
) -> None:
    """Fail loud on malformed K3 THD inputs without a device-to-host scalar sync."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("K3 THD input_ids must have shape [1, total_tokens]")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("K3 THD labels must have the same shape as input_ids")
    if loss_mask is not None and loss_mask.shape != input_ids.shape:
        raise ValueError("K3 THD loss_mask must have the same shape as input_ids")
    if getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise ValueError("K3 packed_seq_params.qkv_format must be 'thd'")
    cu_q = getattr(packed_seq_params, "cu_seqlens_q", None)
    cu_kv = getattr(packed_seq_params, "cu_seqlens_kv", None)
    if not isinstance(cu_q, torch.Tensor) or not isinstance(cu_kv, torch.Tensor):
        raise ValueError("K3 THD requires cu_seqlens_q and cu_seqlens_kv")
    if cu_q.ndim != 1 or cu_kv.ndim != 1 or cu_q.numel() != cu_kv.numel():
        raise ValueError("K3 THD cu_seqlens must be matching rank-1 tensors")
    total_tokens = getattr(packed_seq_params, "total_tokens", None)
    if total_tokens is None or int(total_tokens) != input_ids.shape[1]:
        raise ValueError("K3 THD total_tokens must match input_ids.shape[1]")


__all__ = ["validate_thd_inputs"]
