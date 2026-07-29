"""Structural contract for K3 packed-sequence execution."""

from __future__ import annotations

from typing import Any

import torch


def thd_requires_context_parallel_slice(
    packed_seq_params: Any,
    *,
    cp_size: int,
) -> bool:
    """Return whether a full THD row still needs its model-boundary CP slice."""
    local_cp_size_value = getattr(packed_seq_params, "local_cp_size", None)
    if local_cp_size_value is None:
        raise ValueError("K3 THD local_cp_size must be explicitly set")
    local_cp_size = int(local_cp_size_value)
    if local_cp_size < 1:
        raise ValueError("K3 THD local_cp_size must be positive")
    if local_cp_size not in (1, cp_size):
        raise ValueError(
            "K3 THD local_cp_size="
            f"{local_cp_size} does not match model cp_size={cp_size}"
        )
    return cp_size > 1 and local_cp_size == 1


def validate_thd_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    packed_seq_params: Any,
) -> None:
    """Fail loud on malformed K3 THD inputs at the model boundary."""
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
    if not torch.equal(cu_q, cu_kv):
        raise ValueError("K3 THD q and kv cu_seqlens must match")
    if cu_q.numel() < 2:
        raise ValueError("K3 THD cu_seqlens must contain at least one sequence")
    if int(cu_q[0].item()) != 0:
        raise ValueError("K3 THD cu_seqlens must start at 0")
    if not bool(torch.all(cu_q[1:] >= cu_q[:-1]).item()):
        raise ValueError("K3 THD cu_seqlens must be nondecreasing")
    total_tokens = getattr(packed_seq_params, "total_tokens", None)
    if total_tokens is None:
        raise ValueError("K3 THD total_tokens must be explicitly set")
    total_tokens = int(total_tokens)
    if int(cu_q[-1].item()) != total_tokens:
        raise ValueError("K3 THD cu_seqlens must end at total_tokens")
    local_cp_size_value = getattr(packed_seq_params, "local_cp_size", None)
    if local_cp_size_value is None:
        raise ValueError("K3 THD local_cp_size must be explicitly set")
    local_cp_size = int(local_cp_size_value)
    if local_cp_size < 1:
        raise ValueError("K3 THD local_cp_size must be positive")
    if input_ids.shape[1] * local_cp_size != total_tokens:
        raise ValueError(
            "K3 THD local token width times local_cp_size must match total_tokens"
        )


__all__ = ["thd_requires_context_parallel_slice", "validate_thd_inputs"]
