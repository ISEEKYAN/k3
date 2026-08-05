"""K3-owned THD and R3 protocol adapters.

These helpers are intentionally local: K3 must run against MLite main without
depending on model-specific protocol utilities that have not landed there.
"""

from __future__ import annotations

from typing import Any

import torch


def _parallel_state(model):
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.thd import parallel_state_from_model

    return parallel_state_from_model(model) or ParallelState()


def _nested_from_packed(tensor: torch.Tensor | None, seq_lens: torch.Tensor):
    if tensor is None:
        return None
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"PackedBatch tensor must be 1-D, got {tuple(tensor.shape)}.")
    pieces = []
    offset = 0
    for length_tensor in seq_lens:
        length = int(length_tensor.item())
        pieces.append(tensor.narrow(0, offset, length))
        offset += length
    if offset != tensor.numel():
        raise ValueError(
            f"PackedBatch sizes sum to {offset}, tensor has {tensor.numel()} tokens."
        )
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def pack_thd_forward_kwargs(model, batch) -> dict[str, Any]:
    """Pad and zigzag-CP-split a raw THD batch into K3 forward kwargs."""
    from megatron.lite.primitive.parallel.thd import (
        PackedSeqParams,
        pack_nested_thd,
        prepare_packed_thd_kwargs_for_context_parallel,
    )

    ps = _parallel_state(model)
    packed = pack_nested_thd(
        _nested_from_packed(batch.input_ids, batch.seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=_nested_from_packed(batch.labels, batch.seq_lens),
        roll_labels=batch.labels is not None,
        loss_mask=_nested_from_packed(batch.loss_mask, batch.seq_lens),
        roll_loss_mask=batch.loss_mask is not None,
    )
    max_seqlen = (
        int(packed.padded_lengths.max().item()) if packed.padded_lengths.numel() else 0
    )
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": PackedSeqParams.from_cu_seqlens(
            packed.cu_seqlens_padded, max_seqlen=max_seqlen
        ),
    }
    prepare_packed_thd_kwargs_for_context_parallel(model, kwargs)
    return kwargs


def unpack_thd_forward_output(model, batch, output: torch.Tensor) -> torch.Tensor:
    """Reverse a zigzag-CP THD K3 output to jagged true-length form."""
    from megatron.lite.primitive.parallel.thd import thd_pack_meta, unpack_thd_to_nested

    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=False)


def pack_routed_experts(model, batch, routed_experts, *, contiguous: bool = False):
    """Pack jagged R3 routes into the local K3 router layout."""
    from megatron.lite.primitive.parallel.thd import (
        split_packed_to_cp_local,
        thd_pack_meta,
    )

    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    rows = (
        list(routed_experts.unbind(0))
        if getattr(routed_experts, "is_nested", False)
        else [routed_experts[i] for i in range(routed_experts.size(0))]
    )
    if len(rows) != int(meta.lengths.numel()) or not rows or rows[0].dim() != 3:
        raise ValueError(
            "K3 routed experts must provide one [seq, layers, topk] row per sequence."
        )
    num_layers, topk = int(rows[0].size(1)), int(rows[0].size(2))
    total_padded = int(meta.cu_seqlens_padded[-1].item())
    full = torch.zeros(
        total_padded, num_layers, topk, dtype=torch.long, device=batch.input_ids.device
    )
    for index, row in enumerate(rows):
        length = int(meta.lengths[index].item())
        if int(row.size(0)) not in (length - 1, length):
            raise ValueError(
                f"K3 routed experts sequence {index} has invalid token count."
            )
        start = int(meta.cu_seqlens_padded[index].item())
        full[start : start + row.size(0)] = row.to(device=full.device, dtype=torch.long)
    if contiguous:
        local = (
            full
            if ps.cp_size == 1
            else full[
                ps.cp_rank * (total_padded // ps.cp_size) : (ps.cp_rank + 1)
                * (total_padded // ps.cp_size)
            ]
        )
    else:
        local = split_packed_to_cp_local(
            full,
            cu_seqlens_padded=meta.cu_seqlens_padded,
            cp_size=ps.cp_size,
            cp_rank=ps.cp_rank,
            dim=0,
        )
    if ps.tp_size > 1:
        local = local[
            ps.tp_rank * (local.size(0) // ps.tp_size) : (ps.tp_rank + 1)
            * (local.size(0) // ps.tp_size)
        ]
    return [local[:, layer, :].contiguous() for layer in range(num_layers)]


def pack_r3_replay_mask(model, batch, *, contiguous: bool = False) -> torch.Tensor:
    """Build K3's causal R3 replay mask from its full-sequence loss mask."""
    rows = []
    offset = 0
    for length_tensor in batch.seq_lens:
        length = int(length_tensor.item())
        has_response = batch.loss_mask is None or bool(
            batch.loss_mask[offset : offset + length].sum().item()
        )
        row = torch.zeros(length, dtype=torch.long, device=batch.input_ids.device)
        if has_response and length > 1:
            row[:-1] = 1
        rows.append(row[:, None, None])
        offset += length
    nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    return pack_routed_experts(model, batch, nested, contiguous=contiguous)[0][
        :, 0
    ].bool()


__all__ = [
    "pack_r3_replay_mask",
    "pack_routed_experts",
    "pack_thd_forward_kwargs",
    "unpack_thd_forward_output",
]
