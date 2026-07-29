"""Scheduler-only K3 EP and ETP parity validation."""

from __future__ import annotations

import json
import os
import re

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite.model import K3ParallelModel

REL_FLOOR = 1.0e-3
LAYER_ABS_TOLERANCE = 3.0e-2
LOGITS_ABS_TOLERANCE = 6.0e-2


def _config() -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=32,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        kda_num_heads=2,
        kda_short_conv_kernel_size=4,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
    )


def _topology(mode: str, world_size: int) -> ParallelConfig:
    if mode == "prewarm" and world_size == 1:
        return ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
    if world_size != 8:
        raise RuntimeError(f"K3 {mode} parity requires exactly 8 ranks")
    if mode == "ep8":
        return ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1)
    if mode == "ep4_etp2":
        return ParallelConfig(tp=1, ep=4, etp=2, pp=1, cp=1)
    raise ValueError(f"unsupported K3 EP/ETP parity mode: {mode!r}")


def _expert_reference_name(name: str, ps) -> tuple[str, str] | None:
    match = re.match(r"^(.*\.experts\.(fc[12])\.weight)(\d+)$", name)
    if match is None:
        return None
    prefix, projection, local_index = match.groups()
    num_local = 8 // ps.ep_size
    global_index = ps.ep_rank * num_local + int(local_index)
    return f"{prefix}{global_index}", projection


def _copy_parallel_weights(
    reference: K3ParallelModel,
    parallel: K3ParallelModel,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    ps = parallel.ps
    mapped_expert_parameters = 0
    with torch.no_grad():
        for name, parameter in parallel.named_parameters():
            expert_mapping = _expert_reference_name(name, ps)
            mapped_expert_parameters += expert_mapping is not None
            source_name, projection = (
                expert_mapping if expert_mapping is not None else (name, "")
            )
            full = reference_parameters[source_name].detach()
            if projection == "fc1" and ps.etp_size > 1:
                gate, up = full.chunk(2, dim=0)
                full = torch.cat(
                    (
                        gate.chunk(ps.etp_size, dim=0)[ps.etp_rank],
                        up.chunk(ps.etp_size, dim=0)[ps.etp_rank],
                    ),
                    dim=0,
                )
            elif projection == "fc2" and ps.etp_size > 1:
                full = full.chunk(ps.etp_size, dim=1)[ps.etp_rank]
            if full.shape != parameter.shape:
                raise RuntimeError(
                    f"invalid EP/ETP placement for {name}: "
                    f"{tuple(full.shape)} -> {tuple(parameter.shape)}"
                )
            parameter.copy_(full)
    expected_expert_parameters = 2 * (8 // ps.ep_size)
    if mapped_expert_parameters != expected_expert_parameters:
        raise RuntimeError(
            "K3 EP/ETP parity did not map every local expert parameter: "
            f"{mapped_expert_parameters} != {expected_expert_parameters}"
        )


def _capture_layers(
    model: K3ParallelModel,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list]:
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(_module, _inputs, output) -> None:
        hidden_states, block_residual = output
        hidden_states.retain_grad()
        block_residual.retain_grad()
        outputs.append((hidden_states, block_residual))

    handles = [layer.register_forward_hook(capture) for layer in model.layers]
    return outputs, handles


def _metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    abs_tolerance: float,
) -> dict[str, float | int]:
    actual_float = actual.detach().float()
    reference_float = reference.detach().float()
    difference = (actual_float - reference_float).abs()
    reference_magnitude = reference_float.abs()
    relative_mask = reference_magnitude > REL_FLOOR
    count_exceeds = int((difference > abs_tolerance).sum().item())
    count_total = difference.numel()
    if relative_mask.any():
        max_rel = (
            (difference[relative_mask] / reference_magnitude[relative_mask])
            .max()
            .item()
        )
        relative_count = int(relative_mask.sum().item())
    else:
        max_rel = 0.0
        relative_count = 0
    return {
        "max_abs": difference.max().item() if count_total else 0.0,
        "mean_abs": difference.mean().item() if count_total else 0.0,
        "rel_floor": REL_FLOOR,
        "max_rel_above_floor": max_rel,
        "relative_count": relative_count,
        "abs_tolerance": abs_tolerance,
        "count_exceeds": count_exceeds,
        "ratio_exceeds": count_exceeds / count_total if count_total else 0.0,
    }


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mode = os.environ["K3_PARALLEL_MODE"]
    topology = _topology(mode, world_size)
    config = _config()

    torch.manual_seed(20260727)
    reference = (
        K3ParallelModel(config, ParallelState())
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    assert reference.layers[0].self_attention.A_log.dtype == torch.float32
    assert reference.layers[0].self_attention.dt_bias.dtype == torch.float32
    reference_outputs, reference_handles = _capture_layers(reference)
    input_ids = torch.tensor(
        [[1, 7, 3, 9, 2, 8, 5, 4, 11, 6, 13, 12, 10, 15, 14, 16]],
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.roll(-1, dims=-1)
    reference_result = reference(input_ids=input_ids, labels=labels)
    reference_result["loss"].backward()
    for handle in reference_handles:
        handle.remove()

    ps = init_parallel(topology)
    parallel = (
        K3ParallelModel(config, ps).to(device=device, dtype=torch.bfloat16).train()
    )
    assert parallel.layers[0].self_attention.A_log.dtype == torch.float32
    assert parallel.layers[0].self_attention.dt_bias.dtype == torch.float32
    _copy_parallel_weights(reference, parallel)
    parallel_outputs, parallel_handles = _capture_layers(parallel)
    parallel_result = parallel(input_ids=input_ids, labels=labels)
    parallel_result["loss"].backward()
    for handle in parallel_handles:
        handle.remove()

    layer_metrics = []
    for index, (expected_pair, actual_pair) in enumerate(
        zip(reference_outputs, parallel_outputs, strict=True)
    ):
        expected_hidden, expected_block_residual = expected_pair
        actual_hidden, actual_block_residual = actual_pair
        hidden_forward = _metrics(
            actual_hidden,
            expected_hidden,
            abs_tolerance=LAYER_ABS_TOLERANCE,
        )
        block_forward = _metrics(
            actual_block_residual,
            expected_block_residual,
            abs_tolerance=LAYER_ABS_TOLERANCE,
        )
        hidden_backward = _metrics(
            actual_hidden.grad,
            expected_hidden.grad,
            abs_tolerance=LAYER_ABS_TOLERANCE,
        )
        block_backward = _metrics(
            actual_block_residual.grad,
            expected_block_residual.grad,
            abs_tolerance=LAYER_ABS_TOLERANCE,
        )
        layer_metrics.append(
            {
                "layer": index,
                "hidden_forward": hidden_forward,
                "block_residual_forward": block_forward,
                "hidden_backward": hidden_backward,
                "block_residual_backward": block_backward,
            }
        )
    log_probs_metrics = _metrics(
        parallel_result["log_probs"],
        reference_result["log_probs"],
        abs_tolerance=LOGITS_ABS_TOLERANCE,
    )
    result = {
        "mode": mode,
        "rank": rank,
        "world_size": world_size,
        "parallel": {
            "tp": ps.tp_size,
            "ep": ps.ep_size,
            "etp": ps.etp_size,
            "pp": ps.pp_size,
            "cp": ps.cp_size,
        },
        "layers": layer_metrics,
        "log_probs": log_probs_metrics,
        "loss_reference": reference_result["loss"].item(),
        "loss_parallel": parallel_result["loss"].item(),
        "loss_abs": abs(
            parallel_result["loss"].item() - reference_result["loss"].item()
        ),
    }
    if rank == 0:
        print("K3_EP_ETP_PARITY=" + json.dumps(result, sort_keys=True), flush=True)
    parity_metrics = [log_probs_metrics]
    for layer in layer_metrics:
        parity_metrics.extend(
            (
                layer["hidden_forward"],
                layer["block_residual_forward"],
                layer["hidden_backward"],
                layer["block_residual_backward"],
            )
        )
    if any(metric["count_exceeds"] for metric in parity_metrics):
        raise AssertionError(f"K3 {mode} parity exceeded bf16 tolerance: {result}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
