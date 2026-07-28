"""Scheduler-only A/E K3 combined-parallel parity, speed, and memory smoke."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite.model import K3ParallelModel

REL_FLOOR = 1.0e-3
ABS_TOLERANCE = 3.0e-2
MEASURE_STEPS = 3


def _config(sequence_length: int) -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=sequence_length,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        kda_head_dim=64,
        kda_num_heads=4,
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


def _topology(phase: str, world_size: int) -> ParallelConfig:
    if phase == "baseline":
        if world_size != 1:
            raise RuntimeError("K3 baseline phase requires one rank")
        return ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
    if phase == "combined":
        if world_size != 8:
            raise RuntimeError("K3 combined phase requires eight ranks")
        return ParallelConfig(tp=2, ep=2, etp=1, pp=1, cp=2)
    raise ValueError(f"unsupported phase: {phase!r}")


def _split_gate_up(
    tensor: torch.Tensor,
    rank: int,
    size: int,
) -> torch.Tensor:
    gate, up = tensor.chunk(2, dim=0)
    return torch.cat(
        (gate.chunk(size, dim=0)[rank], up.chunk(size, dim=0)[rank]),
        dim=0,
    ).contiguous()


def _tp_local_tensor(
    name: str,
    full: torch.Tensor,
    local_shape: torch.Size,
    rank: int,
    size: int,
) -> torch.Tensor:
    if tuple(full.shape) == tuple(local_shape):
        return full
    if (
        ".gate_up.linear.weight" in name
        and full.shape[0] == local_shape[0] * size
        and tuple(full.shape[1:]) == tuple(local_shape[1:])
    ):
        return _split_gate_up(full, rank, size)
    if full.shape[0] == local_shape[0] * size and tuple(full.shape[1:]) == tuple(
        local_shape[1:]
    ):
        return full.chunk(size, dim=0)[rank].contiguous()
    if (
        full.ndim >= 2
        and full.shape[0] == local_shape[0]
        and full.shape[1] == local_shape[1] * size
        and tuple(full.shape[2:]) == tuple(local_shape[2:])
    ):
        return full.chunk(size, dim=1)[rank].contiguous()
    raise RuntimeError(
        f"no TP placement for {name}: {tuple(full.shape)} -> {tuple(local_shape)}"
    )


def _canonical_name(name: str, ps) -> str:
    match = re.match(r"^(.*\.experts\.fc[12]\.weight)(\d+)$", name)
    if match is None:
        return name
    prefix, local_index = match.groups()
    local_experts = 8 // ps.ep_size
    return f"{prefix}{ps.ep_rank * local_experts + int(local_index)}"


def _load_canonical_state(
    model: K3ParallelModel,
    canonical: dict[str, torch.Tensor],
) -> None:
    loaded = {}
    for name, target in model.state_dict().items():
        source_name = _canonical_name(name, model.ps)
        if source_name not in canonical:
            raise RuntimeError(f"missing canonical state for {name}: {source_name}")
        loaded[name] = _tp_local_tensor(
            name,
            canonical[source_name],
            target.shape,
            model.ps.tp_rank,
            model.ps.tp_size,
        )
    model.load_state_dict(loaded, strict=True)


def _capture_layers(model: K3ParallelModel):
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(_module, _inputs, output) -> None:
        hidden, block_residual = output
        hidden.retain_grad()
        block_residual.retain_grad()
        outputs.append((hidden, block_residual))

    return outputs, [layer.register_forward_hook(capture) for layer in model.layers]


def _capture_attention_and_router(model: K3ParallelModel):
    attention_outputs: list[torch.Tensor] = []
    router_inputs: list[torch.Tensor] = []
    router_logits: list[torch.Tensor] = []
    router_indices: list[torch.Tensor] = []

    def capture_attention(_module, _inputs, output) -> None:
        output.retain_grad()
        attention_outputs.append(output)

    def capture_router(_module, inputs, output) -> None:
        router_input = inputs[0]
        indices = output[1]
        tokens = router_input.shape[0]
        router_inputs.append(router_input.view(tokens, 1, -1))
        router_indices.append(indices.view(tokens, 1, -1))

    def capture_router_gate(_module, _inputs, output) -> None:
        router_logits.append(output.view(output.shape[0], 1, output.shape[-1]))

    handles = [
        layer.self_attention.register_forward_hook(capture_attention)
        for layer in model.layers
    ]
    handles.extend(
        layer.moe.router.register_forward_hook(capture_router)
        for layer in model.layers
        if layer.moe is not None
    )
    handles.extend(
        layer.moe.router.gate.register_forward_hook(capture_router_gate)
        for layer in model.layers
        if layer.moe is not None
    )
    return attention_outputs, router_inputs, router_logits, router_indices, handles


def _router_modules(model: K3ParallelModel):
    return [layer.moe.router for layer in model.layers if layer.moe is not None]


def _selected_indices_by_expert(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Match the shared router contract: select by score, order by expert id."""
    selected = torch.topk(scores, topk, dim=-1, sorted=False).indices
    return selected.sort(dim=-1).values


def _router_margin(
    scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = scores.sort(dim=-1, descending=True).values
    top1_top2 = ordered[..., 0] - ordered[..., 1]
    decision_boundary = ordered[..., topk - 1] - ordered[..., topk]
    return top1_top2, decision_boundary


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": values.min().item(),
        "median": values.median().item(),
        "max": values.max().item(),
    }


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = (actual - reference).abs()
    relative_mask = reference.abs() > REL_FLOOR
    count_exceeds = int((difference > ABS_TOLERANCE).sum().item())
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "rel_floor": REL_FLOOR,
        "max_rel_above_floor": (
            (difference[relative_mask] / reference.abs()[relative_mask]).max().item()
            if relative_mask.any()
            else 0.0
        ),
        "relative_count": int(relative_mask.sum().item()),
        "count_exceeds": count_exceeds,
        "ratio_exceeds": count_exceeds / difference.numel(),
    }


def _expected_layer_shard(reference: torch.Tensor, ps) -> torch.Tensor:
    """Mirror model entry layout: CP zigzag slice, then contiguous TP/SP slice."""
    reference = zigzag_slice_for_cp(
        reference,
        ps.cp_rank,
        ps.cp_size,
        seq_dim=0,
    )
    local_sequence = reference.shape[0] // ps.tp_size
    return reference.narrow(
        0,
        ps.tp_rank * local_sequence,
        local_sequence,
    ).contiguous()


def _run_step(
    model: K3ParallelModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_divisor: int,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels)
    (output["loss"] / loss_divisor).backward()
    return output


def _measure(
    model: K3ParallelModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_divisor: int,
) -> tuple[float, int, int]:
    _run_step(model, input_ids, labels, loss_divisor=loss_divisor)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        _run_step(model, input_ids, labels, loss_divisor=loss_divisor)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / MEASURE_STEPS
    return (
        elapsed_ms,
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    phase = os.environ["K3_COMBINED_PHASE"]
    sequence_length = int(os.environ.get("K3_SEQUENCE_LENGTH", "4096"))
    artifact_dir = Path(os.environ["K3_COMBINED_ARTIFACT"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ps = init_parallel(_topology(phase, world_size))
    config = _config(sequence_length)

    torch.manual_seed(20260727)
    model = (
        K3ParallelModel(
            config,
            ps,
            deterministic=False,
            kda_cp_mode="headwise",
        )
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    if phase == "combined":
        canonical = torch.load(
            artifact_dir / "canonical.pt",
            map_location="cpu",
            weights_only=False,
        )
        _load_canonical_state(model, canonical)
        del canonical

    input_ids = (
        torch.arange(sequence_length, device=device, dtype=torch.long).view(1, -1)
        % config.vocab_size
    )
    labels = input_ids.roll(-1, dims=-1)
    step_ms, peak_allocated, peak_reserved = _measure(
        model,
        input_ids,
        labels,
        loss_divisor=ps.cp_size,
    )

    captured, handles = _capture_layers(model)
    (
        attention_outputs,
        router_inputs,
        router_logits,
        router_indices,
        diagnostic_handles,
    ) = _capture_attention_and_router(model)
    output = _run_step(
        model,
        input_ids,
        labels,
        loss_divisor=ps.cp_size,
    )
    for handle in handles:
        handle.remove()
    for handle in diagnostic_handles:
        handle.remove()

    if phase == "baseline":
        if rank == 0:
            torch.save(model.state_dict(), artifact_dir / "canonical.pt")
            torch.save(
                {
                    "layers": [
                        {
                            "hidden": hidden.detach().cpu(),
                            "hidden_grad": hidden.grad.detach().cpu(),
                            "block_residual": block.detach().cpu(),
                            "block_residual_grad": block.grad.detach().cpu(),
                        }
                        for hidden, block in captured
                    ],
                    "attention": [
                        {
                            "output": attention.detach().cpu(),
                            "output_grad": attention.grad.detach().cpu(),
                        }
                        for attention in attention_outputs
                    ],
                    "router_indices": [
                        indices.detach().cpu() for indices in router_indices
                    ],
                    "router_inputs": [
                        router_input.detach().cpu() for router_input in router_inputs
                    ],
                    "router_logits": [
                        logits.detach().cpu() for logits in router_logits
                    ],
                    "router_fp32_logits": [
                        F.linear(
                            router_input[:, 0].float(),
                            router.gate.weight.float(),
                        )
                        .view(router_input.shape[0], 1, -1)
                        .detach()
                        .cpu()
                        for router_input, router in zip(
                            router_inputs,
                            _router_modules(model),
                            strict=True,
                        )
                    ],
                    "log_probs": output["log_probs"].detach().cpu(),
                },
                artifact_dir / "baseline_outputs.pt",
            )
            result = {
                "phase": "A",
                "parallel": {"tp": 1, "ep": 1, "etp": 1, "pp": 1, "cp": 1},
                "sequence_length": sequence_length,
                "step_ms": step_ms,
                "peak_allocated_bytes_per_rank": [peak_allocated],
                "peak_reserved_bytes_per_rank": [peak_reserved],
            }
            (artifact_dir / "baseline_result.json").write_text(
                json.dumps(result, sort_keys=True)
            )
            print("K3_COMBINED_SMOKE=" + json.dumps(result, sort_keys=True), flush=True)
    else:
        reference = torch.load(
            artifact_dir / "baseline_outputs.pt",
            map_location=device,
            weights_only=False,
        )
        layer_metrics = []
        for index, ((hidden, block), expected) in enumerate(
            zip(captured, reference["layers"], strict=True)
        ):
            metrics = {
                "layer": index,
                "hidden_forward": _metrics(
                    hidden,
                    _expected_layer_shard(expected["hidden"], ps),
                ),
                "hidden_backward": _metrics(
                    hidden.grad,
                    _expected_layer_shard(expected["hidden_grad"], ps),
                ),
                "block_residual_forward": _metrics(
                    block,
                    _expected_layer_shard(expected["block_residual"], ps),
                ),
                "block_residual_backward": _metrics(
                    block.grad,
                    _expected_layer_shard(expected["block_residual_grad"], ps),
                ),
            }
            layer_metrics.append(metrics)
        expected_log_probs = zigzag_slice_for_cp(
            reference["log_probs"],
            ps.cp_rank,
            ps.cp_size,
            seq_dim=1,
        )
        log_probs = _metrics(output["log_probs"], expected_log_probs)
        attention_metrics = []
        for index, (attention, expected) in enumerate(
            zip(attention_outputs, reference["attention"], strict=True)
        ):
            metrics = {
                "layer": index,
                "forward": _metrics(
                    attention,
                    _expected_layer_shard(expected["output"], ps),
                ),
                "backward": _metrics(
                    attention.grad,
                    _expected_layer_shard(expected["output_grad"], ps),
                ),
            }
            attention_metrics.append(metrics)
        router_metrics = []
        global_token_ids = _expected_layer_shard(
            torch.arange(sequence_length, device=device).view(-1, 1, 1),
            ps,
        ).flatten()
        cp_global_token_ids = zigzag_slice_for_cp(
            torch.arange(sequence_length, device=device),
            ps.cp_rank,
            ps.cp_size,
            seq_dim=0,
        )
        for index, (
            router_input,
            logits,
            indices,
            expected_input,
            expected_logits,
            expected_fp32_logits,
            expected_indices,
            router,
        ) in enumerate(
            zip(
                router_inputs,
                router_logits,
                router_indices,
                reference["router_inputs"],
                reference["router_logits"],
                reference["router_fp32_logits"],
                reference["router_indices"],
                _router_modules(model),
                strict=True,
            )
        ):
            expected_input = _expected_layer_shard(expected_input, ps)
            expected_logits = _expected_layer_shard(expected_logits, ps)
            expected_fp32_logits = _expected_layer_shard(
                expected_fp32_logits,
                ps,
            )
            expected_indices = _expected_layer_shard(expected_indices, ps)
            fp32_logits = F.linear(
                router_input[:, 0].float(),
                router.gate.weight.float(),
            ).view(router_input.shape[0], 1, -1)
            baseline_scores = torch.sigmoid(expected_logits.float())
            parallel_scores = torch.sigmoid(logits.float())
            baseline_fp32_scores = torch.sigmoid(expected_fp32_logits)
            parallel_fp32_scores = torch.sigmoid(fp32_logits)
            topk = indices.shape[-1]
            if not torch.equal(
                _selected_indices_by_expert(baseline_scores, topk),
                expected_indices,
            ):
                raise AssertionError(
                    "captured A router indices do not match its logits"
                )
            if not torch.equal(
                _selected_indices_by_expert(parallel_scores, topk),
                indices,
            ):
                raise AssertionError(
                    "captured E router indices do not match its logits"
                )

            fp32_expected_indices = _selected_indices_by_expert(
                baseline_fp32_scores,
                topk,
            )
            fp32_indices = _selected_indices_by_expert(
                parallel_fp32_scores,
                topk,
            )
            mismatches = indices != expected_indices
            mismatched_tokens = mismatches.any(dim=-1).flatten()
            set_changed = (
                (indices.sort(dim=-1).values != expected_indices.sort(dim=-1).values)
                .any(dim=-1)
                .flatten()
            )
            fp32_mismatches = fp32_indices != fp32_expected_indices
            fp32_mismatched_tokens = fp32_mismatches.any(dim=-1).flatten()
            fp32_set_changed = (
                (
                    fp32_indices.sort(dim=-1).values
                    != fp32_expected_indices.sort(dim=-1).values
                )
                .any(dim=-1)
                .flatten()
            )
            baseline_top12, baseline_boundary = _router_margin(
                baseline_scores,
                topk,
            )
            parallel_top12, parallel_boundary = _router_margin(
                parallel_scores,
                topk,
            )
            flip_records = []
            for local_token in mismatched_tokens.nonzero().flatten().tolist():
                global_token = int(global_token_ids[local_token].item())
                log_prob_position = int(
                    (cp_global_token_ids == global_token).nonzero().item()
                )
                baseline_order = baseline_scores[local_token, 0].sort(descending=True)
                parallel_order = parallel_scores[local_token, 0].sort(descending=True)
                baseline_scale = (
                    torch.finfo(torch.bfloat16).eps
                    * baseline_order.values[: topk + 1].abs().max()
                )
                parallel_scale = (
                    torch.finfo(torch.bfloat16).eps
                    * parallel_order.values[: topk + 1].abs().max()
                )
                input_difference = (
                    router_input[local_token].float()
                    - expected_input[local_token].float()
                ).abs()
                flip_records.append(
                    {
                        "global_token": global_token,
                        "baseline_indices": expected_indices[local_token, 0].tolist(),
                        "parallel_indices": indices[local_token, 0].tolist(),
                        "baseline_fp32_indices": fp32_expected_indices[
                            local_token, 0
                        ].tolist(),
                        "parallel_fp32_indices": fp32_indices[local_token, 0].tolist(),
                        "baseline_top_scores": baseline_order.values[
                            : topk + 1
                        ].tolist(),
                        "parallel_top_scores": parallel_order.values[
                            : topk + 1
                        ].tolist(),
                        "baseline_top1_top2_margin": baseline_top12[
                            local_token, 0
                        ].item(),
                        "parallel_top1_top2_margin": parallel_top12[
                            local_token, 0
                        ].item(),
                        "baseline_topk_boundary_margin": baseline_boundary[
                            local_token, 0
                        ].item(),
                        "parallel_topk_boundary_margin": parallel_boundary[
                            local_token, 0
                        ].item(),
                        "baseline_bf16_effective_score_scale": (baseline_scale.item()),
                        "parallel_bf16_effective_score_scale": (parallel_scale.item()),
                        "router_input_max_abs": input_difference.max().item(),
                        "router_input_mean_abs": input_difference.mean().item(),
                        "router_logit_max_abs": (
                            logits[local_token].float()
                            - expected_logits[local_token].float()
                        )
                        .abs()
                        .max()
                        .item(),
                        "log_prob_abs": (
                            output["log_probs"][0, log_prob_position].float()
                            - expected_log_probs[0, log_prob_position].float()
                        )
                        .abs()
                        .item(),
                    }
                )
            input_metrics = _metrics(router_input, expected_input)
            router_metrics.append(
                {
                    "moe_index": index,
                    "rank": rank,
                    "mismatched_indices": int(mismatches.sum().item()),
                    "total_indices": mismatches.numel(),
                    "ratio_mismatched": mismatches.float().mean().item(),
                    "mismatched_global_tokens": global_token_ids[
                        mismatched_tokens
                    ].tolist(),
                    "set_changed_global_tokens": global_token_ids[set_changed].tolist(),
                    "fp32_mismatched_indices": int(fp32_mismatches.sum().item()),
                    "fp32_mismatched_global_tokens": global_token_ids[
                        fp32_mismatched_tokens
                    ].tolist(),
                    "fp32_set_changed_global_tokens": global_token_ids[
                        fp32_set_changed
                    ].tolist(),
                    "router_input": input_metrics,
                    "flipped_baseline_top1_top2": _distribution(
                        baseline_top12.flatten()[mismatched_tokens]
                    ),
                    "flipped_baseline_topk_boundary": _distribution(
                        baseline_boundary.flatten()[mismatched_tokens]
                    ),
                    "flipped_parallel_top1_top2": _distribution(
                        parallel_top12.flatten()[mismatched_tokens]
                    ),
                    "flipped_parallel_topk_boundary": _distribution(
                        parallel_boundary.flatten()[mismatched_tokens]
                    ),
                    "flip_records": flip_records,
                }
            )

        step_tensor = torch.tensor(step_ms, device=device)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
        allocated_by_rank = [None] * world_size
        reserved_by_rank = [None] * world_size
        router_by_rank = [None] * world_size
        dist.all_gather_object(allocated_by_rank, peak_allocated)
        dist.all_gather_object(reserved_by_rank, peak_reserved)
        dist.all_gather_object(router_by_rank, router_metrics)
        result = {
            "phase": "E",
            "parallel": {"tp": 2, "ep": 2, "etp": 1, "pp": 1, "cp": 2},
            "kda_cp_mode": "headwise",
            "sequence_length": sequence_length,
            "step_ms_slowest_rank": step_tensor.item(),
            "peak_allocated_bytes_per_rank": allocated_by_rank,
            "peak_reserved_bytes_per_rank": reserved_by_rank,
            "layers": layer_metrics,
            "attention": attention_metrics,
            "router": router_by_rank,
            "log_probs": log_probs,
        }
        baseline_result = json.loads(
            (artifact_dir / "baseline_result.json").read_text()
        )
        baseline_peak = baseline_result["peak_allocated_bytes_per_rank"][0]
        memory_reduction = 1.0 - max(allocated_by_rank) / baseline_peak
        result["peak_allocated_reduction_vs_A"] = memory_reduction
        attention_failures = [
            metric
            for layer in attention_metrics
            for metric in (layer["forward"], layer["backward"])
            if metric["count_exceeds"]
        ]
        if memory_reduction < 0.10:
            raise AssertionError(
                f"K3 headwise combined memory reduction is not significant: {result}"
            )
        if attention_failures:
            raise AssertionError(
                f"K3 combined attention parity exceeded tolerance: {result}"
            )
        if rank == 0:
            (artifact_dir / "combined_result.json").write_text(
                json.dumps(result, sort_keys=True)
            )
            print("K3_COMBINED_SMOKE=" + json.dumps(result, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
