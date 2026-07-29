"""Scheduler-only K3 CP2 parity and FLA cp_context validation."""

from __future__ import annotations

import inspect
import json
import os

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite.model import K3ParallelModel

REL_FLOOR = 1.0e-3
ABS_TOLERANCE = 3.0e-2
WEIGHT_GRAD_TOLERANCE = 6.0e-2
KDA_CP_MODE = "headwise"

_elastic_error_file = os.environ.get("TORCHELASTIC_ERROR_FILE")
if _elastic_error_file and "{}" in _elastic_error_file:
    os.environ["TORCHELASTIC_ERROR_FILE"] = _elastic_error_file.format(
        os.environ.get("LOCAL_RANK", "unknown")
    )


def _config() -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=64,
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
    magnitude = reference_float.abs()
    relative_mask = magnitude > REL_FLOOR
    count_exceeds = int((difference > abs_tolerance).sum().item())
    count_total = difference.numel()
    max_rel = (
        (difference[relative_mask] / magnitude[relative_mask]).max().item()
        if relative_mask.any()
        else 0.0
    )
    return {
        "max_abs": difference.max().item() if count_total else 0.0,
        "mean_abs": difference.mean().item() if count_total else 0.0,
        "rel_floor": REL_FLOOR,
        "max_rel_above_floor": max_rel,
        "relative_count": int(relative_mask.sum().item()),
        "abs_tolerance": abs_tolerance,
        "count_exceeds": count_exceeds,
        "ratio_exceeds": count_exceeds / count_total if count_total else 0.0,
    }


def _fla_cp_context_probe(
    device: torch.device,
    cp_group: dist.ProcessGroup,
) -> dict[str, str | int | float]:
    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda

    rank = dist.get_rank(cp_group)
    generator = torch.Generator(device=device).manual_seed(1701 + rank)
    q = torch.randn(
        1, 8, 2, 128, device=device, dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    k = torch.randn(
        1, 8, 2, 128, device=device, dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    v = torch.randn(
        1, 8, 2, 128, device=device, dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    gate = torch.randn(
        1, 8, 2, 128, device=device, dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    beta = torch.full(
        (1, 8, 2),
        0.5,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    a_log = torch.zeros(2, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(256, device=device, dtype=torch.float32)
    global_cu_seqlens = torch.tensor([0, 16], device=device, dtype=torch.long)
    cp_context = build_cp_context(
        cu_seqlens=global_cu_seqlens,
        group=cp_group,
        conv1d_kernel_size=4,
    )
    implementation = inspect.unwrap(chunk_kda)
    source_file = inspect.getsourcefile(implementation) or "<unknown>"
    source_line = inspect.getsourcelines(implementation)[1]
    output, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        cp_context=cp_context,
    )
    output.float().square().mean().backward()
    if final_state is not None:
        raise AssertionError("FLA CP probe unexpectedly returned a final state")
    return {
        "status": "forward_backward_accepted",
        "source_file": source_file,
        "source_line": source_line,
        "output_max_abs": output.detach().float().abs().max().item(),
        "q_grad_max_abs": q.grad.detach().float().abs().max().item(),
    }


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("K3 CP parity requires exactly two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = _config()
    input_ids = (
        torch.arange(32, device=device, dtype=torch.long).view(1, -1)
        % config.vocab_size
    )
    labels = input_ids.roll(-1, dims=-1)

    torch.manual_seed(20260727)
    reference = (
        K3ParallelModel(
            config,
            ParallelState(),
            deterministic=False,
            kda_cp_mode=KDA_CP_MODE,
        )
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    reference_layers, reference_handles = _capture_layers(reference)
    reference_result = reference(input_ids=input_ids, labels=labels)
    reference_result["loss"].backward()
    for handle in reference_handles:
        handle.remove()

    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=2))
    torch.manual_seed(20260727)
    parallel = (
        K3ParallelModel(
            config,
            ps,
            deterministic=False,
            kda_cp_mode=KDA_CP_MODE,
        )
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    parallel.load_state_dict(reference.state_dict(), strict=True)
    assert parallel.layers[0].self_attention.A_log.dtype == torch.float32
    assert parallel.layers[0].self_attention.dt_bias.dtype == torch.float32
    parallel_layers, parallel_handles = _capture_layers(parallel)
    torch.cuda.synchronize(device)
    memory_before_parallel = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    parallel_result = parallel(input_ids=input_ids, labels=labels)
    (parallel_result["loss"] / ps.cp_size).backward()
    torch.cuda.synchronize(device)
    memory_peak_parallel = torch.cuda.max_memory_allocated(device)
    for handle in parallel_handles:
        handle.remove()

    layer_metrics = []
    parity_metrics = []
    for index, (reference_pair, parallel_pair) in enumerate(
        zip(reference_layers, parallel_layers, strict=True)
    ):
        reference_hidden, reference_block = reference_pair
        parallel_hidden, parallel_block = parallel_pair
        expected_hidden = zigzag_slice_for_cp(
            reference_hidden,
            ps.cp_rank,
            ps.cp_size,
            seq_dim=0,
        )
        expected_block = zigzag_slice_for_cp(
            reference_block,
            ps.cp_rank,
            ps.cp_size,
            seq_dim=0,
        )
        metrics = {
            "layer": index,
            "hidden_forward": _metrics(
                parallel_hidden,
                expected_hidden,
                abs_tolerance=ABS_TOLERANCE,
            ),
            "block_residual_forward": _metrics(
                parallel_block,
                expected_block,
                abs_tolerance=ABS_TOLERANCE,
            ),
            "hidden_backward": _metrics(
                parallel_hidden.grad,
                zigzag_slice_for_cp(
                    reference_hidden.grad,
                    ps.cp_rank,
                    ps.cp_size,
                    seq_dim=0,
                ),
                abs_tolerance=ABS_TOLERANCE,
            ),
            "block_residual_backward": _metrics(
                parallel_block.grad,
                zigzag_slice_for_cp(
                    reference_block.grad,
                    ps.cp_rank,
                    ps.cp_size,
                    seq_dim=0,
                ),
                abs_tolerance=ABS_TOLERANCE,
            ),
        }
        layer_metrics.append(metrics)
        parity_metrics.extend(value for key, value in metrics.items() if key != "layer")

    expected_log_probs = zigzag_slice_for_cp(
        reference_result["log_probs"],
        ps.cp_rank,
        ps.cp_size,
        seq_dim=1,
    )
    log_probs_metrics = _metrics(
        parallel_result["log_probs"],
        expected_log_probs,
        abs_tolerance=ABS_TOLERANCE,
    )
    parity_metrics.append(log_probs_metrics)

    weight_grad_metrics = {}
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in parallel.named_parameters():
        if parameter.grad is None:
            continue
        reduced_grad = parameter.grad.detach().clone()
        dist.all_reduce(reduced_grad, group=ps.cp_group)
        metric = _metrics(
            reduced_grad,
            reference_parameters[name].grad,
            abs_tolerance=WEIGHT_GRAD_TOLERANCE,
        )
        weight_grad_metrics[name] = metric
        parity_metrics.append(metric)

    loss = parallel_result["loss"].detach().clone()
    dist.all_reduce(loss, group=ps.cp_group)
    loss /= ps.cp_size
    cp_context_probe = _fla_cp_context_probe(device, ps.cp_group)
    result = {
        "mode": "cp2",
        "rank": rank,
        "world_size": world_size,
        "parallel": {"tp": 1, "ep": 1, "etp": 1, "pp": 1, "cp": 2},
        "kda_cp_mode": KDA_CP_MODE,
        "memory": {
            "allocated_before_parallel_bytes": memory_before_parallel,
            "peak_allocated_bytes": memory_peak_parallel,
            "parallel_peak_delta_bytes": (
                memory_peak_parallel - memory_before_parallel
            ),
        },
        "layers": layer_metrics,
        "log_probs": log_probs_metrics,
        "weight_grad_worst": max(
            weight_grad_metrics.items(),
            key=lambda item: item[1]["max_abs"],
        ),
        "loss_reference": reference_result["loss"].item(),
        "loss_parallel_cp_average": loss.item(),
        "loss_abs": abs(loss.item() - reference_result["loss"].item()),
        "fla_cp_context_probe": cp_context_probe,
    }
    if rank == 0:
        print("K3_CP_PARITY=" + json.dumps(result, sort_keys=True), flush=True)
    if any(metric["count_exceeds"] for metric in parity_metrics):
        raise AssertionError(f"K3 CP parity exceeded bf16 tolerance: {result}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
