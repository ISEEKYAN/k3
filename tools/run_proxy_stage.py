#!/usr/bin/env python3
"""Run one fail-local K3 proxy training stage on an 8-rank Slurm allocation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.lite.protocol import (
    ImplConfig,
    build_model,
    build_model_config,
    is_expert_param,
)


def _mark(rank: int, phase: str) -> None:
    print(
        "K3_PROXY_PHASE="
        + json.dumps(
            {
                "rank": rank,
                "phase": phase,
                "time": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _logical_numel(model: torch.nn.Module, device: torch.device) -> dict[str, int]:
    expert = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if is_expert_param(name)
    )
    dense = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not is_expert_param(name)
    )
    expert_tensor = torch.tensor(expert, device=device, dtype=torch.int64)
    dense_tensor = torch.tensor(dense, device=device, dtype=torch.int64)
    expert_min = expert_tensor.clone()
    expert_max = expert_tensor.clone()
    dist.all_reduce(expert_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(expert_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(expert_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(dense_tensor, op=dist.ReduceOp.MAX)
    if expert_min.item() != expert_max.item():
        raise RuntimeError(
            f"uneven EP expert numel min={expert_min.item()} max={expert_max.item()}"
        )
    return {
        "dense_replicated_once": int(dense_tensor.item()),
        "expert_local": int(expert_min.item()),
        "expert_across_ep": int(expert_tensor.item()),
        "logical_total": int(dense_tensor.item() + expert_tensor.item()),
    }


@record
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("construct", "fwbw", "qat"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"K3 proxy stages require exactly 8 ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _mark(rank, "dist_ready")

    config = build_model_config(args.model_path)
    if config.num_hidden_layers != 12 or config.num_experts != 56:
        raise RuntimeError(
            "proxy config mismatch: "
            f"layers={config.num_hidden_layers} experts={config.num_experts}"
        )
    _mark(rank, "build_start")
    bundle = build_model(
        config,
        impl_cfg=ImplConfig(
            parallel=ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1),
            device=f"cuda:{local_rank}",
            dtype="bfloat16",
            qat={
                "enabled": args.stage == "qat",
                "format": "mxfp4",
                "group_size": 32,
            },
            moe_router_fusion=False,
        ),
    )
    model = bundle.chunks[0].train()
    _mark(rank, "build_done")
    if any(getattr(module, "moe_router_fusion", False) for module in model.modules()):
        raise RuntimeError("R3 carrier found a fused router")
    counts = _logical_numel(model, device)
    _mark(rank, "numel_done")
    result: dict[str, object] = {
        "stage": args.stage,
        "world_size": world_size,
        "layers": config.num_hidden_layers,
        "experts": config.num_experts,
        "topk": config.num_experts_per_token,
        "numel_formula": counts,
        "qat_modules": bundle.extras["qat"]["quantized_modules"],
        "moe_router_fusion": False,
    }

    if args.stage != "construct":
        tokens = (
            torch.arange(args.sequence_length, device=device, dtype=torch.long)
            % config.vocab_size
        )
        batch = SimpleNamespace(
            input_ids=tokens.view(1, -1),
            labels=tokens.roll(-1).view(1, -1),
            loss_mask=torch.ones(1, args.sequence_length, device=device),
        )
        model.zero_grad(set_to_none=True)
        _mark(rank, "forward_start")
        output = bundle.forward_step(model, batch)
        _mark(rank, "forward_done")
        loss = output["loss"]
        _mark(rank, "backward_start")
        loss.backward()
        _mark(rank, "backward_done")
        finite_grads = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not torch.isfinite(loss) or not finite_grads:
            raise RuntimeError(
                f"non-finite proxy step loss={float(loss.detach())} "
                f"finite_grads={finite_grads}"
            )
        result["loss"] = float(loss.detach())
        result["finite_grads"] = finite_grads

    if rank == 0:
        print("K3_PROXY_STAGE_OK=" + json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
