"""Isolate one production-shape K3 EP8 LatentMoE forward/backward."""

from __future__ import annotations

import datetime
import json
import os

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.lite.model import ParallelLatentMoE
from mlite_k3.lite.protocol import build_model_config


def mark(rank: int, phase: str) -> None:
    print(
        "K3_EP8_MOE_PHASE="
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


def finite_sample(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    flat = tensor.detach().reshape(-1)
    return bool(torch.isfinite(flat[: min(4096, flat.numel())]).all())


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("K3 EP8 MoE probe requires exactly 8 ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    ps = init_parallel(ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1))
    config = build_model_config(os.environ["MODEL_PATH"])

    mark(rank, "build_start")
    module = ParallelLatentMoE(config, ps).to(
        device=device,
        dtype=torch.bfloat16,
    )
    mark(rank, "build_done")
    generator = torch.Generator(device=device).manual_seed(20260730)
    hidden = torch.randn(
        16,
        1,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )

    mark(rank, "forward_start")
    output = module(hidden)
    mark(rank, "forward_done")
    mark(rank, "backward_start")
    output.float().square().mean().backward()
    mark(rank, "backward_done")

    samples = {
        "hidden": finite_sample(hidden.grad),
        "router": finite_sample(module.router.weight.grad),
        "down_proj": finite_sample(module.routed_expert_down_proj.weight.grad),
        "up_proj": finite_sample(module.routed_expert_up_proj.weight.grad),
    }
    if not all(samples.values()):
        raise RuntimeError(f"K3 EP8 MoE missing/non-finite gradient samples: {samples}")
    if rank == 0:
        print(
            "K3_EP8_MOE_BACKWARD_OK="
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "hidden_size": config.hidden_size,
                    "experts": config.num_experts,
                    "topk": config.num_experts_per_token,
                    "gradient_samples": samples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
