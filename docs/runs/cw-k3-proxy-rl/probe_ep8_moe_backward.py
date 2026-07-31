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


def mark(rank: int, phase: str, **details: int) -> None:
    print(
        "K3_EP8_MOE_PHASE="
        + json.dumps(
            {
                "rank": rank,
                "phase": phase,
                "time": datetime.datetime.now(datetime.UTC).isoformat(),
                **details,
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


def mark_backward_entry(rank: int, layer: int, grad: torch.Tensor) -> torch.Tensor:
    mark(rank, "backward_layer_enter", layer=layer)
    return grad


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
    layers = int(os.environ.get("MOE_LAYERS", "1"))
    tokens = int(os.environ.get("TOKENS", "3072"))
    if not 1 <= layers <= 3:
        raise RuntimeError(f"MOE_LAYERS must be in [1, 3], got {layers}")
    if not 1 <= tokens <= 4096:
        raise RuntimeError(f"TOKENS must be in [1, 4096], got {tokens}")

    mark(rank, "build_start")
    modules = torch.nn.ModuleList(
        [
            ParallelLatentMoE(config, ps).to(
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(layers)
        ]
    )
    mark(rank, "build_done")
    generator = torch.Generator(device=device).manual_seed(20260730)
    hidden = torch.randn(
        tokens,
        1,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )

    mark(rank, "forward_start")
    output = hidden
    for layer, module in enumerate(modules):
        output = module(output)
        output.register_hook(
            lambda grad, layer=layer: mark_backward_entry(rank, layer, grad)
        )
        mark(rank, "forward_layer_done", layer=layer)
    mark(rank, "forward_done")
    mark(rank, "backward_start")
    output.float().square().mean().backward()
    mark(rank, "backward_done")

    first_module = modules[0]
    last_module = modules[-1]
    router_parameter = next(first_module.router.parameters())
    samples = {
        "hidden": finite_sample(hidden.grad),
        "router": finite_sample(router_parameter.grad),
        "first_down_proj": finite_sample(
            first_module.routed_expert_down_proj.weight.grad
        ),
        "last_up_proj": finite_sample(last_module.routed_expert_up_proj.weight.grad),
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
                    "layers": layers,
                    "tokens": tokens,
                    "gradient_samples": samples,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
