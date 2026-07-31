"""Build the production-width EP8 actor and validate dist-opt grad buffers."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from mlite_k3.lite.protocol import ImplConfig, build_model, build_model_config
from mlite_k3.primitive.router import K3SigmoidTopKRouter


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("K3 EP8 dist-opt init probe requires exactly 8 ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = build_model_config(os.environ["MODEL_PATH"])
    torch.cuda.reset_peak_memory_stats(device)
    bundle = build_model(
        config,
        impl_cfg=ImplConfig(
            parallel=ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1),
            optimizer="dist_opt",
            optimizer_config=OptimizerConfig(
                optimizer="adam",
                lr=1.0e-6,
                weight_decay=0.1,
                clip_grad=1.0,
                offload_fraction=1.0,
                use_precision_aware_optimizer=True,
                decoupled_weight_decay=True,
            ),
            device=str(device),
            dtype="bfloat16",
            use_thd=True,
            moe_router_fusion=False,
            grad_reduce_in_fp32=False,
            qat={"enabled": True, "format": "mxfp4", "group_size": 32},
        ),
    )
    if bundle.finalize_grads is None:
        raise RuntimeError("K3 dist-opt bundle did not expose finalize_grads")
    pg_collection = bundle.optimizer._dist_opt_pg_collection
    if not hasattr(pg_collection, "tp_dp_cp") or pg_collection.tp_dp_cp is None:
        raise RuntimeError("K3 dist-opt pg_collection did not expose tp_dp_cp")
    router_input = torch.zeros(
        1, config.hidden_size, dtype=torch.bfloat16, device=device
    )
    routers = [
        module
        for chunk in bundle.chunks
        for module in chunk.modules()
        if isinstance(module, K3SigmoidTopKRouter)
    ]
    if not routers:
        raise RuntimeError("K3 dist-opt probe found no K3 sigmoid routers")
    for router in routers:
        router(router_input)
        if router.local_tokens_per_expert.sum().item() != router.topk:
            raise RuntimeError(
                "K3 router did not accumulate its dispatched expert counts"
            )
    bundle.finalize_grads()
    if any(torch.count_nonzero(router.local_tokens_per_expert) for router in routers):
        raise RuntimeError("K3 finalize_grads did not reset router expert counts")
    print(
        "K3_EP8_DISTOPT_FINALIZE_OK="
        + json.dumps({"ep": 8, "rank": rank, "world_size": 8}, sort_keys=True),
        flush=True,
    )
    chunk = bundle.chunks[0]
    buffers = [*chunk.buffers, *chunk.expert_parallel_buffers]
    dtype_bytes: dict[str, int] = {}
    mismatched_dtypes: list[dict[str, str]] = []
    for buffer in buffers:
        param_dtype = str(buffer.param_dtype)
        grad_dtype = str(buffer.grad_dtype)
        if grad_dtype != param_dtype:
            mismatched_dtypes.append(
                {"grad_dtype": grad_dtype, "param_dtype": param_dtype}
            )
        dtype_bytes[grad_dtype] = dtype_bytes.get(grad_dtype, 0) + (
            buffer.grad_data.numel() * buffer.grad_data.element_size()
        )
    if mismatched_dtypes:
        raise RuntimeError(
            "K3 dist-opt expected every grad buffer to match its parameter dtype, "
            f"got {mismatched_dtypes}"
        )
    largest_buffer = max(
        buffers,
        key=lambda buffer: buffer.grad_data.numel() * buffer.grad_data.element_size(),
    )
    largest_buffer_bytes = (
        largest_buffer.grad_data.numel() * largest_buffer.grad_data.element_size()
    )
    if largest_buffer.grad_dtype != torch.bfloat16:
        raise RuntimeError(
            "K3 dist-opt expected the largest grad buffer to be BF16, "
            f"got {largest_buffer.grad_dtype}"
        )
    grad_dtypes = sorted(dtype_bytes)
    evidence = {
        "buffers": len(buffers),
        "bytes_by_grad_dtype": dtype_bytes,
        "ep": 8,
        "grad_dtypes": grad_dtypes,
        "largest_buffer_dtype": str(largest_buffer.grad_dtype),
        "largest_buffer_gib": round(largest_buffer_bytes / (1024**3), 3),
        "max_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / (1024**3), 3
        ),
        "max_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / (1024**3), 3
        ),
        "rank": rank,
        "world_size": dist.get_world_size(),
    }
    print("K3_EP8_DISTOPT_RANK_OK=" + json.dumps(evidence, sort_keys=True), flush=True)
    dist.barrier()
    if rank == 0:
        print(
            "K3_EP8_DISTOPT_INIT_OK="
            + json.dumps(
                {
                    "bytes_by_grad_dtype": dtype_bytes,
                    "ep": 8,
                    "grad_dtypes": grad_dtypes,
                    "largest_buffer_dtype": str(largest_buffer.grad_dtype),
                    "largest_buffer_gib": round(largest_buffer_bytes / (1024**3), 3),
                    "world_size": 8,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
