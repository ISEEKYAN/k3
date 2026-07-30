#!/usr/bin/env python3
"""Run one offline Kimi-K3 proxy generation with vLLM's external launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import vllm
import vllm._custom_ops as ops
import vllm.envs as envs
import wandb
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def ensure_k3_env_compatibility() -> None:
    """Disable the optional overlap when its env registration was not merged."""
    if not hasattr(envs, "VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD"):
        setattr(envs, "VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD", 0)
        print(
            "K3_VLLM_ENV_COMPAT VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD=0",
            flush=True,
        )


def ensure_moe_sum_compatibility() -> None:
    """Use the inherited two-argument MoE binary only for the non-EP path."""
    if len(torch.ops._moe_C.moe_sum.default._schema.arguments) != 2:
        return

    def moe_sum_legacy_binary_compatibility(
        input: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        if topk_ids is not None or expert_map is not None:
            raise RuntimeError("legacy two-argument _moe_C cannot apply an expert map")
        torch.ops._moe_C.moe_sum(input, output)

    ops.moe_sum = moe_sum_legacy_binary_compatibility
    print("K3_VLLM_MOE_SUM_COMPAT schema_args=2", flush=True)


def initialize_world() -> None:
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group()


def main() -> None:
    assert "RAY_ADDRESS" not in os.environ
    assert int(os.environ["WORLD_SIZE"]) == 8
    ensure_k3_env_compatibility()
    ensure_moe_sum_compatibility()
    initialize_world()
    rank = dist.get_rank()

    if rank == 0:
        print(
            "K3_PROXY_GENERATE_PRE",
            f"vllm_version={vllm.__version__}",
            f"vllm_file={vllm.__file__}",
            f"world_size={dist.get_world_size()}",
            flush=True,
        )

    llm = LLM(
        model=os.environ["K3_MODEL_PATH"],
        trust_remote_code=True,
        tensor_parallel_size=8,
        # H100 uses the Marlin MXFP4 fallback. This overlay's Marlin post-load
        # path assumes the full expert axis and is not compatible with EP
        # sharding, so this rollout gate uses TP8 without EP.
        enable_expert_parallel=False,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.70,
        max_model_len=128,
        seed=1,
        skip_tokenizer_init=True,
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert token_ids, "generation returned no token IDs"

    if rank == 0:
        result = {
            "backend": "external_launcher",
            "world_size": dist.get_world_size(),
            "tensor_parallel_size": 8,
            "expert_parallel": False,
            "prompt_token_ids": [1, 2, 3, 4],
            "generated_token_ids": token_ids,
        }
        Path(os.environ["K3_RESPONSE_FILE"]).write_text(
            json.dumps(result, indent=2) + "\n"
        )
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "megatron-core-moe-dev"),
            project=os.environ.get("WANDB_PROJECT", "k3-proxy-generate"),
            name=os.environ.get("RUN_NAME", "k3-proxy-12l-56e-generate"),
            config=result,
        )
        run.log({"generated_tokens": len(token_ids)})
        print(f"K3_GENERATE_WANDB_URL {run.url}", flush=True)
        run.finish()
        print(
            "K3_PROXY_GENERATE_OK",
            f"generated_token_ids={token_ids}",
            flush=True,
        )
    dist.barrier()


if __name__ == "__main__":
    main()
