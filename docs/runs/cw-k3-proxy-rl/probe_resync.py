#!/usr/bin/env python3
"""Exercise VERL's bucketed actor-to-vLLM reload transaction on one GPU."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import torch
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightSender,
)
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from vllm.model_executor.model_loader.reload import (
    freeze_load_plan,
    record_metadata_for_reloading,
)


class ReloadModel(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.left = torch.nn.Parameter(
            torch.zeros(262_144, device=device), requires_grad=False
        )
        self.right = torch.nn.Parameter(
            torch.zeros(262_144, device=device), requires_grad=False
        )
        self.left.weight_loader = self._load
        self.right.weight_loader = self._load

    @staticmethod
    def _load(parameter, weight):
        parameter.data.copy_(weight)

    def load_weights(self, weights):
        loaded = set()
        for name, weight in weights:
            parameter = self.get_parameter(name)
            parameter.weight_loader(parameter, weight)
            loaded.add(name)
        return loaded


def main() -> None:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    model = ReloadModel(device)
    record_metadata_for_reloading(model)
    model.load_weights(
        (
            ("left", torch.full_like(model.left, -1)),
            ("right", torch.full_like(model.right, -2)),
        )
    )
    freeze_load_plan(model)

    actor_left = torch.arange(model.left.numel(), device=device, dtype=torch.float32)
    actor_right = actor_left.neg().sub_(3)
    socket_path = f"ipc:///tmp/k3-resync-{os.getpid()}.sock"
    sender = BucketedWeightSender(
        zmq_handle=socket_path,
        bucket_size_mb=1,
        use_shm=True,
    )
    state = {"bucket_count": 0}

    extension = SimpleNamespace(
        _is_qat_model=False,
        _is_modelopt_qat=False,
        _is_mxfp4_qat_model=True,
        device=device,
        local_rank=0,
    )
    extension.model_runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(load_format="safetensors"),
            model_config=SimpleNamespace(dtype=torch.float32),
        ),
    )
    extension._iter_all_models = MethodType(lambda self: iter((model,)), extension)
    extension._iter_all_models_with_config = MethodType(
        lambda self: iter(((model, extension.model_runner.vllm_config.model_config),)),
        extension,
    )
    extension._get_zmq_handle = MethodType(lambda self: socket_path, extension)

    def update_weights(self, weights, peft_config=None, base_sync_done=False):
        del self, peft_config, base_sync_done
        state["bucket_count"] += 1
        model.load_weights(weights)

    extension._update_weights = MethodType(update_weights, extension)

    async def actor_weights():
        yield "left", actor_left
        yield "right", actor_right

    def send_actor_weights():
        asyncio.run(sender.async_send_weights(actor_weights()))

    with ThreadPoolExecutor(max_workers=1) as executor:
        sender_future = executor.submit(send_actor_weights)
        vLLMColocateWorkerExtension.update_weights_from_ipc(
            extension,
            use_shm=True,
        )
        sender_future.result(timeout=60)

    bucket_count = state["bucket_count"]
    assert bucket_count >= 2, bucket_count
    assert torch.equal(model.left, actor_left)
    assert torch.equal(model.right, actor_right)
    print(
        "K3_RESYNC_OK="
        + json.dumps(
            {
                "bucket_count": bucket_count,
                "exact_left": True,
                "exact_right": True,
                "transport": "verl-bucketed-shm",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
