"""Scheduler-only K3 R3 record/replay plus MXFP4-QAT smoke."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.modules.router_replay import RouterReplay
from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite import protocol
from mlite_k3.lite.protocol import ImplConfig, build_model


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


def _run_forward(model, batch):
    return model(
        input_ids=batch.input_ids.view(1, -1),
        labels=batch.labels.view(1, -1),
    )


def _driver(model, action: str) -> RouterReplayDriver:
    handle = SimpleNamespace(
        _model=model,
        _extras={"model_chunks": [model], "protocol": protocol},
    )
    driver = RouterReplayDriver.maybe_create(handle, {"action": action})
    assert driver is not None
    return driver


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError("K3 R3/QAT smoke requires eight ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    sequence_length = int(os.environ.get("K3_SEQUENCE_LENGTH", "256"))

    torch.manual_seed(20260728)
    bundle = build_model(
        _config(sequence_length),
        impl_cfg=ImplConfig(
            parallel=ParallelConfig(tp=2, ep=2, etp=1, pp=1, cp=2),
            device=f"cuda:{local_rank}",
            dtype="bfloat16",
            qat={"enabled": True, "format": "mxfp4"},
        ),
    )
    model = bundle.chunks[0].train()
    input_ids = (
        torch.arange(sequence_length, device=device, dtype=torch.long)
        % model.config.vocab_size
    )
    labels = input_ids.roll(-1)
    base_batch = PackedBatch(
        input_ids=input_ids,
        labels=labels,
        seq_lens=torch.tensor([sequence_length], device=device),
    )

    record_driver = _driver(model, "record")
    record_driver.begin()
    try:
        model.zero_grad(set_to_none=True)
        recorded_output = record_driver.wrap(_run_forward)(model, base_batch)
        recorded = RouterReplay.get_recorded_data()
        if not recorded or any(value is None for value in recorded):
            raise RuntimeError("K3 R3 record did not capture every live MoE router")
        recorded_rows = sum(
            int(value.numel()) for value in recorded if value is not None
        )
        recorded_output["loss"].backward()
    finally:
        record_driver.end()

    target = torch.tensor([0, 1], device=device, dtype=torch.long).view(1, 1, 2)
    target = target.expand(sequence_length, 1, 2).contiguous()
    routed_experts = torch.nested.as_nested_tensor([target], layout=torch.jagged)
    replay_mask = torch.ones(sequence_length, device=device, dtype=torch.bool)
    replay_mask[-1] = False
    replay_batch = PackedBatch(
        input_ids=input_ids,
        labels=labels,
        seq_lens=torch.tensor([sequence_length], device=device),
        routed_experts=routed_experts,
        r3_replay_mask=replay_mask,
    )

    replay_driver = _driver(model, "replay")
    replay_driver.begin()
    try:
        model.zero_grad(set_to_none=True)
        replayed_output = replay_driver.wrap(_run_forward)(model, replay_batch)
        replay_stats = RouterReplay.replay_stats()
        replayed_output["loss"].backward()
        if replay_stats["calls"] == 0 or replay_stats["rows"] == 0:
            raise RuntimeError(f"K3 R3 replay was inactive: {replay_stats}")
        if replay_stats["changed"] == 0:
            raise RuntimeError(f"K3 R3 replay changed no routes: {replay_stats}")
        if not torch.isfinite(replayed_output["loss"]):
            raise RuntimeError("K3 R3/QAT replay loss is not finite")
    finally:
        replay_driver.end()

    metrics = torch.tensor(
        [
            recorded_rows,
            replay_stats["calls"],
            replay_stats["rows"],
            replay_stats["changed"],
            bundle.extras["qat"]["quantized_modules"],
        ],
        device=device,
        dtype=torch.long,
    )
    minimum = metrics.clone()
    maximum = metrics.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "world_size": world_size,
            "parallel": {"tp": 2, "ep": 2, "etp": 1, "pp": 1, "cp": 2},
            "sequence_length": sequence_length,
            "recorded_elements_per_rank": [
                int(minimum[0].item()),
                int(maximum[0].item()),
            ],
            "replay_calls_per_rank": [
                int(minimum[1].item()),
                int(maximum[1].item()),
            ],
            "replay_elements_per_rank": [
                int(minimum[2].item()),
                int(maximum[2].item()),
            ],
            "changed_elements_per_rank": [
                int(minimum[3].item()),
                int(maximum[3].item()),
            ],
            "qat_modules_per_rank": [
                int(minimum[4].item()),
                int(maximum[4].item()),
            ],
            "record_loss": float(recorded_output["loss"].detach()),
            "replay_loss": float(replayed_output["loss"].detach()),
        }
        print("K3_R3_QAT_SMOKE=" + json.dumps(result, sort_keys=True), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
