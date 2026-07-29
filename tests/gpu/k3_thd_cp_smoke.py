"""Eight-rank K3 packed-THD + TP2/EP2/CP2 forward/backward smoke."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from mlite_k3.config import K3Config
from mlite_k3.lite.protocol import ImplConfig, build_model


def _config() -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=64,
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


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"K3 THD+CP smoke requires 8 ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    bundle = build_model(
        _config(),
        impl_cfg=ImplConfig(
            parallel=ParallelConfig(tp=2, ep=2, etp=1, pp=1, cp=2),
            device=f"cuda:{local_rank}",
            dtype="bfloat16",
            use_thd=True,
            kda_cp_mode="headwise",
        ),
    )
    model = bundle.chunks[0].train()
    assert bundle.forward_step is not None

    lengths = [16, 24]
    total_tokens = sum(lengths)
    input_ids = torch.arange(total_tokens, device=device, dtype=torch.long) % 256
    batch = PackedBatch(
        input_ids=input_ids,
        labels=input_ids.roll(-1),
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        loss_mask=torch.ones(total_tokens, device=device, dtype=torch.float32),
    )

    observed: dict[str, int] = {}

    def capture_model_input(_module, _args, kwargs) -> None:
        packed_seq_params = kwargs["packed_seq_params"]
        observed["protocol_local_tokens"] = int(kwargs["input_ids"].shape[1])
        observed["local_cp_size"] = int(packed_seq_params.local_cp_size)

    def capture_first_layer_input(_module, args) -> None:
        observed["first_layer_sequence"] = int(args[0].shape[0])

    model_hook = model.register_forward_pre_hook(capture_model_input, with_kwargs=True)
    layer_hook = model.layers[0].register_forward_pre_hook(capture_first_layer_input)
    output = bundle.forward_step(model, batch)
    model_hook.remove()
    layer_hook.remove()

    expected_cp_tokens = total_tokens // bundle.parallel_state.cp_size
    expected_layer_sequence = expected_cp_tokens // bundle.parallel_state.tp_size
    if observed != {
        "protocol_local_tokens": expected_cp_tokens,
        "local_cp_size": bundle.parallel_state.cp_size,
        "first_layer_sequence": expected_layer_sequence,
    }:
        raise AssertionError(
            "packed THD must be CP-split once in the protocol and not again "
            f"in the model: observed={observed}"
        )

    loss = output["loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss: {loss}")
    loss.backward()
    grad_norm = torch.zeros((), device=device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm += parameter.grad.detach().float().norm()
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError(f"invalid grad norm: {grad_norm}")

    result = {
        "world_size": world_size,
        "topology": {"tp": 2, "ep": 2, "cp": 2},
        "lengths": lengths,
        **observed,
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "non_skip": True,
    }
    if rank == 0:
        print("K3_THD_CP_SMOKE=" + json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
