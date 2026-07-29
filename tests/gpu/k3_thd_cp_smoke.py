"""Eight-rank K3 packed-THD + CP forward/backward and token-parity smoke."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from mlite_k3.config import K3Config
from mlite_k3.lite.protocol import ImplConfig, build_model


def _expected_local_tokens(
    full_tokens: torch.Tensor,
    lengths: list[int],
    *,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Independently calculate the per-sequence zigzag shard for one CP rank."""
    pieces = []
    offset = 0
    for length in lengths:
        if length % (2 * cp_size):
            raise ValueError(
                f"sequence length {length} must be divisible by 2 * cp_size={2 * cp_size}"
            )
        row = full_tokens[offset : offset + length]
        offset += length
        if cp_size == 1:
            pieces.append(row)
            continue
        chunks = row.view(2 * cp_size, length // (2 * cp_size))
        pieces.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    if offset != full_tokens.numel():
        raise ValueError(
            f"packed lengths sum to {offset}, tensor has {full_tokens.numel()} tokens"
        )
    return torch.cat(pieces)


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
    cp_size = int(os.environ.get("K3_CP_SIZE", "2"))
    tp_size = 2
    if cp_size not in (1, 2, 4) or world_size % (tp_size * cp_size):
        raise RuntimeError(
            "K3_CP_SIZE must be one of 1, 2, or 4 and divide the eight-rank "
            f"topology with TP={tp_size}; got CP={cp_size}"
        )
    ep_size = world_size // (tp_size * cp_size)

    bundle = build_model(
        _config(),
        impl_cfg=ImplConfig(
            parallel=ParallelConfig(
                tp=tp_size,
                ep=ep_size,
                etp=1,
                pp=1,
                cp=cp_size,
            ),
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
    protocol_input_ids = None

    def capture_model_input(_module, _args, kwargs) -> None:
        nonlocal protocol_input_ids
        packed_seq_params = kwargs["packed_seq_params"]
        observed["protocol_local_tokens"] = int(kwargs["input_ids"].shape[1])
        observed["local_cp_size"] = int(packed_seq_params.local_cp_size)
        protocol_input_ids = kwargs["input_ids"].detach().flatten().clone()

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
    if protocol_input_ids is None:
        raise AssertionError(
            "model input hook did not capture protocol-local token ids"
        )
    expected_input_ids = _expected_local_tokens(
        input_ids,
        lengths,
        cp_rank=bundle.parallel_state.cp_rank,
        cp_size=bundle.parallel_state.cp_size,
    )
    if not torch.equal(protocol_input_ids, expected_input_ids):
        raise AssertionError(
            "packed THD protocol changed token identity/order: "
            f"cp_rank={bundle.parallel_state.cp_rank}, "
            f"actual={protocol_input_ids.tolist()}, "
            f"expected={expected_input_ids.tolist()}"
        )
    if cp_size == 1:
        gathered_tokens = protocol_input_ids
    else:
        cp_parts = [torch.empty_like(protocol_input_ids) for _ in range(cp_size)]
        dist.all_gather(
            cp_parts,
            protocol_input_ids,
            group=bundle.parallel_state.cp_group,
        )
        gathered_tokens = torch.cat(cp_parts)
    if not torch.equal(
        torch.sort(gathered_tokens).values,
        torch.sort(input_ids).values,
    ):
        raise AssertionError(
            "packed THD CP shards must conserve every input token exactly once: "
            f"gathered={gathered_tokens.tolist()}, full={input_ids.tolist()}"
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
        "topology": {"tp": tp_size, "ep": ep_size, "cp": cp_size},
        "lengths": lengths,
        **observed,
        "token_order_parity": True,
        "token_multiset_conserved": True,
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "non_skip": True,
    }
    if rank == 0:
        print("K3_THD_CP_SMOKE=" + json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
