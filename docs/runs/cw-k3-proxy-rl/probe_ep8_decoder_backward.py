"""Isolate production K3 decoder blocks on EP8."""

from __future__ import annotations

import datetime
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.model.protocol_utils import pack_thd_forward_kwargs
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.backends.mlite.runtime import _apply_attention_backend_env
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from mlite_k3.lite.model import K3ParallelDecoderLayer
from mlite_k3.lite.protocol import _warmup_ep_collective, build_model_config


def mark(rank: int, phase: str, **details: int) -> None:
    print(
        "K3_EP8_DECODER_PHASE="
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


def backward_entry(rank: int, layer: int, grad: torch.Tensor) -> torch.Tensor:
    mark(rank, "backward_layer_enter", layer=layer)
    return grad


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("K3 EP8 decoder probe requires exactly 8 ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    ps = init_parallel(ParallelConfig(tp=1, ep=8, etp=1, pp=1, cp=1))
    _warmup_ep_collective(ps, device=str(device))
    attention_backend = os.environ.get("ATTENTION_BACKEND_OVERRIDE", "auto")
    _apply_attention_backend_env(attention_backend, tag="k3-ep8-decoder-probe")
    config = build_model_config(os.environ["MODEL_PATH"])
    layers = int(os.environ.get("DECODER_LAYERS", "4"))
    tokens = int(os.environ.get("TOKENS", "3072"))
    num_seqs = int(os.environ.get("NUM_SEQS", "16"))
    if not 1 <= layers <= 4:
        raise RuntimeError(f"DECODER_LAYERS must be in [1, 4], got {layers}")
    if not 1 <= tokens <= 4096:
        raise RuntimeError(f"TOKENS must be in [1, 4096], got {tokens}")
    if not 1 <= num_seqs <= tokens or tokens % num_seqs:
        raise RuntimeError(
            "NUM_SEQS must divide TOKENS and be in [1, TOKENS]; "
            f"got NUM_SEQS={num_seqs}, TOKENS={tokens}"
        )
    layer_types = [config.attention_type(layer) for layer in range(layers)]
    if "mla" not in layer_types:
        raise RuntimeError(
            "K3 EP8 decoder probe must include an MLA layer; "
            f"got layer_types={layer_types}"
        )

    mark(rank, "build_start")
    modules = torch.nn.ModuleList(
        [
            K3ParallelDecoderLayer(
                config,
                layer,
                ps,
                use_thd=True,
                use_deepep=False,
                deterministic=False,
                kda_cp_mode="headwise",
            )
            for layer in range(layers)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    mark(rank, "build_done")
    seq_lens = torch.full(
        (num_seqs,), tokens // num_seqs, dtype=torch.int32, device=device
    )
    token_ids = torch.zeros(tokens, dtype=torch.long, device=device)
    packed_kwargs = pack_thd_forward_kwargs(
        SimpleNamespace(ps=ps),
        PackedBatch(
            input_ids=token_ids,
            labels=token_ids.clone(),
            seq_lens=seq_lens,
            loss_mask=torch.ones(tokens, dtype=torch.float32, device=device),
        ),
    )
    packed_seq_params = packed_kwargs["packed_seq_params"]
    packed_tokens = packed_kwargs["input_ids"].numel()
    if packed_tokens != tokens:
        raise RuntimeError(
            f"packed token count changed from {tokens} to {packed_tokens}"
        )
    hidden = torch.randn(
        packed_tokens,
        1,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    block_residual = hidden.new_zeros(packed_tokens, 0, config.hidden_size)

    mark(rank, "forward_start")
    for layer, module in enumerate(modules):
        hidden, block_residual = module(
            hidden,
            block_residual,
            packed_seq_params=packed_seq_params,
        )
        hidden.register_hook(
            lambda grad, layer=layer: backward_entry(rank, layer, grad)
        )
        mark(rank, "forward_layer_done", layer=layer)
    mark(rank, "forward_done")
    mark(rank, "backward_start")
    hidden.float().square().mean().backward()
    mark(rank, "backward_done")

    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in modules.parameters()
    )
    if not finite:
        raise RuntimeError("K3 EP8 decoder probe found non-finite gradients")
    if rank == 0:
        print(
            "K3_EP8_DECODER_BACKWARD_OK="
            + json.dumps(
                {
                    "layers": layers,
                    "layer_types": layer_types,
                    "world_size": dist.get_world_size(),
                    "finite_grads": finite,
                    "tokens": tokens,
                    "num_seqs": num_seqs,
                    "attention_backend": attention_backend,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
