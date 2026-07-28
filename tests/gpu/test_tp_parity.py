"""Scheduler-only K3 TP parity validation."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

print("K3_TP_IMPORT=torch", flush=True)
from megatron.lite.primitive.parallel import ParallelState  # noqa: E402
from megatron.lite.runtime.contracts import ParallelConfig  # noqa: E402

print("K3_TP_IMPORT=mlite_parallel", flush=True)
from mlite_k3.config import K3Config  # noqa: E402
from mlite_k3.lite.model import K3ParallelModel  # noqa: E402
from mlite_k3.lite.protocol import ImplConfig, build_model  # noqa: E402

print("K3_TP_IMPORT=k3_model", flush=True)


def _config() -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=32,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        kda_num_heads=2,
        kda_short_conv_kernel_size=4,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=2,
    )


def _copy_tp_shards(
    reference: K3ParallelModel,
    parallel: K3ParallelModel,
    *,
    rank: int,
    world_size: int,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in parallel.named_parameters():
            full = reference_parameters[name].detach()
            if full.shape == parameter.shape:
                parameter.copy_(full)
                continue
            differing = [
                dim
                for dim, (full_size, local_size) in enumerate(
                    zip(full.shape, parameter.shape, strict=True)
                )
                if full_size != local_size
            ]
            if (
                len(differing) != 1
                or full.shape[differing[0]]
                != parameter.shape[differing[0]] * world_size
            ):
                raise RuntimeError(
                    f"cannot infer TP placement for {name}: "
                    f"{tuple(full.shape)} -> {tuple(parameter.shape)}"
                )
            shard_dim = differing[0]
            is_gated_fc1 = "gate_up" in name or ".experts.fc1." in name
            if is_gated_fc1:
                gate, up = full.chunk(2, dim=shard_dim)
                shard = torch.cat(
                    (
                        gate.chunk(world_size, dim=shard_dim)[rank],
                        up.chunk(world_size, dim=shard_dim)[rank],
                    ),
                    dim=shard_dim,
                ).contiguous()
            else:
                shard = full.chunk(world_size, dim=shard_dim)[rank].contiguous()
            parameter.copy_(shard)


def _capture_layers(model: K3ParallelModel) -> tuple[list[torch.Tensor], list]:
    outputs: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(output[0].detach())
        )
        for layer in model.layers
    ]
    return outputs, handles


def _capture_components(
    model: K3ParallelModel,
) -> tuple[dict[str, torch.Tensor], list]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(model.layers):
        for component_name in ("self_attention", "mlp", "moe"):
            component = getattr(layer, component_name)
            if component is None:
                continue
            key = f"layers.{index}.{component_name}"
            handles.append(
                component.register_forward_hook(
                    lambda _module, _inputs, output, capture_key=key: (
                        outputs.__setitem__(capture_key, output.detach())
                    )
                )
            )
    return outputs, handles


def _gather_sequence(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    parts = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(parts, value.contiguous())
    return torch.cat(parts, dim=0)


def main() -> None:
    print("K3_TP_STAGE=before_dist_init", flush=True)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"K3_TP_STAGE=after_dist_init rank={rank}", flush=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    config = _config()

    torch.manual_seed(20260727)
    print(f"K3_TP_STAGE=before_reference_build rank={rank}", flush=True)
    reference = (
        K3ParallelModel(config, ParallelState())
        .to(device=device, dtype=torch.bfloat16)
        .train()
    )
    print(f"K3_TP_STAGE=after_reference_build rank={rank}", flush=True)
    reference_outputs, reference_handles = _capture_layers(reference)
    reference_components, reference_component_handles = _capture_components(reference)

    input_ids = torch.tensor(
        [[1, 7, 3, 9, 2, 8, 5, 4, 11, 6, 13, 12, 10, 15, 14, 16]],
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.roll(-1, dims=-1)
    with torch.no_grad():
        print(f"K3_TP_STAGE=before_reference_forward rank={rank}", flush=True)
        reference_result = reference(input_ids=input_ids, labels=labels)
        print(f"K3_TP_STAGE=after_reference_forward rank={rank}", flush=True)
    for handle in reference_handles:
        handle.remove()
    for handle in reference_component_handles:
        handle.remove()

    parallel_config = ParallelConfig(
        tp=world_size,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
    )
    print(f"K3_TP_STAGE=before_parallel_build rank={rank}", flush=True)
    bundle = build_model(
        config,
        impl_cfg=ImplConfig(
            parallel=parallel_config,
            optimizer=None,
            device="cuda",
            dtype="bfloat16",
        ),
    )
    parallel = bundle.chunks[0]
    print(f"K3_TP_STAGE=after_parallel_build rank={rank}", flush=True)
    assert isinstance(parallel, K3ParallelModel)
    _copy_tp_shards(reference, parallel, rank=rank, world_size=world_size)
    print(f"K3_TP_STAGE=after_weight_shard rank={rank}", flush=True)
    for layer in parallel.layers:
        if hasattr(layer.self_attention, "A_log"):
            assert layer.self_attention.A_log.dtype == torch.float32
            assert layer.self_attention.dt_bias.dtype == torch.float32

    parallel_outputs, parallel_handles = _capture_layers(parallel)
    parallel_components, parallel_component_handles = _capture_components(parallel)
    print(f"K3_TP_STAGE=before_parallel_forward rank={rank}", flush=True)
    parallel_result = parallel(input_ids=input_ids, labels=labels)
    print(f"K3_TP_STAGE=before_parallel_backward rank={rank}", flush=True)
    parallel_result["loss"].backward()
    print(f"K3_TP_STAGE=after_parallel_backward rank={rank}", flush=True)
    for handle in parallel_handles:
        handle.remove()
    for handle in parallel_component_handles:
        handle.remove()

    layer_metrics = []
    for index, (expected, local_actual) in enumerate(
        zip(reference_outputs, parallel_outputs, strict=True)
    ):
        actual = _gather_sequence(local_actual, world_size)
        difference = (actual.float() - expected.float()).abs()
        layer_metrics.append(
            {
                "layer": index,
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "cosine": torch.nn.functional.cosine_similarity(
                    actual.float().flatten(),
                    expected.float().flatten(),
                    dim=0,
                ).item(),
            }
        )

    logits_difference = (
        parallel_result["logits"].float() - reference_result["logits"].float()
    ).abs()
    component_metrics = {}
    for name, expected in reference_components.items():
        actual = _gather_sequence(parallel_components[name], world_size)
        difference = (actual.float() - expected.float()).abs()
        component_metrics[name] = {
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                actual.float().flatten(),
                expected.float().flatten(),
                dim=0,
            ).item(),
        }
    result = {
        "rank": rank,
        "world_size": world_size,
        "parallel": {
            "tp": world_size,
            "ep": 1,
            "etp": 1,
            "pp": 1,
            "cp": 1,
        },
        "validated_scope": bundle.extras["validated_scope"],
        "layers": layer_metrics,
        "components": component_metrics,
        "logits_max_abs": logits_difference.max().item(),
        "logits_mean_abs": logits_difference.mean().item(),
        "loss_reference": reference_result["loss"].item(),
        "loss_parallel": parallel_result["loss"].item(),
        "loss_abs": abs(
            parallel_result["loss"].item() - reference_result["loss"].item()
        ),
    }
    if rank == 0:
        print("K3_TP_PARITY=" + json.dumps(result, sort_keys=True), flush=True)

    max_layer_error = max(metric["max_abs"] for metric in layer_metrics)
    if max_layer_error > 0.08 or result["logits_max_abs"] > 0.12:
        raise AssertionError(f"TP parity exceeded bf16 tolerance: {result}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
