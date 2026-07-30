"""Scheduler-only K3 reduced-checkpoint load smoke over TP/EP/PP."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite.checkpoint import (
    export_hf_weights,
    load_hf_weights as load_checkpoint,
    save_hf_weights,
)
from mlite_k3.lite.model import K3ParallelModel
from mlite_k3.lite.protocol import ImplConfig, build_model, load_hf_weights


def _config() -> K3Config:
    return K3Config(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        intermediate_size=256,
        max_position_embeddings=32,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        kda_head_dim=64,
        kda_num_heads=4,
        kda_short_conv_kernel_size=4,
        full_attention_layers=(2, 4),
        kda_layers=(1, 3),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=128,
        routed_expert_hidden_size=128,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
    )


def _quantization_config(config: K3Config) -> dict:
    return {
        "text_config": {
            "num_hidden_layers": config.num_hidden_layers,
            "first_k_dense_replace": config.first_k_dense_replace,
            "num_experts": config.num_experts,
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "format": "mxfp4-pack-quantized",
                        "targets": ["Linear"],
                        "weights": {
                            "dynamic": False,
                            "group_size": 32,
                            "num_bits": 4,
                            "scale_dtype": "torch.uint8",
                            "symmetric": True,
                            "type": "float",
                        },
                    }
                },
                "format": "mxfp4-pack-quantized",
                "ignore": [
                    r"re:.*self_attn.*",
                    r"re:.*shared_experts.*",
                    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
                    r"re:.*lm_head.*",
                    r"re:.*vision_tower.*",
                    r"re:.*mm_projector.*",
                ],
                "quant_method": "compressed-tensors",
            },
        }
    }


def _checkpoint_state(model: torch.nn.Module):
    from megatron.lite.primitive.quantization.qat import canonical_state_key

    for state_name, tensor in model.state_dict(keep_vars=True).items():
        is_qat_auxiliary = (
            ".parametrizations." in state_name and not state_name.endswith(".original")
        )
        if not is_qat_auxiliary:
            yield canonical_state_key(state_name), tensor


def _assert_exports_bitwise_equal(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    context: str,
) -> None:
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise RuntimeError(
            f"{context} changed the HF key set: "
            f"missing={missing[:1]!r}, unexpected={unexpected[:1]!r}"
        )
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if torch.equal(expected_tensor, actual_tensor):
            continue
        detail = (
            f"shape={tuple(expected_tensor.shape)}/{tuple(actual_tensor.shape)}, "
            f"dtype={expected_tensor.dtype}/{actual_tensor.dtype}"
        )
        if (
            expected_tensor.shape == actual_tensor.shape
            and expected_tensor.is_floating_point()
            and actual_tensor.is_floating_point()
        ):
            max_diff = (
                (expected_tensor.float() - actual_tensor.float()).abs().max().item()
            )
            detail += f", max_abs_diff={max_diff}"
        raise RuntimeError(f"{context} differs at {name!r}: {detail}")


@record
def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (1, 8):
        raise RuntimeError("K3 checkpoint load smoke requires one or eight ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    root = Path(os.environ["K3_LOAD_SMOKE_DIR"])
    config = _config()

    torch.manual_seed(20260729)
    reference = K3ParallelModel(config, ParallelState()).to(
        device=device,
        dtype=torch.bfloat16,
    )
    summary = save_hf_weights(
        reference,
        root,
        config,
        ParallelState(),
        target="mxfp4",
    )
    del reference
    torch.cuda.empty_cache()
    if rank == 0:
        (root / "config.json").write_text(
            json.dumps(_quantization_config(config), sort_keys=True),
            encoding="utf-8",
        )
        print(
            "K3_REDUCED_CHECKPOINT="
            + json.dumps(
                {
                    "shards": summary.shards,
                    "quantized_weights": summary.quantized_weights,
                    "plain_tensors": summary.plain_tensors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()

    single = K3ParallelModel(config, ParallelState()).to(
        device=device,
        dtype=torch.bfloat16,
    )
    load_checkpoint(single, root, config, ParallelState())
    baseline = dict(
        export_hf_weights(
            single,
            config,
            ParallelState(),
            target="bf16",
            cpu=True,
        )
    )
    from megatron.lite.primitive.quantization.qat import (
        QATSpec,
        apply_qat_to_chunks,
    )

    qat_stats = apply_qat_to_chunks(
        [single],
        QATSpec(enabled=True, format="mxfp4", ignore_patterns=()),
    )
    if qat_stats["quantized_modules"] <= 0:
        raise RuntimeError("K3 QAT smoke did not parametrize any module")
    qat_state = list(_checkpoint_state(single))
    with torch.no_grad():
        for _, tensor in qat_state:
            if tensor.is_floating_point():
                tensor.fill_(torch.nan)
    load_checkpoint(single, root, config, ParallelState())
    qat_export = dict(
        export_hf_weights(
            single,
            config,
            ParallelState(),
            target="bf16",
            cpu=True,
        )
    )
    _assert_exports_bitwise_equal(
        baseline,
        qat_export,
        context="K3 QAT export versus unparametrized export",
    )
    del qat_export, single
    torch.cuda.empty_cache()

    parallel = (
        ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
        if world_size == 1
        else ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=1)
    )
    bundle = build_model(
        config,
        impl_cfg=ImplConfig(
            parallel=parallel,
            device=f"cuda:{local_rank}",
            dtype="bfloat16",
            qat=QATSpec(enabled=True, format="mxfp4", ignore_patterns=()),
        ),
    )
    model = bundle.chunks[0]
    distributed_qat_modules = bundle.extras["qat"]["quantized_modules"]
    if distributed_qat_modules <= 0:
        raise RuntimeError("K3 distributed QAT smoke did not parametrize any module")
    checkpoint_state = list(_checkpoint_state(model))
    with torch.no_grad():
        for _, tensor in checkpoint_state:
            if tensor.is_floating_point():
                tensor.fill_(torch.nan)

    manifest = load_hf_weights(model, str(root), config, bundle.parallel_state)
    expert_bias = [
        tensor
        for name, tensor in checkpoint_state
        if name.endswith(".moe.router.expert_bias")
    ]
    if expert_bias and any(
        tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
        for tensor in expert_bias
    ):
        raise RuntimeError("K3 expert_bias was not loaded as finite FP32 state")
    gathered_export = dict(
        export_hf_weights(
            model,
            config,
            bundle.parallel_state,
            target="bf16",
            cpu=True,
        )
    )
    _assert_exports_bitwise_equal(
        baseline,
        gathered_export,
        context="K3 TP/EP/PP export versus single-rank export",
    )

    local_metrics = torch.tensor(
        [
            len(checkpoint_state),
            len(expert_bias),
            len(model.layer_indices),
            int(model.pre_process),
            int(model.post_process),
            distributed_qat_modules,
        ],
        device=device,
        dtype=torch.long,
    )
    gathered = [torch.zeros_like(local_metrics) for _ in range(world_size)]
    dist.all_gather(gathered, local_metrics)
    if rank == 0:
        print(
            "K3_CHECKPOINT_LOAD_SMOKE="
            + json.dumps(
                {
                    "world_size": world_size,
                    "parallel": {
                        name: int(getattr(parallel, name))
                        for name in ("tp", "ep", "etp", "pp", "cp")
                    },
                    "rank_metrics": [value.cpu().tolist() for value in gathered],
                    "manifest_logical_tensors": (
                        manifest.weights.quantized_weights
                        + manifest.weights.plain_tensors
                    ),
                    "expert_bias_dtype": "torch.float32",
                    # This is deliberately one coverage cell per execution output:
                    # the preceding check proves this exact persistent checkpoint
                    # field was restored as finite FP32 state after the QAT load.
                    "assertions": [
                        {
                            "cell": "router_expert_bias.load",
                            "assertion": "expert_bias_is_finite_fp32",
                        }
                    ],
                    "axes": [
                        name
                        for name in ("tp", "ep", "etp", "pp", "cp")
                        if int(getattr(parallel, name)) > 1
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
