"""Two-axis whole-model-matched Kimi-K3/Kimi-K2 training benchmark."""

from __future__ import annotations

import gc
import json
import os
import re
import statistics
import time
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.distributed as dist

from k3_k2_proxy_contract import (
    FEATURE_MATRIX,
    PROXY_SPECS,
    ProxySpec,
    validate_contract,
)
from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.model.kimi_k2.lite.model import KimiK2Model
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
from megatron.lite.runtime.contracts import ParallelConfig
from mlite_k3.config import K3Config
from mlite_k3.lite.model import K3ParallelModel


SEED = 20260729
LEARNING_RATE = 1.0e-4


def k3_config(spec: ProxySpec) -> K3Config:
    return K3Config(
        num_hidden_layers=spec.num_layers,
        num_experts=spec.num_experts,
        num_experts_per_token=spec.k3_topk,
        full_attention_layers=spec.full_attention_layers,
        kda_layers=spec.kda_layers,
        attn_res_block_size=spec.attn_res_block_size,
        max_position_embeddings=spec.sequence_length,
    )


def k2_config(spec: ProxySpec) -> KimiK2Config:
    return KimiK2Config(
        num_hidden_layers=spec.num_layers,
        n_routed_experts=spec.num_experts,
        num_experts_per_tok=spec.k2_topk,
        moe_intermediate_size=spec.k2_moe_intermediate_size,
        max_position_embeddings=spec.sequence_length,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
    )


def parallel_config(spec: ProxySpec) -> ParallelConfig:
    return ParallelConfig(
        tp=spec.tp,
        ep=spec.ep,
        etp=spec.etp,
        pp=spec.pp,
        cp=spec.cp,
    )


def k2_train_config(spec: ProxySpec) -> SimpleNamespace:
    return SimpleNamespace(
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=False,
        tp=spec.tp,
        ep=spec.ep,
        etp=spec.etp,
        pp=spec.pp,
        cp=spec.cp,
    )


def build(name: str, spec: ProxySpec, ps, device: torch.device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if name == "k3":
        model = K3ParallelModel(
            k3_config(spec),
            ps,
            deterministic=False,
            kda_cp_mode="headwise",
        )
    elif name == "k2":
        model = KimiK2Model(
            k2_config(spec),
            k2_train_config(spec),
            ps,
            attention_backend_override="flash",
        )
    else:
        raise ValueError(name)
    return model.to(device=device, dtype=torch.bfloat16).train()


def input_batch(spec: ProxySpec, device: torch.device) -> SimpleNamespace:
    input_ids = (
        torch.arange(spec.sequence_length, device=device, dtype=torch.long).view(1, -1)
        % 163840
    )
    return SimpleNamespace(input_ids=input_ids, labels=input_ids.roll(-1, dims=-1))


def forward_step(model, batch):
    return model(input_ids=batch.input_ids, labels=batch.labels)


def forward_backward(model, batch, spec: ProxySpec, ps) -> None:
    if ps.pp_size == 1:
        output = forward_step(model, batch)
        output["loss"].backward()
        return
    forward_backward_pipelining(
        forward_step,
        [model],
        iter([batch]),
        SimpleNamespace(num_microbatches=1),
        ps,
        tensor_shape=(spec.sequence_length, 1, 7168),
    )


def maximum_across_ranks(value: float | int, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def parameter_numel_across_ranks(
    model: torch.nn.Module, device: torch.device
) -> dict[str, int]:
    local_numel = sum(parameter.numel() for parameter in model.parameters())
    values: dict[str, int] = {"local": local_numel}
    for label, op in (
        ("local_min", dist.ReduceOp.MIN),
        ("local_max", dist.ReduceOp.MAX),
        ("summed_rank_local", dist.ReduceOp.SUM),
    ):
        tensor = torch.tensor(local_numel, device=device, dtype=torch.int64)
        dist.all_reduce(tensor, op=op)
        values[label] = int(tensor.item())
    return values


def _canonical_parameter_name(model: torch.nn.Module, name: str) -> str:
    match = re.match(r"^layers\.(\d+)(\..+)$", name)
    if match is None:
        return name
    local_index = int(match.group(1))
    global_index = model.layer_indices[local_index]
    return f"layers.{global_index}{match.group(2)}"


def _logical_parameter_multiplier(
    model: torch.nn.Module,
    name: str,
    ps,
) -> int:
    modules = dict(model.named_modules())
    parts = name.split(".")
    for end in range(len(parts) - 1, 0, -1):
        path = ".".join(parts[:end])
        module = modules.get(path)
        if module is None:
            continue
        relative_name = ".".join(parts[end:])
        class_name = type(module).__name__
        if class_name in {"Experts", "K3LatentExperts"} and relative_name.startswith(
            ("fc1.", "fc2.")
        ):
            return ps.ep_size * ps.etp_size
        if class_name in {
            "ColumnParallelLinear",
            "RowParallelLinear",
            "VanillaColumnParallelLinear",
            "VocabParallelEmbedding",
            "VocabParallelOutput",
            "_LocalLinear",
        } and relative_name in {
            "weight",
            "linear.weight",
            "embedding.weight",
            "col.linear.weight",
        }:
            return ps.tp_size
        if class_name in {"K3FullRankGatedDeltaNet", "_K3FullRankDeltaNet"}:
            if relative_name in {
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "A_log",
                "dt_bias",
            }:
                return ps.tp_size
    return 1


def global_unique_parameter_numel(
    model: torch.nn.Module,
    ps,
) -> dict[str, int]:
    local_records = []
    for name, parameter in model.named_parameters():
        multiplier = _logical_parameter_multiplier(model, name, ps)
        local_records.append(
            (
                _canonical_parameter_name(model, name),
                parameter.numel(),
                multiplier,
                parameter.numel() * multiplier,
            )
        )
    gathered: list[list[tuple[str, int, int, int]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_records)
    logical_by_name: dict[str, int] = {}
    for rank_records in gathered:
        assert rank_records is not None
        for name, _local_numel, _multiplier, logical_numel in rank_records:
            previous = logical_by_name.setdefault(name, logical_numel)
            if previous != logical_numel:
                raise AssertionError(
                    "inconsistent reconstructed global numel for "
                    f"{name}: {previous} != {logical_numel}"
                )
    return {
        "global_unique": sum(logical_by_name.values()),
        "canonical_parameter_keys": len(logical_by_name),
    }


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def memory_snapshot(device: torch.device) -> dict[str, int]:
    """Capture allocator occupancy and cache/fragmentation headroom together."""
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "scratch_bytes": reserved - allocated,
    }


def profile_step(
    name: str,
    model: torch.nn.Module,
    batch,
    spec: ProxySpec,
    ps,
    device: torch.device,
) -> None:
    """Emit raw Kineto events for a separate, explicitly enabled diagnosis run."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        model.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        forward_backward(model, batch, spec, ps)
        torch.cuda.synchronize()

    local_events = [
        {
            "key": event.key,
            "count": event.count,
            "cpu_time_total_us": event.cpu_time_total,
            "device_time_total_us": event.device_time_total,
            "self_device_time_total_us": event.self_device_time_total,
        }
        for event in profiler.key_averages()
        if event.device_time_total > 0
    ]
    local_events.sort(
        key=lambda event: event["device_time_total_us"],
        reverse=True,
    )
    gathered: list[list[dict] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_events[:200])
    if dist.get_rank() == 0:
        print(
            "PROFILE_RAW="
            + json.dumps(
                {
                    "name": name,
                    "rank_events": gathered,
                    "units": "microseconds",
                },
                sort_keys=True,
            ),
            flush=True,
        )


def benchmark(
    name: str,
    spec: ProxySpec,
    ps,
    device: torch.device,
) -> dict[str, float | int | str | dict]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_model = torch.cuda.memory_allocated(device)
    model = build(name, spec, ps, device)
    parameter_numel = parameter_numel_across_ranks(model, device)
    global_unique_numel = global_unique_parameter_numel(model, ps)
    if dist.get_rank() == 0:
        print(
            "BUILD_NUMEL="
            + json.dumps(
                {
                    "name": name,
                    "placement_weighted": parameter_numel,
                    **global_unique_numel,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    model_storage = torch.cuda.memory_allocated(device) - before_model
    batch = input_batch(spec, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    def step(
        *, measure_memory: bool
    ) -> tuple[float, int, int, dict[str, int], dict[str, int]]:
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        if measure_memory:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        forward_backward(model, batch, spec, ps)
        torch.cuda.synchronize()
        train_peak = torch.cuda.max_memory_allocated(device)
        train_snapshot = memory_snapshot(device)
        if measure_memory:
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_peak = torch.cuda.max_memory_allocated(device)
        optimizer_snapshot = memory_snapshot(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            elapsed_ms,
            train_peak,
            optimizer_peak,
            train_snapshot,
            optimizer_snapshot,
        )

    for _ in range(spec.warmup_steps):
        step(measure_memory=False)

    if os.environ.get("K3_PROFILE") == "1":
        profile_step(name, model, batch, spec, ps, device)

    elapsed: list[float] = []
    train_peaks: list[int] = []
    optimizer_peaks: list[int] = []
    train_snapshots: list[dict[str, int]] = []
    optimizer_snapshots: list[dict[str, int]] = []
    for _ in range(spec.measure_steps):
        (
            elapsed_ms,
            train_peak,
            optimizer_peak,
            train_snapshot,
            optimizer_snapshot,
        ) = step(measure_memory=True)
        elapsed.append(maximum_across_ranks(elapsed_ms, device))
        train_peaks.append(int(maximum_across_ranks(train_peak, device)))
        optimizer_peaks.append(int(maximum_across_ranks(optimizer_peak, device)))
        train_snapshots.append(
            {
                key: int(maximum_across_ranks(value, device))
                for key, value in train_snapshot.items()
            }
        )
        optimizer_snapshots.append(
            {
                key: int(maximum_across_ranks(value, device))
                for key, value in optimizer_snapshot.items()
            }
        )

    step_ms = statistics.median(elapsed)
    global_tokens = spec.sequence_length * ps.dp_size
    steady_snapshot = {
        key: int(maximum_across_ranks(value, device))
        for key, value in memory_snapshot(device).items()
    }
    result: dict[str, float | int | str | dict] = {
        "name": name,
        "parallel": {
            "tp": spec.tp,
            "ep": spec.ep,
            "etp": spec.etp,
            "pp": spec.pp,
            "cp": spec.cp,
            "dp": ps.dp_size,
        },
        "sequence_length": spec.sequence_length,
        "warmup_steps": spec.warmup_steps,
        "measure_steps": spec.measure_steps,
        "global_unique_parameter_numel": global_unique_numel,
        "parameter_numel": parameter_numel,
        "step_ms_median_slowest_rank": step_ms,
        "step_ms_samples_slowest_rank": elapsed,
        "tokens_per_second_per_gpu": (
            global_tokens / (step_ms / 1000.0) / dist.get_world_size()
        ),
        "model_storage_bytes_peak_rank": int(
            maximum_across_ranks(model_storage, device)
        ),
        "training_forward_backward_peak_bytes": max(train_peaks),
        "optimizer_step_peak_bytes": max(optimizer_peaks),
        "optimizer_state_bytes_peak_rank": int(
            maximum_across_ranks(optimizer_state_bytes(optimizer), device)
        ),
        "training_memory_snapshots_peak_rank": train_snapshots,
        "optimizer_memory_snapshots_peak_rank": optimizer_snapshots,
        "steady_memory_snapshot_peak_rank": steady_snapshot,
        "scratch_definition": "reserved_bytes - allocated_bytes at the same sync point",
        "steady_allocated_bytes_peak_rank": steady_snapshot["allocated_bytes"],
        "reserved_bytes_peak_rank": int(
            maximum_across_ranks(torch.cuda.max_memory_reserved(device), device)
        ),
    }
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def init_wandb(spec: ProxySpec, contract: dict):
    if dist.get_rank() != 0:
        return None
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "k3-k2-proxy-benchmark"),
        name=f"k3-k2-{spec.name}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        mode="online",
        config={
            "proxy": asdict(spec),
            "whole_model_size_contract": contract,
            "features": FEATURE_MATRIX[spec.name],
            "optimizer": "SGD",
            "dtype": "bfloat16",
            "k3_source_commit": os.environ["K3_SOURCE_COMMIT"],
            "mlite_source_commit": os.environ["MLITE_SOURCE_COMMIT"],
            "harness_commit": os.environ["HARNESS_COMMIT"],
        },
    )
    print(f"WANDB_URL={run.url}", flush=True)
    return run


def main() -> None:
    arm = os.environ["K3_PROXY_ARM"]
    spec = PROXY_SPECS[arm]
    contract = validate_contract(spec)
    if spec.num_layers != len(spec.full_attention_layers) + len(spec.kda_layers):
        raise AssertionError("K3 attention schedule does not cover every proxy layer")
    if dist.is_initialized():
        raise RuntimeError("distributed process group must not be initialized early")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if dist.get_world_size() != 8:
        raise RuntimeError("K3/K2 proxy benchmark requires exactly eight ranks")

    ps = init_parallel(parallel_config(spec))
    wandb_run = init_wandb(spec, contract)
    results = [benchmark(name, spec, ps, device) for name in ("k3", "k2")]
    k3_result, k2_result = results
    k3_global_unique = int(k3_result["global_unique_parameter_numel"]["global_unique"])
    k2_global_unique = int(k2_result["global_unique_parameter_numel"]["global_unique"])
    global_unique_cross_check = {
        "k3_build": k3_global_unique,
        "k3_formula": int(contract["k3_total"]),
        "k3_relative_mismatch": abs(k3_global_unique - int(contract["k3_total"]))
        / int(contract["k3_total"]),
        "k2_build": k2_global_unique,
        "k2_formula": int(contract["k2_total"]),
        "k2_relative_mismatch": abs(k2_global_unique - int(contract["k2_total"]))
        / int(contract["k2_total"]),
        "limit": 0.02,
    }
    if rank == 0:
        print(
            "GLOBAL_UNIQUE_NUMEL_CONTRACT="
            + json.dumps(global_unique_cross_check, sort_keys=True),
            flush=True,
        )
    if (
        global_unique_cross_check["k3_relative_mismatch"] >= 0.02
        or global_unique_cross_check["k2_relative_mismatch"] >= 0.02
    ):
        raise AssertionError(
            "reconstructed global-unique numel disagrees with formula by >=2%"
        )
    k3_realized_numel = int(k3_result["parameter_numel"]["summed_rank_local"])
    k2_realized_numel = int(k2_result["parameter_numel"]["summed_rank_local"])
    realized_numel_mismatch = abs(k3_realized_numel - k2_realized_numel) / max(
        k3_realized_numel, k2_realized_numel
    )
    built_numel_cross_check = {
        "scope": "sum_of_rank_local_parameter_numel_under_identical_parallel_layout",
        "k3": k3_realized_numel,
        "k2": k2_realized_numel,
        "relative_mismatch": realized_numel_mismatch,
        "is_whole_model_size_gate": False,
        "interpretation": (
            "Placement-weighted count includes TP/EP/PP replication and sharding; "
            "the whole-model architecture formula remains the same-size gate."
        ),
    }
    if rank == 0:
        print(
            "BUILT_NUMEL_CROSS_CHECK="
            + json.dumps(built_numel_cross_check, sort_keys=True),
            flush=True,
        )
    speed_ratio = float(k2_result["step_ms_median_slowest_rank"]) / float(
        k3_result["step_ms_median_slowest_rank"]
    )
    memory_ratio = float(k3_result["training_forward_backward_peak_bytes"]) / float(
        k2_result["training_forward_backward_peak_bytes"]
    )
    payload = {
        "arm": arm,
        "source": {
            "k3": os.environ["K3_SOURCE_COMMIT"],
            "mlite": os.environ["MLITE_SOURCE_COMMIT"],
            "harness": os.environ["HARNESS_COMMIT"],
        },
        "contract": {
            "count_scope": "whole_model_architecture_not_parallel_shard",
            **contract,
            "global_unique_build_cross_check": global_unique_cross_check,
            "built_model_cross_check": built_numel_cross_check,
            "dtype": "bfloat16",
            "optimizer": "SGD",
        },
        "features": FEATURE_MATRIX[arm],
        "results": results,
        "comparison": {
            "k3_over_k2_step_speedup": speed_ratio,
            "k3_over_k2_training_peak_ratio": memory_ratio,
            "k3_not_slower": speed_ratio >= 1.0,
            "k3_not_more_training_memory": memory_ratio <= 1.0,
        },
    }
    if rank == 0:
        print("K3_K2_PROXY_BENCH=" + json.dumps(payload, sort_keys=True), flush=True)
        assert wandb_run is not None
        wandb_run.log(
            {
                "k3/step_ms": k3_result["step_ms_median_slowest_rank"],
                "k2/step_ms": k2_result["step_ms_median_slowest_rank"],
                "k3/tokens_per_second_per_gpu": k3_result["tokens_per_second_per_gpu"],
                "k2/tokens_per_second_per_gpu": k2_result["tokens_per_second_per_gpu"],
                "k3/training_peak_bytes": k3_result[
                    "training_forward_backward_peak_bytes"
                ],
                "k2/training_peak_bytes": k2_result[
                    "training_forward_backward_peak_bytes"
                ],
                "k3/steady_allocated_bytes": k3_result[
                    "steady_memory_snapshot_peak_rank"
                ]["allocated_bytes"],
                "k3/steady_reserved_bytes": k3_result[
                    "steady_memory_snapshot_peak_rank"
                ]["reserved_bytes"],
                "k3/steady_scratch_bytes": k3_result[
                    "steady_memory_snapshot_peak_rank"
                ]["scratch_bytes"],
                "k2/steady_allocated_bytes": k2_result[
                    "steady_memory_snapshot_peak_rank"
                ]["allocated_bytes"],
                "k2/steady_reserved_bytes": k2_result[
                    "steady_memory_snapshot_peak_rank"
                ]["reserved_bytes"],
                "k2/steady_scratch_bytes": k2_result[
                    "steady_memory_snapshot_peak_rank"
                ]["scratch_bytes"],
                "comparison/k3_over_k2_step_speedup": speed_ratio,
                "comparison/k3_over_k2_training_peak_ratio": memory_ratio,
            }
        )
        wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
