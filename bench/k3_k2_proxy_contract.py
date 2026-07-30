"""Frozen whole-model size contract for the two K3/K2 benchmark proxies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProxySpec:
    name: str
    num_layers: int
    num_experts: int
    k3_topk: int
    k2_topk: int
    k2_moe_intermediate_size: int
    full_attention_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    attn_res_block_size: int
    tp: int
    ep: int
    etp: int
    pp: int
    cp: int
    sequence_length: int
    warmup_steps: int
    measure_steps: int


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    activated: int


_FULL_ATTENTION_93 = tuple(range(4, 93, 4)) + (93,)
_KDA_93 = tuple(layer for layer in range(1, 94) if layer not in _FULL_ATTENTION_93)

PROXY_SPECS = {
    "reduced_layers_full_experts": ProxySpec(
        name="reduced_layers_full_experts",
        num_layers=2,
        num_experts=896,
        k3_topk=16,
        k2_topk=44,
        k2_moe_intermediate_size=1584,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        tp=1,
        ep=8,
        etp=1,
        pp=1,
        cp=1,
        sequence_length=1024,
        warmup_steps=1,
        measure_steps=3,
    ),
    "full_layers_reduced_experts": ProxySpec(
        name="full_layers_reduced_experts",
        num_layers=93,
        num_experts=16,
        k3_topk=16,
        k2_topk=16,
        k2_moe_intermediate_size=2752,
        full_attention_layers=_FULL_ATTENTION_93,
        kda_layers=_KDA_93,
        attn_res_block_size=12,
        tp=4,
        ep=1,
        etp=4,
        pp=2,
        cp=1,
        sequence_length=8,
        warmup_steps=1,
        measure_steps=2,
    ),
}

_PRESENT = {"present": True, "impact": "Measured in the production forward path."}
FEATURE_MATRIX = {
    arm: {
        "attn_res": _PRESENT,
        "kda": _PRESENT,
        "mla": _PRESENT,
        "noaux_tc_expert_bias": _PRESENT,
        "shared_experts": _PRESENT,
        "mtp": {
            "present": False,
            "impact": (
                "The current K3 production config/model has no MTP block; "
                "the benchmark therefore does not include MTP compute or memory."
            ),
        },
    }
    for arm in PROXY_SPECS
}


def _mla_parameters(
    *,
    hidden: int,
    heads: int,
    q_lora: int,
    kv_lora: int,
    qk_nope: int,
    qk_rope: int,
    value: int,
    output_gate: bool,
) -> int:
    count = (
        heads * value * hidden
        + hidden * q_lora
        + q_lora * heads * (qk_nope + qk_rope)
        + q_lora
        + hidden * (kv_lora + qk_rope)
        + kv_lora * heads * (qk_nope + value)
        + kv_lora
    )
    if output_gate:
        count += hidden * heads * value
    return count


def _kda_parameters(*, hidden: int, heads: int, head_dim: int, conv_kernel: int) -> int:
    projection = heads * head_dim
    return (
        3 * hidden * projection
        + 3 * projection * conv_kernel
        + heads
        + projection
        + hidden * head_dim
        + head_dim * projection
        + hidden * heads
        + hidden * projection
        + head_dim
        + projection * hidden
    )


def k3_parameter_counts(spec: ProxySpec) -> ParameterCounts:
    hidden = 7168
    vocab = 163840
    intermediate = 33792
    heads = 96
    q_lora = 1536
    kv_lora = 512
    qk_nope = 128
    qk_rope = 64
    value = 128
    routed_hidden = 3584
    moe_intermediate = 3072

    total = 2 * vocab * hidden + 3 * hidden
    total += spec.num_layers * 6 * hidden
    total += len(spec.full_attention_layers) * _mla_parameters(
        hidden=hidden,
        heads=heads,
        q_lora=q_lora,
        kv_lora=kv_lora,
        qk_nope=qk_nope,
        qk_rope=qk_rope,
        value=value,
        output_gate=True,
    )
    total += len(spec.kda_layers) * _kda_parameters(
        hidden=hidden,
        heads=96,
        head_dim=128,
        conv_kernel=4,
    )
    total += 3 * hidden * intermediate

    moe_layers = spec.num_layers - 1
    one_expert = 3 * routed_hidden * moe_intermediate
    one_moe_layer = (
        hidden * spec.num_experts
        + 2 * hidden * routed_hidden
        + routed_hidden
        + spec.num_experts * one_expert
        + 3 * hidden * (moe_intermediate * 2)
    )
    total += moe_layers * one_moe_layer
    activated = total - moe_layers * (spec.num_experts - spec.k3_topk) * one_expert
    return ParameterCounts(total=total, activated=activated)


def k2_parameter_counts(spec: ProxySpec) -> ParameterCounts:
    hidden = 7168
    vocab = 163840
    intermediate = 18432
    heads = 64
    q_lora = 1536
    kv_lora = 512
    qk_nope = 128
    qk_rope = 64
    value = 128
    moe_intermediate = spec.k2_moe_intermediate_size

    total = 2 * vocab * hidden + hidden
    total += spec.num_layers * (
        hidden
        + _mla_parameters(
            hidden=hidden,
            heads=heads,
            q_lora=q_lora,
            kv_lora=kv_lora,
            qk_nope=qk_nope,
            qk_rope=qk_rope,
            value=value,
            output_gate=False,
        )
    )
    total += 3 * hidden * intermediate + hidden

    moe_layers = spec.num_layers - 1
    one_expert = 3 * hidden * moe_intermediate
    one_moe_layer = (
        hidden
        + hidden * spec.num_experts
        + spec.num_experts * one_expert
        + 3 * hidden * moe_intermediate
    )
    total += moe_layers * one_moe_layer
    activated = total - moe_layers * (spec.num_experts - spec.k2_topk) * one_expert
    return ParameterCounts(total=total, activated=activated)


def relative_mismatch(left: int, right: int) -> float:
    return abs(left - right) / left


def validate_contract(spec: ProxySpec) -> dict[str, float | int]:
    k3 = k3_parameter_counts(spec)
    k2 = k2_parameter_counts(spec)
    total_mismatch = relative_mismatch(k3.total, k2.total)
    activated_mismatch = relative_mismatch(k3.activated, k2.activated)
    if total_mismatch >= 0.02 or activated_mismatch >= 0.02:
        raise RuntimeError(
            "whole-model same-size contract failed: "
            f"arm={spec.name}, k3={k3}, k2={k2}, "
            f"total_mismatch={total_mismatch}, "
            f"activated_mismatch={activated_mismatch}"
        )
    return {
        "k3_total": k3.total,
        "k3_activated": k3.activated,
        "k2_total": k2.total,
        "k2_activated": k2.activated,
        "total_relative_mismatch": total_mismatch,
        "activated_relative_mismatch": activated_mismatch,
    }
