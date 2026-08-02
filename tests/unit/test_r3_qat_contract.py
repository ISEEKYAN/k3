from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from mlite_k3.config import K3Config
from mlite_k3.lite import protocol
from mlite_k3.lite.loss_layout import prepare_labels_and_loss_mask
from mlite_k3.lite.protocol import ImplConfig, build_model


def _tiny_config() -> K3Config:
    return K3Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        kda_head_dim=8,
        kda_num_heads=4,
        kda_short_conv_kernel_size=2,
        full_attention_layers=(2,),
        kda_layers=(1,),
        attn_res_block_size=2,
        first_k_dense_replace=1,
        moe_intermediate_size=32,
        routed_expert_hidden_size=32,
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=2,
    )


def test_protocol_exports_shared_zigzag_r3_contract():
    from megatron.lite.model import protocol_utils

    assert protocol.pack_routed_experts is protocol_utils.pack_routed_experts
    assert protocol.pack_r3_replay_mask is protocol_utils.pack_r3_replay_mask
    assert (
        protocol.unpack_thd_forward_output is protocol_utils.unpack_thd_forward_output
    )


def test_protocol_exposes_verl_thd_output_unpack_hook(monkeypatch):
    model = object()
    batch = object()
    output = object()
    expected = object()
    calls = []

    def unpack(model_arg, batch_arg, output_arg):
        calls.append((model_arg, batch_arg, output_arg))
        return expected

    monkeypatch.setattr(protocol, "unpack_thd_forward_output", unpack)

    assert protocol.unpack_forward_output(model, batch, output) is expected
    assert calls == [(model, batch, output)]


def test_k3_r3_selects_only_live_moe_layers_after_dense_prefix():
    model = SimpleNamespace(config=_tiny_config())
    routed = torch.arange(3 * 2 * 1).reshape(3, 2, 1)

    selected = protocol.select_routed_experts(model, routed)

    assert selected.shape == (3, 1, 1)
    torch.testing.assert_close(selected, routed[:, 1:, :])


def test_k3_r3_fails_loudly_when_transformer_config_lost_dense_prefix():
    model = SimpleNamespace(config=SimpleNamespace())
    routed = torch.arange(3 * 3 * 1).reshape(3, 3, 1)

    with pytest.raises(AttributeError, match="first_k_dense_replace"):
        protocol.select_routed_experts(model, routed)


def test_k3_r3_dense_prefix_selection_preserves_jagged_sequences():
    model = SimpleNamespace(config=_tiny_config())
    rows = [
        torch.arange(3 * 2 * 1).reshape(3, 2, 1),
        torch.arange(2 * 2 * 1).reshape(2, 2, 1),
    ]
    routed = torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    selected = protocol.select_routed_experts(model, routed)

    assert selected.is_nested
    selected_rows = list(selected.unbind(0))
    for selected_row, row in zip(selected_rows, rows, strict=True):
        torch.testing.assert_close(selected_row, row[:, 1:, :])


def test_k3_dist_opt_uses_shared_training_optimizer_primitive(monkeypatch):
    calls = []
    expected_optimizer = object()
    expected_finalize = object()

    def build(chunks, **kwargs):
        calls.append((chunks, kwargs))
        return expected_optimizer, expected_finalize

    monkeypatch.setattr(
        "megatron.lite.primitive.optimizers.megatron_wrap."
        "build_dist_opt_training_optimizer",
        build,
    )
    chunks = [object()]
    model_cfg = _tiny_config()
    impl_cfg = ImplConfig(
        optimizer="dist_opt",
        optimizer_config=SimpleNamespace(lr=1e-5),
        grad_reduce_in_fp32=False,
    )
    ps = object()

    optimizer, finalize = protocol._build_dist_opt_optimizer(
        chunks, model_cfg, impl_cfg, ps
    )

    assert (optimizer, finalize) == (expected_optimizer, expected_finalize)
    assert calls == [
        (
            chunks,
            {
                "model_cfg": model_cfg,
                "impl_cfg": impl_cfg,
                "ps": ps,
                "model_name": "k3",
                "is_expert": protocol.is_expert_param,
                "deterministic": False,
                "grad_reduce_in_fp32": False,
            },
        )
    ]


def test_k3_warms_ep_collective_before_model_allocation(monkeypatch):
    calls = []
    group = object()

    def all_to_all_single(output, input_, *, group):
        calls.append((output.numel(), input_.numel(), group))
        output.copy_(input_)

    monkeypatch.setattr(protocol.dist, "all_to_all_single", all_to_all_single)

    protocol._warmup_ep_collective(
        SimpleNamespace(ep_size=8, ep_group=group), device="cpu"
    )

    assert calls == [(8, 8, group)]


def test_k3_skips_ep_collective_warmup_for_ep1(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("EP1 must not issue all-to-all")

    monkeypatch.setattr(protocol.dist, "all_to_all_single", unexpected)

    protocol._warmup_ep_collective(
        SimpleNamespace(ep_size=1, ep_group=None), device="cpu"
    )


@pytest.mark.parametrize(
    ("name", "expert", "expected"),
    (
        (
            "layers.1.moe.experts.fc1.weight0",
            True,
            (Replicate, Replicate, Shard, Shard),
        ),
        (
            "layers.1.moe.experts.fc2.weight0",
            True,
            (Replicate, Replicate, Shard, Shard),
        ),
        (
            "layers.1.moe.shared_experts.gate_up.linear.weight",
            False,
            (Replicate, Replicate, Replicate, Shard),
        ),
        (
            "layers.1.self_attention.o_proj.linear.weight",
            False,
            (Replicate, Replicate, Replicate, Shard),
        ),
    ),
)
def test_dist_opt_checkpoint_placement_matches_k3_parallel_layout(
    name, expert, expected
):
    placements = protocol.PLACEMENT_FN(name)

    assert protocol.is_expert_param(name) is expert
    assert tuple(type(placement) for placement in placements) == expected
    if name.endswith("fc1.weight0"):
        assert placements[2].dim == 0
        assert placements[3].dim == 0
    elif name.endswith("fc2.weight0"):
        assert placements[2].dim == 0
        assert placements[3].dim == 1
    elif "gate_up" in name:
        assert placements[3].dim == 0
    elif "o_proj" in name:
        assert placements[3].dim == 1


def test_unknown_optimizer_fails_before_model_initialization():
    with pytest.raises(ValueError, match="Unknown K3 lite optimizer"):
        build_model(
            _tiny_config(),
            impl_cfg=ImplConfig(
                device="cpu",
                dtype="float32",
                optimizer="not-an-optimizer",
            ),
        )


def test_fused_router_is_rejected_before_model_initialization():
    with pytest.raises(ValueError, match="moe_router_fusion=False"):
        build_model(
            _tiny_config(),
            impl_cfg=ImplConfig(
                device="cpu",
                dtype="float32",
                moe_router_fusion=True,
            ),
        )


def test_k3_replay_roots_are_owned_by_k3_decoder_topology():
    layer_a, layer_b = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    chunk = SimpleNamespace(layers=torch.nn.ModuleList([layer_a, layer_b]))

    assert protocol.router_replay_roots(chunk) == [layer_a, layer_b]
    fallback = torch.nn.Linear(1, 1)
    assert protocol.router_replay_roots(fallback) == [fallback]


def test_k3_parallel_kda_imports_against_latest_mlite():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src/mlite_k3/primitive/kda_parallel.py"
    ).read_text()

    assert "FullRankGatedDeltaNet," not in source
    assert "class _K3FullRankDeltaNet(nn.Module):" in source
    assert "GatedDeltaNet._headwise_cp2hp" in source
    model_source = (
        Path(__file__).parents[2] / "src/mlite_k3/lite/model.py"
    ).read_text()
    assert "from mlite_k3.primitive.mla import K3MultiLatentAttention" in model_source
    assert "self.self_attention = K3MultiLatentAttention(" in model_source
    assert "from mlite_k3.primitive.experts import K3LatentExperts" in model_source
    assert "self.experts = K3LatentExperts(" in model_source


def test_latest_mlite_imports_the_distributed_k3_model():
    pytest.importorskip("transformer_engine")

    from mlite_k3.lite.model import K3ParallelModel
    from mlite_k3.primitive.kda_parallel import K3FullRankGatedDeltaNet

    assert K3ParallelModel.__name__ == "K3ParallelModel"
    assert K3FullRankGatedDeltaNet.__name__ == "K3FullRankGatedDeltaNet"


def test_kda_restores_activation_dtype_before_output_projection():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src/mlite_k3/primitive/kda_parallel.py"
    ).read_text()

    assert "def _output_projection(" in source
    assert source.count("return self._output_projection(output, x)") == 2


def test_mxfp4_qat_only_parametrizes_routed_expert_linears():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(
            device="cpu",
            dtype="float32",
            qat={"enabled": True, "format": "mxfp4", "ignore_patterns": ()},
        ),
    )
    names = {
        name
        for name, _ in bundle.chunks[0].named_parameters()
        if name.endswith(".parametrizations.weight.original")
    }

    assert names
    assert all(
        any(
            component in name.split(".")
            for component in (
                "routed_expert_down_proj",
                "experts",
                "routed_expert_up_proj",
            )
        )
        for name in names
    ), sorted(names)
    assert not any("shared_experts" in name for name in names)
    assert not any("self_attention" in name for name in names)
    assert not any(".mlp." in name for name in names)
    assert bundle.extras["qat"]["quantized_modules"] == len(names)


def test_disabled_qat_is_inert():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(device="cpu", dtype="float32"),
    )

    assert not any(
        "parametrizations" in name for name, _ in bundle.chunks[0].named_parameters()
    )
    assert bundle.extras["qat"]["quantized_modules"] == 0


def test_unproven_ep_axis_is_not_reported_as_validated():
    dimensions = {"tp": 1, "ep": 2, "etp": 1, "pp": 1, "cp": 1}

    axes, evidence = protocol._resolve_validated_axes(
        dimensions,
        use_thd=False,
    )

    assert axes == ()
    assert evidence == {}


def test_parallel_axis_evidence_is_explicit_and_traceable():
    dimensions = {"tp": 1, "ep": 2, "etp": 1, "pp": 1, "cp": 1}
    source = f"job:12345:assertion:ep_parallel_contract#sha256:{'a' * 64}"

    axes, evidence = protocol._resolve_validated_axes(
        dimensions,
        use_thd=False,
        validation_evidence={"ep": (source,)},
    )

    assert axes == ("ep",)
    assert evidence == {"ep": (source,)}


def test_parallel_axis_rejects_unfingerprinted_source():
    dimensions = {"tp": 1, "ep": 2, "etp": 1, "pp": 1, "cp": 1}

    with pytest.raises(RuntimeError, match="invalid K3 validation evidence"):
        protocol._resolve_validated_axes(
            dimensions,
            use_thd=False,
            validation_evidence={"ep": ("job:12345",)},
        )


def test_build_model_passes_execution_evidence_to_axis_resolver(monkeypatch):
    supplied = {"tp": ("test:tests/gpu/test_tp_parity.py::test_tp2",)}
    seen = []

    def resolve(dimensions, *, use_thd, validation_evidence=None):
        seen.append((dimensions, use_thd, validation_evidence))
        return (), {}

    monkeypatch.setattr(protocol, "_resolve_validated_axes", resolve)

    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(
            device="cpu",
            dtype="float32",
            validation_evidence=supplied,
        ),
    )

    assert seen == [
        (
            {"tp": 1, "ep": 1, "etp": 1, "pp": 1, "cp": 1},
            False,
            supplied,
        )
    ]
    assert bundle.extras["validated_axes"] == ()


def test_validation_stage_evidence_cannot_drift_from_runtime_contract():
    from pathlib import Path

    validation_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )

    protocol._assert_validation_doc_contract(validation_doc)


def test_validation_stage_evidence_drift_fails_loudly():
    from pathlib import Path

    validation_doc = (Path(__file__).parents[2] / "docs/validation.md").read_text(
        encoding="utf-8"
    )
    drifted = validation_doc.replace(
        "```json\n{}\n```",
        '```json\n{"ep": ["test:tests/gpu/fake.py::test_fake"]}\n```',
        1,
    )

    with pytest.raises(RuntimeError, match="validated-axis evidence drift"):
        protocol._assert_validation_doc_contract(drifted)


def test_dense_bundle_forwards_mask_into_cp_permuted_loss():
    bundle = build_model(
        _tiny_config(),
        impl_cfg=ImplConfig(device="cpu", dtype="float32"),
    )
    ps = SimpleNamespace(cp_size=2, cp_rank=1)

    class _MaskedLossChunk:
        def __call__(self, *, input_ids, labels, loss_mask=None):
            del input_ids
            labels_sb, mask_sb = prepare_labels_and_loss_mask(labels, loss_mask, ps)
            token_loss = labels_sb.float()
            if mask_sb is None:
                return {"loss": token_loss.mean()}
            mask_sb = mask_sb.to(token_loss.dtype)
            return {"loss": (token_loss * mask_sb).sum() / mask_sb.sum().clamp_min(1)}

    batch = SimpleNamespace(
        input_ids=torch.arange(8).view(1, 8),
        labels=torch.arange(8).view(1, 8),
        loss_mask=torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0]]),
    )

    output = bundle.forward_step(_MaskedLossChunk(), batch)

    assert output["loss"].item() == 3.0
