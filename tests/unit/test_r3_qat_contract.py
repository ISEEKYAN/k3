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

    assert protocol.router_replay_roots is protocol_utils.router_replay_roots
    assert protocol.pack_routed_experts is protocol_utils.pack_routed_experts
    assert protocol.pack_r3_replay_mask is protocol_utils.pack_r3_replay_mask
    assert (
        protocol.unpack_thd_forward_output is protocol_utils.unpack_thd_forward_output
    )


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
            },
        )
    ]


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
    source = f"job:12345#sha256:{'a' * 64}"

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
