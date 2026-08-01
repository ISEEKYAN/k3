from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F


def _install_transformer_engine_import_stub(monkeypatch) -> None:
    root = ModuleType("transformer_engine")
    pytorch = ModuleType("transformer_engine.pytorch")
    permutation = ModuleType("transformer_engine.pytorch.permutation")
    router = ModuleType("transformer_engine.pytorch.router")
    cpp_extensions = ModuleType("transformer_engine.pytorch.cpp_extensions")
    module = ModuleType("transformer_engine.pytorch.module")
    module_base = ModuleType("transformer_engine.pytorch.module.base")

    def unavailable_kernel(*_args, **_kwargs):
        raise RuntimeError("Transformer Engine fused kernels are unavailable")

    for name in (
        "moe_permute",
        "moe_permute_and_pad_with_probs",
        "moe_permute_with_probs",
        "moe_unpermute",
    ):
        setattr(permutation, name, unavailable_kernel)
    for name in (
        "fused_compute_score_for_moe_aux_loss",
        "fused_moe_aux_loss",
        "fused_topk_with_score_function",
    ):
        setattr(router, name, unavailable_kernel)
    cpp_extensions.general_gemm = lambda *_args, **_kwargs: None
    module_base.get_workspace = lambda: None
    module.base = module_base
    pytorch.permutation = permutation
    pytorch.router = router
    pytorch.cpp_extensions = cpp_extensions
    pytorch.module = module
    root.pytorch = pytorch
    modules = {
        "transformer_engine": root,
        "transformer_engine.pytorch": pytorch,
        "transformer_engine.pytorch.permutation": permutation,
        "transformer_engine.pytorch.router": router,
        "transformer_engine.pytorch.cpp_extensions": cpp_extensions,
        "transformer_engine.pytorch.module": module,
        "transformer_engine.pytorch.module.base": module_base,
    }
    for name, stub in modules.items():
        monkeypatch.setitem(sys.modules, name, stub)


def _k3_router_call() -> ast.Call:
    model_path = Path(__file__).parents[2] / "src/mlite_k3/lite/model.py"
    tree = ast.parse(model_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "K3SigmoidTopKRouter"
    ]
    assert len(calls) == 1
    return calls[0]


def _dense_outputs(
    scores: torch.Tensor, indices: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = scores.new_zeros(scores.shape[0], num_experts)
    probabilities.scatter_(1, indices, scores)
    routing_map = torch.zeros(
        scores.shape[0],
        num_experts,
        dtype=torch.bool,
        device=scores.device,
    )
    routing_map.scatter_(1, indices, True)
    return probabilities, routing_map


def _reference_outputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(x.float(), weight.float())
    scores = torch.sigmoid(logits)
    selected = torch.topk(scores + expert_bias.float(), topk, dim=-1).indices
    selected = selected.sort(dim=-1).values
    probabilities = scores.gather(1, selected)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return _dense_outputs(probabilities, selected, weight.shape[0])


def test_k3_declares_and_matches_fp32_shared_router_contract(monkeypatch):
    call = _k3_router_call()
    router_dtype = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "router_dtype"),
        None,
    )
    assert isinstance(router_dtype, ast.Attribute)
    assert isinstance(router_dtype.value, ast.Name)
    assert (router_dtype.value.id, router_dtype.attr) == ("torch", "float32")
    expert_bias_persistent = next(
        (
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "expert_bias_persistent"
        ),
        None,
    )
    assert isinstance(expert_bias_persistent, ast.Constant)
    assert expert_bias_persistent.value is True

    _install_transformer_engine_import_stub(monkeypatch)
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    config = SimpleNamespace(
        hidden_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        aux_loss_alpha=0.0,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    parallel_state = SimpleNamespace(tp_size=1, tp_group=None)
    generator = torch.Generator().manual_seed(0)
    weight = (torch.randn(8, 16, generator=generator) * 2).to(torch.bfloat16)
    hidden = (torch.randn(32, 16, generator=generator) * 2).to(torch.bfloat16)

    default_router = SigmoidTopKRouter(
        config, parallel_state, compute_aux_loss=False
    ).to(torch.bfloat16)
    fp32_router = SigmoidTopKRouter(
        config,
        parallel_state,
        compute_aux_loss=False,
        router_dtype=torch.float32,
        expert_bias_persistent=True,
    ).to(torch.bfloat16)
    expert_bias = torch.linspace(-0.2, 0.2, config.n_routed_experts)
    with torch.no_grad():
        default_router.gate.weight.copy_(weight)
        fp32_router.gate.weight.copy_(weight)
        default_router.expert_bias.copy_(expert_bias)
        fp32_router.expert_bias.copy_(expert_bias)

    default_scores, default_indices = default_router(hidden)
    actual_scores, actual_indices = fp32_router(hidden)
    assert default_scores.dtype == torch.bfloat16
    assert fp32_router.router_dtype is torch.float32
    assert actual_scores.dtype == torch.float32
    default_probs, default_map = _dense_outputs(
        default_scores, default_indices, config.n_routed_experts
    )
    actual_probs, actual_map = _dense_outputs(
        actual_scores, actual_indices, config.n_routed_experts
    )
    reference_probs, reference_map = _reference_outputs(
        hidden,
        weight,
        expert_bias,
        config.num_experts_per_tok,
    )
    assert "expert_bias" in fp32_router.state_dict()
    metrics = {
        "default_probs_max_abs": (default_probs.float() - reference_probs)
        .abs()
        .max()
        .item(),
        "default_routing_map_mismatches": int((default_map != reference_map).sum()),
        "fp32_probs_max_abs": (actual_probs.float() - reference_probs)
        .abs()
        .max()
        .item(),
        "fp32_routing_map_mismatches": int((actual_map != reference_map).sum()),
    }
    print("K3_ROUTER_FP32_PARITY=" + json.dumps(metrics, sort_keys=True))

    assert metrics["default_probs_max_abs"] > 0.0
    assert metrics["fp32_probs_max_abs"] == 0.0
    assert metrics["fp32_routing_map_mismatches"] == 0


def test_k3_router_accumulates_replayed_expert_counts_for_mcore_finalize(monkeypatch):
    _install_transformer_engine_import_stub(monkeypatch)
    from mlite_k3.primitive.router import K3SigmoidTopKRouter

    config = SimpleNamespace(
        hidden_size=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        aux_loss_alpha=0.0,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    router = K3SigmoidTopKRouter(
        config,
        SimpleNamespace(tp_size=1, tp_group=None),
        compute_aux_loss=False,
        router_dtype=torch.float32,
        expert_bias_persistent=True,
    )
    hidden = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )

    class _Replay:
        def select_indices(self, native_indices):
            return torch.tensor([[0, 1], [0, 1]], device=native_indices.device)

    router.router_replay = _Replay()

    _, indices = router(hidden)
    expected = torch.bincount(
        indices.flatten(), minlength=config.n_routed_experts
    ).float()
    assert router.local_tokens_per_expert.dtype is torch.float32
    assert torch.equal(router.local_tokens_per_expert, expected)
    assert "expert_bias" in router.state_dict()
    assert "local_tokens_per_expert" not in router.state_dict()

    with torch.no_grad():
        router(hidden.detach())
    assert torch.equal(router.local_tokens_per_expert, expected)

    router.local_tokens_per_expert.zero_()
    assert torch.count_nonzero(router.local_tokens_per_expert) == 0
