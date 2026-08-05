"""CPU contracts for the K3 disabled-DCP vLLM compatibility patch."""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest


def _make_mla_common_impl(*, set_sentinel: bool = True):
    class FakeMLACommonImpl:
        def __init__(
            self,
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
            indexer=None,
            q_pad_num_heads=None,
        ):
            if set_sentinel:
                self.dcp_world_size = -1

    return FakeMLACommonImpl


def _load_patch(monkeypatch, impl_cls):
    attention = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    attention.MLACommonImpl = impl_cls
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.attention": types.ModuleType(
            "vllm.model_executor.layers.attention"
        ),
        "vllm.model_executor.layers.attention.mla_attention": attention,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("mlite_k3.vllm_patch", None)
    return importlib.import_module("mlite_k3.vllm_patch")


def _construct(impl_cls):
    return impl_cls(
        128,
        576,
        1.0,
        1,
        None,
        None,
        "auto",
        None,
        "decoder",
        None,
        1536,
        512,
        128,
        64,
        192,
        128,
        object(),
    )


def test_patch_changes_disabled_dcp_sentinel_to_single_rank_state(monkeypatch):
    impl_cls = _make_mla_common_impl()
    unpatched = _construct(impl_cls)
    with pytest.raises(AssertionError, match="must be positive"):
        assert unpatched.dcp_world_size > 0, "cp_world_size must be positive"

    module = _load_patch(monkeypatch, impl_cls)
    module.apply_kimi_k3_mla_patch()
    patched = _construct(impl_cls)

    assert (patched.dcp_world_size, patched.dcp_rank) == (1, 0)


def test_patch_fails_loudly_when_vllm_constructor_contract_changes(monkeypatch):
    class ChangedMLACommonImpl:
        def __init__(self, incompatible_argument):
            self.dcp_world_size = -1

    module = _load_patch(monkeypatch, ChangedMLACommonImpl)
    with pytest.raises(RuntimeError, match="constructor signature changed"):
        module.apply_kimi_k3_mla_patch()


def test_patch_fails_loudly_when_disabled_dcp_sentinel_disappears(monkeypatch):
    impl_cls = _make_mla_common_impl(set_sentinel=False)
    module = _load_patch(monkeypatch, impl_cls)
    module.apply_kimi_k3_mla_patch()

    with pytest.raises(RuntimeError, match="dcp_world_size"):
        _construct(impl_cls)


def test_patch_contract_tracks_the_expected_vllm_constructor_shape(monkeypatch):
    impl_cls = _make_mla_common_impl()
    module = _load_patch(monkeypatch, impl_cls)

    assert (
        tuple(inspect.signature(impl_cls.__init__).parameters)
        == module._EXPECTED_INIT_PARAMETERS
    )
