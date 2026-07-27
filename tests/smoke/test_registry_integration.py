from __future__ import annotations

from mlite_k3 import register_model


def test_real_mlite_registry_resolves_public_k3_model_types():
    from megatron.lite.model.registry import (
        get_model_package,
        resolve_model_type_from_hf,
    )

    register_model()

    assert resolve_model_type_from_hf({"model_type": "kimi_k3"}) == "k3"
    assert resolve_model_type_from_hf({"model_type": "kimi_linear"}) == "k3"
    assert get_model_package("k3").__name__ == "mlite_k3"
