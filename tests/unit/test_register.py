from __future__ import annotations

import importlib
import sys
from types import ModuleType


def test_register_model_is_explicit_and_uses_external_package_paths(monkeypatch):
    calls = []
    registry = ModuleType("megatron.lite.model.registry")
    registry.register_model = lambda *args, **kwargs: calls.append((args, kwargs))

    modules = {
        "megatron": ModuleType("megatron"),
        "megatron.lite": ModuleType("megatron.lite"),
        "megatron.lite.model": ModuleType("megatron.lite.model"),
        "megatron.lite.model.registry": registry,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("mlite_k3.register", None)
    register = importlib.import_module("mlite_k3.register")
    assert calls == []

    register.register_model()

    assert calls == [
        (
            ("k3",),
            {
                "package": "mlite_k3",
                "hf_model_types": ["kimi_k3", "kimi_linear"],
                "impls": {"lite": "mlite_k3.lite.protocol"},
            },
        )
    ]


def test_package_import_has_no_registration_side_effect(monkeypatch):
    calls = []
    registry = ModuleType("megatron.lite.model.registry")
    registry.register_model = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "megatron.lite.model.registry", registry)

    sys.modules.pop("mlite_k3", None)
    package = importlib.import_module("mlite_k3")

    assert callable(package.register_model)
    assert calls == []
