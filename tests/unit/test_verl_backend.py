from __future__ import annotations

import importlib
import sys
from types import ModuleType


def test_verl_backend_registers_k3_before_loading_mlite_backend(monkeypatch):
    events: list[str] = []
    package = importlib.import_module("mlite_k3")
    monkeypatch.setattr(package, "register_model", lambda: events.append("register"))

    engine = ModuleType("verl_mlite.engine.mlite_engine")
    engine.__dict__["loaded"] = True
    monkeypatch.setitem(sys.modules, "verl_mlite", ModuleType("verl_mlite"))
    monkeypatch.setitem(
        sys.modules, "verl_mlite.engine", ModuleType("verl_mlite.engine")
    )
    monkeypatch.setitem(sys.modules, "verl_mlite.engine.mlite_engine", engine)

    sys.modules.pop("mlite_k3.verl_backend", None)
    backend = importlib.import_module("mlite_k3.verl_backend")

    assert events == ["register"]
    assert backend.mlite_engine is engine
