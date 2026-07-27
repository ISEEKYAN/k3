"""Explicit Megatron Lite registration entry point."""

from __future__ import annotations


def register_model() -> None:
    """Register Kimi K3 without modifying Megatron Lite's built-in registry."""
    from megatron.lite.model.registry import register_model as register

    register(
        "k3",
        package="mlite_k3",
        hf_model_types=["kimi_k3", "kimi_linear"],
        impls={"lite": "mlite_k3.lite.protocol"},
    )


__all__ = ["register_model"]
