"""Kimi K3 text-backbone architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import Any


_FULL_ATTENTION_LAYERS = tuple(range(4, 93, 4)) + (93,)
_KDA_LAYERS = tuple(
    layer for layer in range(1, 94) if layer not in _FULL_ATTENTION_LAYERS
)


@dataclass(frozen=True)
class K3Config:
    """Native configuration for the text backbone in moonshotai/Kimi-K3.

    Layer numbers in ``full_attention_layers`` and ``kda_layers`` intentionally
    remain one-based, matching the public Kimi K3 configuration.
    """

    num_hidden_layers: int = 93
    hidden_size: int = 7168
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    vocab_size: int = 163840
    intermediate_size: int = 33792
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0

    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True

    kda_head_dim: int = 128
    kda_num_heads: int = 96
    kda_short_conv_kernel_size: int = 4
    kda_use_full_rank_gate: bool = True
    kda_gate_lower_bound: float = -5.0
    full_attention_layers: tuple[int, ...] = _FULL_ATTENTION_LAYERS
    kda_layers: tuple[int, ...] = _KDA_LAYERS
    attn_res_block_size: int = 12

    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: int = 3584
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    topk_method: str = "noaux_tc"
    use_grouped_topk: bool = True
    latent_moe_use_norm: bool = True

    source_model_type: str = "kimi_k3"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "full_attention_layers", tuple(self.full_attention_layers)
        )
        object.__setattr__(self, "kda_layers", tuple(self.kda_layers))
        self._validate()

    @property
    def layer_types(self) -> list[str]:
        full = set(self.full_attention_layers)
        return [
            "mla" if layer in full else "kda"
            for layer in range(1, self.num_hidden_layers + 1)
        ]

    def attention_type(self, layer_index: int) -> str:
        if not 0 <= layer_index < self.num_hidden_layers:
            raise IndexError(f"layer index out of range: {layer_index}")
        return self.layer_types[layer_index]

    @property
    def shared_expert_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def num_experts_per_tok(self) -> int:
        """Shared MoE primitive compatibility alias."""
        return self.num_experts_per_token

    @property
    def n_routed_experts(self) -> int:
        """Shared sigmoid-router compatibility alias."""
        return self.num_experts

    @property
    def scoring_func(self) -> str:
        """Shared MLite router spelling."""
        return self.moe_router_activation_func

    @staticmethod
    def ensure_text_only_inputs(
        *,
        pixel_values: Any | None = None,
        images: Any | None = None,
    ) -> None:
        if pixel_values is not None or images is not None:
            raise NotImplementedError(
                "mlite-k3 currently supports KimiLinearForCausalLM text inputs only; "
                "MoonViT-V2 and multimodal inputs are outside this release"
            )

    def _validate(self) -> None:
        errors: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        check(self.hidden_size > 0, "hidden_size must be positive")
        check(self.num_hidden_layers > 0, "num_hidden_layers must be positive")
        check(self.num_attention_heads > 0, "num_attention_heads must be positive")
        check(self.num_key_value_heads > 0, "num_key_value_heads must be positive")
        check(self.kda_num_heads > 0, "kda_num_heads must be positive")
        check(self.kda_head_dim > 0, "kda_head_dim must be positive")
        check(
            self.kda_short_conv_kernel_size > 0,
            "kda_short_conv_kernel_size must be positive",
        )
        check(self.q_lora_rank > 0, "q_lora_rank must be positive")
        check(self.kv_lora_rank > 0, "kv_lora_rank must be positive")
        check(self.qk_nope_head_dim > 0, "qk_nope_head_dim must be positive")
        check(self.qk_rope_head_dim > 0, "qk_rope_head_dim must be positive")
        check(self.v_head_dim > 0, "v_head_dim must be positive")
        check(self.mla_use_nope, "mla_use_nope must be enabled")
        check(self.mla_use_output_gate, "mla_use_output_gate must be enabled")
        check(self.kda_use_full_rank_gate, "kda_use_full_rank_gate must be enabled")
        check(
            self.kda_gate_lower_bound <= 0,
            "kda_gate_lower_bound must be non-positive",
        )

        expected = set(range(1, self.num_hidden_layers + 1))
        full = set(self.full_attention_layers)
        kda = set(self.kda_layers)
        check(
            full.isdisjoint(kda) and full | kda == expected,
            "attention layer schedule must cover each layer exactly once",
        )
        check(
            len(full) == len(self.full_attention_layers)
            and len(kda) == len(self.kda_layers),
            "attention layer schedule must not contain duplicates",
        )
        check(self.attn_res_block_size > 0, "attn_res_block_size must be positive")
        check(
            0 <= self.first_k_dense_replace <= self.num_hidden_layers,
            "first_k_dense_replace is out of range",
        )
        check(self.num_experts > 0, "num_experts must be positive")
        check(
            1 <= self.num_experts_per_token <= self.num_experts,
            "num_experts_per_token is out of range",
        )
        check(self.num_shared_experts == 2, "num_shared_experts must be 2")
        check(
            self.moe_router_activation_func == "sigmoid",
            "moe_router_activation_func must be sigmoid",
        )
        check(self.moe_renormalize, "moe_renormalize must be enabled")
        check(self.topk_method == "noaux_tc", "topk_method must be noaux_tc")
        check(self.use_grouped_topk, "use_grouped_topk must be enabled")
        check(self.latent_moe_use_norm, "latent_moe_use_norm must be enabled")
        check(self.hidden_act == "situ", "hidden_act must be situ")
        if errors:
            raise ValueError("Invalid K3Config:\n  " + "\n  ".join(errors))

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> "K3Config":
        from megatron.lite.primitive.config import load_hf_config_dict

        return cls._from_hf_dict(load_hf_config_dict(path_or_name), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> "K3Config":
        source = (
            hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        )
        return cls._from_hf_dict(source, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> "K3Config":
        source_type = hf.get("model_type", "kimi_k3")
        if source_type not in {"kimi_k3", "kimi_linear"}:
            raise ValueError(
                f"model_type must be kimi_k3 or kimi_linear, got {source_type!r}"
            )
        text = hf.get("text_config", hf)
        text_type = text.get("model_type", "kimi_linear")
        if text_type != "kimi_linear":
            raise ValueError(
                f"text_config.model_type must be kimi_linear, got {text_type!r}"
            )

        valid = {item.name for item in dc_fields(cls)}
        kwargs = {key: value for key, value in text.items() if key in valid}
        linear = text.get("linear_attn_config", {})
        aliases = {
            "head_dim": "kda_head_dim",
            "num_heads": "kda_num_heads",
            "short_conv_kernel_size": "kda_short_conv_kernel_size",
            "use_full_rank_gate": "kda_use_full_rank_gate",
            "gate_lower_bound": "kda_gate_lower_bound",
            "full_attn_layers": "full_attention_layers",
            "kda_layers": "kda_layers",
        }
        for source, target in aliases.items():
            if source in linear:
                kwargs[target] = linear[source]
        kwargs["source_model_type"] = source_type
        kwargs.update(overrides)
        return cls(**kwargs)


__all__ = ["K3Config"]
