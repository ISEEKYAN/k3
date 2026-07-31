"""K3 routing adapters for Megatron-Core training contracts."""

from __future__ import annotations

import torch

from megatron.lite.primitive.modules.router import SigmoidTopKRouter


class K3SigmoidTopKRouter(SigmoidTopKRouter):
    """Keep K3's static checkpoint bias compatible with MCore grad finalization."""

    def __init__(self, config, ps, **kwargs) -> None:
        super().__init__(config, ps, **kwargs)
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _apply(self, fn):
        super()._apply(fn)
        self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_scores, topk_indices = super().forward(x)
        if torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert.add_(
                    torch.bincount(
                        topk_indices.reshape(-1), minlength=self.num_experts
                    ).to(dtype=torch.float32)
                )
        return topk_scores, topk_indices


__all__ = ["K3SigmoidTopKRouter"]
