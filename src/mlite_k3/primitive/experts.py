"""K3 latent-width routed experts using MLite's grouped-linear contract."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.experts import _AllReduceETP
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible


class K3LatentExperts(nn.Module):
    """Grouped experts whose input/output width is K3's latent MoE width."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.activation = activation
        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            hidden_size,
            intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            intermediate_size // ps.etp_size,
            hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            for parameter in self.parameters():

                def _all_reduce(gradient, group=ps.tp_group):
                    dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=group)
                    return gradient

                parameter.register_hook(_all_reduce)

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        real_length = x.shape[0]
        if self.etp_group is not None:
            max_length = torch.tensor(
                [real_length],
                device=x.device,
                dtype=torch.int64,
            )
            dist.all_reduce(max_length, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_length = int(max_length.item())
            if real_length < max_length:
                x = torch.cat(
                    (
                        x,
                        x.new_zeros(max_length - real_length, x.shape[1]),
                    ),
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        (
                            permuted_probs,
                            permuted_probs.new_zeros(max_length - real_length),
                        ),
                        dim=0,
                    )
                splits = list(splits)
                splits[-1] += max_length - real_length

        probabilities = (
            permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        )
        hidden = self.activation(self.fc1(x, splits), probabilities)
        output = self.fc2(hidden, splits)
        if self.etp_group is not None:
            output = _AllReduceETP.apply(output, self.etp_group)
            output = output[:real_length]
        return output


__all__ = ["K3LatentExperts"]
