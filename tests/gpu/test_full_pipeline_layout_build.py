"""Full-shape K3 pipeline construction probe.

Run this in the validated K3/Transformer Engine container.  Parameters are
created on the meta device so the probe validates the real 93-layer/896-expert
module graph and parallel contracts without allocating the full checkpoint.
"""

from __future__ import annotations

import torch
from unittest.mock import patch

from mlite_k3.config import K3Config


def _target_parallel_state(pp_rank: int):
    from megatron.lite.primitive.parallel import ParallelState

    return ParallelState(
        tp_size=1,
        ep_size=32,
        etp_size=1,
        cp_size=8,
        pp_size=8,
        dp_size=8,
        dp_cp_size=64,
        expert_dp_size=2,
        pp_rank=pp_rank,
        pp_is_first=pp_rank == 0,
        pp_is_last=pp_rank == 7,
    )


def test_full_k3_target_parallel_layout_builds_all_pipeline_stages():
    from mlite_k3.lite.model import K3ParallelModel

    config = K3Config()
    stage_layers: list[list[int]] = []
    parameter_counts: list[int] = []
    real_arange = torch.arange

    def _meta_safe_arange(*args, **kwargs):
        # TokenDispatcher immediately converts this rank/expert permutation to
        # Python lists.  Keep that small bookkeeping tensor on CPU while all
        # model parameters remain allocation-free on meta.
        if args == (config.num_experts,) and "device" not in kwargs:
            kwargs["device"] = "cpu"
        return real_arange(*args, **kwargs)

    with torch.device("meta"), patch("torch.arange", side_effect=_meta_safe_arange):
        for pp_rank in range(8):
            model = K3ParallelModel(config, _target_parallel_state(pp_rank))
            stage_layers.append(model.layer_indices)
            parameter_counts.append(
                sum(parameter.numel() for parameter in model.parameters())
            )

    assert [len(layers) for layers in stage_layers] == [12, 12, 12, 12, 12, 12, 12, 9]
    assert [layer for layers in stage_layers for layer in layers] == list(range(93))
    assert all(count > 0 for count in parameter_counts)
    print(
        "K3_FULL_LAYOUT_BUILD_OK "
        "parallel=TP1/CP8/DP8/PP8/EP32/ETP1 "
        f"stage_layers={stage_layers} "
        f"parameter_counts={parameter_counts}",
        flush=True,
    )


if __name__ == "__main__":
    test_full_k3_target_parallel_layout_builds_all_pipeline_stages()
