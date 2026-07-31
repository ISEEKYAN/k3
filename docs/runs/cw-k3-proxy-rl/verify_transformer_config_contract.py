"""CPU gate for K3-to-MCore TransformerConfig lowering."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.protocol import unpack_forward_output
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.optimizers.megatron_wrap import (
    _build_transformer_config,
)
from megatron.lite.runtime.contracts.data import PackedBatch


def main() -> None:
    model_config = K3Config()
    parallel = SimpleNamespace(tp=1, pp=1, cp=1, ep=1, etp=1)
    transformer_config = _build_transformer_config(
        model_config,
        SimpleNamespace(parallel=parallel),
    )
    model_config.assert_transformer_config_contract(transformer_config)
    constructor, aliases = model_config.transformer_config_contract()
    print(
        "K3_TRANSFORMER_CONFIG_CPU_OK "
        f"mcore_fields={len(constructor)} aliases={len(aliases)} "
        f"first_k_dense_replace={transformer_config.first_k_dense_replace} "
        f"router_topk={transformer_config.moe_router_topk} "
        f"router_groups={transformer_config.moe_router_num_groups}/"
        f"{transformer_config.moe_router_group_topk}"
    )

    del transformer_config.first_k_dense_replace
    try:
        model_config.assert_transformer_config_contract(transformer_config)
    except RuntimeError as exc:
        if "first_k_dense_replace" not in str(exc):
            raise
    else:
        raise AssertionError(
            "K3 TransformerConfig contract accepted a missing first_k_dense_replace"
        )
    print("K3_TRANSFORMER_CONFIG_FAIL_LOUD_OK field=first_k_dense_replace")

    token_ids = torch.arange(5)
    batch = PackedBatch(
        input_ids=token_ids,
        labels=token_ids.clone(),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        loss_mask=torch.ones(5),
    )
    unpacked = unpack_forward_output(
        SimpleNamespace(ps=ParallelState()), batch, token_ids.float()
    )
    rows = list(unpacked.unbind(0))
    torch.testing.assert_close(rows[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(rows[1], torch.tensor([2.0, 3.0, 4.0]))
    print("K3_VERL_THD_UNPACK_CPU_OK seq_lens=[2,3] total_tokens=5")


if __name__ == "__main__":
    main()
