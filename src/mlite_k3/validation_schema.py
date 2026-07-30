"""Single source of truth for K3 validation capability cells."""

CAPABILITIES = (
    "load",
    "save",
    "export_bf16",
    "export_mxfp4",
    "qat_canonical",
    "shard_rules",
)
STRUCTURES = (
    "dense",
    "moe",
    "mla",
    "kda",
    "shared_expert",
    "router_expert_bias",
)


def capability_cells() -> tuple[str, ...]:
    return tuple(
        f"{structure}.{capability}"
        for structure in STRUCTURES
        for capability in CAPABILITIES
    )
