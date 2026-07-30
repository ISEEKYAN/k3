"""Single source of truth for K3 validation capability cells."""

import re

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
VALIDATION_AXES = ("tp", "ep", "etp", "pp", "cp", "thd")
_EVIDENCE_SOURCE = re.compile(r"(?:test:[^#\s]+|job:\d+)#sha256:[0-9a-f]{64}\Z")


def capability_cells() -> tuple[str, ...]:
    return tuple(
        f"{structure}.{capability}"
        for structure in STRUCTURES
        for capability in CAPABILITIES
    )


def is_verified_evidence_source(source: str) -> bool:
    """Accept only harness-fingerprinted test or scheduler evidence IDs."""
    return _EVIDENCE_SOURCE.fullmatch(source) is not None
