from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlite_k3.validation_harness import (
    finalize_evidence_bundle,
    load_evidence_bundle,
    tier_plan,
    write_run_record,
)
from mlite_k3.validation_schema import capability_cells


def _completed_run(tmp_path: Path, *, tier: str = "checkpoint_gather_1n") -> Path:
    run_dir = tmp_path / tier
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(
        'K3_CHECKPOINT_LOAD_SMOKE={"world_size": 8}\n',
        encoding="utf-8",
    )
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    write_run_record(
        run_dir,
        tier=tier,
        command=[
            "torchrun",
            "--nproc-per-node=8",
            "tests/gpu/test_checkpoint_load_smoke.py",
        ],
        returncode=0,
        duration_seconds=12.5,
        git_commit="c3b1a4cd27",
        slurm_job_id="12345",
        slurm_nodes=1 if tier.endswith("1n") else 2,
        slurm_partition="interactive",
    )
    return run_dir


def _completed_sacct(_job_id: str) -> str:
    return "12345|COMPLETED|0:0|13|interactive\n"


def test_tiers_prioritize_one_and_two_node_interactive_gather_validation():
    one = tier_plan("checkpoint_gather_1n")
    two = tier_plan("checkpoint_gather_2n")
    scale = tier_plan("checkpoint_scale_4n")

    assert (one["nodes"], one["partition"], one["tasks"]) == (1, "interactive", 8)
    assert (two["nodes"], two["partition"], two["tasks"]) == (2, "interactive", 8)
    assert (scale["nodes"], scale["partition"]) == (4, "batch_short")
    assert one["blocking"] is True
    assert two["blocking"] is True
    assert scale["blocking"] is False
    assert set(one["capabilities"]) == set(capability_cells())


def test_finalize_derives_capabilities_and_axes_from_completed_run(tmp_path):
    run_dir = _completed_run(tmp_path)
    bundle_path = tmp_path / "evidence.json"

    finalize_evidence_bundle(
        [run_dir],
        bundle_path,
        sacct_query=_completed_sacct,
    )
    evidence = load_evidence_bundle(bundle_path)

    assert evidence.git_commit == "c3b1a4cd27"
    assert set(evidence.axes) == {"tp", "ep", "pp"}
    assert evidence.capabilities["router_expert_bias.load"][0].startswith("test:")
    assert any(
        source.startswith("job:12345#sha256:")
        for source in evidence.capabilities["moe.export_bf16"]
    )


def test_plain_handwritten_manifest_is_rejected(tmp_path):
    path = tmp_path / "handwritten.json"
    path.write_text(
        json.dumps({"dense.load": ["test:i_promise_this_ran"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="evidence bundle"):
        load_evidence_bundle(path)


def test_failed_or_unverifiable_slurm_job_cannot_sign_coverage(tmp_path):
    run_dir = _completed_run(tmp_path)

    with pytest.raises(RuntimeError, match="COMPLETED.*0:0"):
        finalize_evidence_bundle(
            [run_dir],
            tmp_path / "evidence.json",
            sacct_query=lambda _job_id: "12345|FAILED|1:0|2|interactive\n",
        )


def test_tampered_run_artifact_breaks_bundle_verification(tmp_path):
    run_dir = _completed_run(tmp_path)
    bundle_path = tmp_path / "evidence.json"
    finalize_evidence_bundle(
        [run_dir],
        bundle_path,
        sacct_query=_completed_sacct,
    )
    (run_dir / "stdout.log").write_text("forged\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact digest"):
        load_evidence_bundle(bundle_path)
