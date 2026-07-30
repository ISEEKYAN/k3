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


def _completed_run(
    tmp_path: Path,
    *,
    tier: str = "checkpoint_gather_1n",
    assertions: tuple[tuple[str, str], ...] = (
        (
            "router_expert_bias.load",
            "expert_bias_is_finite_fp32",
        ),
    ),
    axes: tuple[str, ...] = ("tp", "ep", "pp"),
) -> Path:
    run_dir = tmp_path / tier
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(
        "K3_CHECKPOINT_LOAD_SMOKE="
        + json.dumps(
            {
                "world_size": 8,
                "assertions": [
                    {"cell": cell, "assertion": assertion}
                    for cell, assertion in assertions
                ],
                "axes": axes,
            }
        )
        + "\n",
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
    assert "capabilities" not in one
    assert "axes" not in one


def test_finalize_binds_each_explicit_cell_to_its_assertion_and_job(tmp_path):
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
    sources = evidence.capabilities["router_expert_bias.load"]
    assert any("::expert_bias_is_finite_fp32#sha256:" in source for source in sources)
    assert any(
        source.startswith("job:12345:assertion:expert_bias_is_finite_fp32#sha256:")
        for source in sources
    )


def test_capabilities_are_derived_from_individual_assertion_records(tmp_path):
    run_dir = _completed_run(
        tmp_path,
        assertions=(("moe.export_bf16", "export_matches_baseline"),),
        axes=("ep",),
    )
    bundle_path = tmp_path / "evidence.json"

    finalize_evidence_bundle(
        [run_dir],
        bundle_path,
        sacct_query=_completed_sacct,
    )
    evidence = load_evidence_bundle(bundle_path)

    assert set(evidence.capabilities) == {"moe.export_bf16"}
    assert set(evidence.axes) == {"ep"}


def test_one_run_cannot_claim_multiple_coverage_cells(tmp_path):
    run_dir = _completed_run(
        tmp_path,
        assertions=(
            ("moe.export_bf16", "export_matches_baseline"),
            ("router_expert_bias.load", "expert_bias_is_finite_fp32"),
        ),
    )

    with pytest.raises(RuntimeError, match="exactly one assertion"):
        finalize_evidence_bundle(
            [run_dir],
            tmp_path / "evidence.json",
            sacct_query=_completed_sacct,
        )


def test_smoke_without_cell_assertions_leaves_the_matrix_uncovered(tmp_path):
    run_dir = _completed_run(tmp_path, assertions=(), axes=("tp",))
    bundle_path = tmp_path / "evidence.json"

    finalize_evidence_bundle([run_dir], bundle_path, sacct_query=_completed_sacct)
    evidence = load_evidence_bundle(bundle_path)

    assert evidence.capabilities == {}
    assert set(evidence.axes) == {"tp"}


def test_missing_test_reported_assertions_cannot_sign_coverage(tmp_path):
    run_dir = _completed_run(tmp_path)
    (run_dir / "stdout.log").write_text(
        'K3_CHECKPOINT_LOAD_SMOKE={"world_size": 8}\n',
        encoding="utf-8",
    )
    write_run_record(
        run_dir,
        tier="checkpoint_gather_1n",
        command=[
            "torchrun",
            "--nproc-per-node=8",
            "tests/gpu/test_checkpoint_load_smoke.py",
        ],
        returncode=0,
        duration_seconds=12.5,
        git_commit="c3b1a4cd27",
        slurm_job_id="12345",
        slurm_nodes=1,
        slurm_partition="interactive",
    )

    with pytest.raises(RuntimeError, match="reported assertions"):
        finalize_evidence_bundle(
            [run_dir],
            tmp_path / "evidence.json",
            sacct_query=_completed_sacct,
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


def test_legacy_blanket_capability_list_is_rejected(tmp_path):
    run_dir = _completed_run(tmp_path)
    (run_dir / "stdout.log").write_text(
        'K3_CHECKPOINT_LOAD_SMOKE={"capabilities":["dense.load"],"axes":[]}\n',
        encoding="utf-8",
    )
    write_run_record(
        run_dir,
        tier="checkpoint_gather_1n",
        command=["torchrun", "tests/gpu/test_checkpoint_load_smoke.py"],
        returncode=0,
        duration_seconds=12.5,
        git_commit="c3b1a4cd27",
        slurm_job_id="12345",
        slurm_nodes=1,
        slurm_partition="interactive",
    )

    with pytest.raises(RuntimeError, match="reported assertions"):
        finalize_evidence_bundle(
            [run_dir], tmp_path / "evidence.json", sacct_query=_completed_sacct
        )
