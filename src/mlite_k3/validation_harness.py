"""Tiered K3 validation runner and execution-derived evidence bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlite_k3.validation_schema import (
    VALIDATION_AXES,
    capability_cells,
)


_SCHEMA = "mlite-k3-validation-evidence-v2"
_GENERATOR = "mlite_k3.validation_harness"
_CHECKPOINT_TEST_ID = "tests/gpu/test_checkpoint_load_smoke.py::main"
_CHECKPOINT_SCRIPT = "tests/gpu/test_checkpoint_load_smoke.py"
_CHECKPOINT_MARKER = "K3_CHECKPOINT_LOAD_SMOKE="


@dataclass(frozen=True)
class ValidationTier:
    name: str
    nodes: int
    tasks: int
    partition: str
    blocking: bool
    test_id: str
    command_marker: str
    success_marker: str


_TIERS = {
    tier.name: tier
    for tier in (
        ValidationTier(
            name="checkpoint_gather_1n",
            nodes=1,
            tasks=8,
            partition="interactive",
            blocking=True,
            test_id=_CHECKPOINT_TEST_ID,
            command_marker=_CHECKPOINT_SCRIPT,
            success_marker=_CHECKPOINT_MARKER,
        ),
        ValidationTier(
            name="checkpoint_gather_2n",
            nodes=2,
            tasks=8,
            partition="interactive",
            blocking=True,
            test_id=_CHECKPOINT_TEST_ID,
            command_marker=_CHECKPOINT_SCRIPT,
            success_marker=_CHECKPOINT_MARKER,
        ),
        ValidationTier(
            name="checkpoint_scale_4n",
            nodes=4,
            tasks=8,
            partition="batch_short",
            blocking=False,
            test_id=_CHECKPOINT_TEST_ID,
            command_marker=_CHECKPOINT_SCRIPT,
            success_marker=_CHECKPOINT_MARKER,
        ),
    )
}


@dataclass(frozen=True)
class VerifiedEvidence:
    git_commit: str
    capabilities: dict[str, tuple[str, ...]]
    axes: dict[str, tuple[str, ...]]
    runs: tuple[dict[str, Any], ...]
    missing_blocking_tiers: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text_atomically(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tier_plan(name: str) -> dict[str, Any]:
    try:
        tier = _TIERS[name]
    except KeyError as error:
        raise ValueError(f"unknown K3 validation tier {name!r}") from error
    return asdict(tier)


def write_run_record(
    run_dir: str | Path,
    *,
    tier: str,
    command: Sequence[str],
    returncode: int,
    duration_seconds: float,
    git_commit: str,
    slurm_job_id: str,
    slurm_nodes: int,
    slurm_partition: str,
) -> Path:
    """Write a fingerprinted record from a harness-owned test execution."""
    root = Path(run_dir)
    plan = _TIERS.get(tier)
    if plan is None:
        raise ValueError(f"unknown K3 validation tier {tier!r}")
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    if not stdout.is_file() or not stderr.is_file():
        raise RuntimeError("run artifacts require stdout.log and stderr.log")
    command = tuple(str(part) for part in command)
    if plan.command_marker not in command:
        raise RuntimeError(
            f"tier {tier!r} must execute {plan.command_marker!r}, got {command!r}"
        )
    record = {
        "schema": _SCHEMA,
        "generator": _GENERATOR,
        "tier": tier,
        "test_id": plan.test_id,
        "command": list(command),
        "returncode": int(returncode),
        "duration_seconds": float(duration_seconds),
        "git_commit": str(git_commit),
        "slurm": {
            "job_id": str(slurm_job_id),
            "nodes": int(slurm_nodes),
            "partition": str(slurm_partition),
        },
        "artifacts": {
            "stdout.log": _digest_file(stdout),
            "stderr.log": _digest_file(stderr),
        },
    }
    record["fingerprint"] = _digest_value(record)
    destination = root / "run.json"
    _write_json(destination, record)
    return destination


def _load_run_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid run record {path}") from error
    fingerprint = record.pop("fingerprint", None)
    if fingerprint != _digest_value(record):
        raise RuntimeError(f"run record fingerprint mismatch: {path}")
    if record.get("schema") != _SCHEMA or record.get("generator") != _GENERATOR:
        raise RuntimeError(f"unrecognized run record schema: {path}")
    record["fingerprint"] = fingerprint
    root = path.parent
    for name, expected in record.get("artifacts", {}).items():
        artifact = root / name
        if not artifact.is_file() or _digest_file(artifact) != expected:
            raise RuntimeError(f"run artifact digest mismatch: {artifact}")
    return record


def _parse_sacct(job_id: str, contents: str) -> dict[str, Any]:
    rows = [line.split("|") for line in contents.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) >= 5 and row[0] == job_id]
    if len(matches) != 1:
        raise RuntimeError(f"sacct did not return exactly one row for job {job_id}")
    _, state, exit_code, elapsed_raw, partition = matches[0][:5]
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(
            f"Slurm evidence requires COMPLETED and 0:0, got {state} and {exit_code}"
        )
    return {
        "job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "elapsed_seconds": int(elapsed_raw),
        "partition": partition,
    }


def _query_sacct(job_id: str) -> str:
    completed = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State,ExitCode,ElapsedRaw,Partition",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _parse_test_report(
    stdout: Path, tier: ValidationTier
) -> dict[str, tuple[str, ...]]:
    prefix = tier.success_marker
    payloads = [
        line.removeprefix(prefix)
        for line in stdout.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        raise RuntimeError(
            f"run must contain exactly one {prefix!r} test report, got {len(payloads)}"
        )
    try:
        report = json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {prefix!r} test report") from error
    if not isinstance(report, dict):
        raise RuntimeError(f"{prefix!r} test report must be a JSON object")
    raw_capabilities = report.get("capabilities")
    raw_axes = report.get("axes")
    if (
        not isinstance(raw_capabilities, list)
        or not all(isinstance(value, str) for value in raw_capabilities)
        or not isinstance(raw_axes, list)
        or not all(isinstance(value, str) for value in raw_axes)
    ):
        raise RuntimeError(
            "invalid test-reported capabilities or axes: "
            f"reported capabilities={raw_capabilities!r}, axes={raw_axes!r}"
        )
    capabilities = tuple(dict.fromkeys(raw_capabilities))
    axes = tuple(dict.fromkeys(raw_axes))
    valid_capabilities = set(capability_cells())
    unknown_capabilities = sorted(set(capabilities) - valid_capabilities)
    unknown_axes = sorted(set(axes) - set(VALIDATION_AXES))
    if not capabilities or unknown_capabilities or unknown_axes:
        raise RuntimeError(
            "invalid test-reported capabilities or axes: "
            f"reported capabilities={list(capabilities)}, "
            f"unknown_capabilities={unknown_capabilities}, "
            f"unknown_axes={unknown_axes}"
        )
    return {"capabilities": capabilities, "axes": axes}


def _validate_completed_run(
    record_path: Path,
    sacct_path: Path,
) -> tuple[
    dict[str, Any],
    ValidationTier,
    dict[str, Any],
    dict[str, tuple[str, ...]],
]:
    record = _load_run_record(record_path)
    tier = _TIERS.get(record["tier"])
    if tier is None:
        raise RuntimeError(f"run names unknown tier {record['tier']!r}")
    if record["returncode"] != 0 or record["duration_seconds"] <= 0:
        raise RuntimeError(
            f"run {record_path} did not complete successfully: "
            f"rc={record['returncode']}, duration={record['duration_seconds']}"
        )
    if record["test_id"] != tier.test_id:
        raise RuntimeError(f"run test id does not match tier {tier.name!r}")
    if tier.command_marker not in record["command"]:
        raise RuntimeError(f"run command does not match tier {tier.name!r}")
    slurm = record["slurm"]
    if slurm["nodes"] != tier.nodes or slurm["partition"] != tier.partition:
        raise RuntimeError(
            f"run Slurm allocation does not match tier {tier.name!r}: {slurm!r}"
        )
    stdout = record_path.parent / "stdout.log"
    test_report = _parse_test_report(stdout, tier)
    if not sacct_path.is_file():
        raise RuntimeError(f"missing sacct artifact {sacct_path}")
    sacct = _parse_sacct(
        slurm["job_id"],
        sacct_path.read_text(encoding="utf-8"),
    )
    if sacct["partition"] != tier.partition:
        raise RuntimeError(
            f"sacct partition {sacct['partition']!r} does not match tier "
            f"{tier.partition!r}"
        )
    return record, tier, sacct, test_report


def finalize_evidence_bundle(
    run_dirs: Sequence[str | Path],
    output: str | Path,
    *,
    sacct_query: Callable[[str], str] = _query_sacct,
) -> Path:
    """Query Slurm and sign coverage derived only from completed tier runs."""
    destination = Path(output)
    entries = []
    commits = set()
    for run_dir in run_dirs:
        root = Path(run_dir)
        record = _load_run_record(root / "run.json")
        job_id = record["slurm"]["job_id"]
        sacct_path = root / "sacct.txt"
        _write_text_atomically(sacct_path, sacct_query(job_id))
        verified, tier, sacct, _test_report = _validate_completed_run(
            root / "run.json",
            sacct_path,
        )
        commits.add(verified["git_commit"])
        entries.append(
            {
                "tier": tier.name,
                "record": os.path.relpath(root / "run.json", destination.parent),
                "sacct": os.path.relpath(sacct_path, destination.parent),
                "sacct_sha256": _digest_file(sacct_path),
                "job_id": sacct["job_id"],
            }
        )
    if len(commits) != 1:
        raise RuntimeError(f"evidence runs must share one git commit, got {commits}")
    payload = {
        "schema": _SCHEMA,
        "generator": _GENERATOR,
        "git_commit": commits.pop(),
        "runs": entries,
    }
    payload["fingerprint"] = _digest_value(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    return destination


def load_evidence_bundle(path: str | Path) -> VerifiedEvidence:
    """Verify bundle, run fingerprints, artifact digests, and Slurm terminal state."""
    source = Path(path)
    try:
        bundle = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid K3 evidence bundle {source}") from error
    fingerprint = bundle.pop("fingerprint", None)
    if (
        bundle.get("schema") != _SCHEMA
        or bundle.get("generator") != _GENERATOR
        or fingerprint != _digest_value(bundle)
    ):
        raise RuntimeError(f"invalid K3 evidence bundle {source}")

    capabilities: dict[str, list[str]] = {}
    axes: dict[str, list[str]] = {}
    verified_runs = []
    for entry in bundle.get("runs", ()):
        record_path = source.parent / entry["record"]
        sacct_path = source.parent / entry["sacct"]
        if _digest_file(sacct_path) != entry["sacct_sha256"]:
            raise RuntimeError(f"sacct artifact digest mismatch: {sacct_path}")
        record, tier, sacct, test_report = _validate_completed_run(
            record_path, sacct_path
        )
        if record["git_commit"] != bundle["git_commit"]:
            raise RuntimeError("run commit does not match evidence bundle commit")
        if sacct["job_id"] != entry["job_id"]:
            raise RuntimeError("sacct job id does not match evidence bundle")
        test_source = f"test:{tier.test_id}#sha256:{record['fingerprint']}"
        job_source = f"job:{sacct['job_id']}#sha256:{entry['sacct_sha256']}"
        for capability in test_report["capabilities"]:
            capabilities.setdefault(capability, []).extend((test_source, job_source))
        for axis in test_report["axes"]:
            axes.setdefault(axis, []).extend((test_source, job_source))
        verified_runs.append(
            {
                "tier": tier.name,
                "test_id": tier.test_id,
                "job_id": sacct["job_id"],
                "duration_seconds": record["duration_seconds"],
                "blocking": tier.blocking,
            }
        )

    if not verified_runs:
        raise RuntimeError("K3 evidence bundle contains no verified runs")
    completed_tiers = {run["tier"] for run in verified_runs}
    missing_blocking_tiers = tuple(
        tier.name
        for tier in _TIERS.values()
        if tier.blocking and tier.name not in completed_tiers
    )
    return VerifiedEvidence(
        git_commit=bundle["git_commit"],
        capabilities={
            key: tuple(dict.fromkeys(values)) for key, values in capabilities.items()
        },
        axes={key: tuple(dict.fromkeys(values)) for key, values in axes.items()},
        runs=tuple(verified_runs),
        missing_blocking_tiers=missing_blocking_tiers,
    )


def _run_tier(args: argparse.Namespace) -> int:
    plan = _TIERS[args.tier]
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if plan.command_marker not in command:
        raise RuntimeError(
            f"tier {plan.name!r} requires command member {plan.command_marker!r}"
        )
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (
        (root / "stdout.log").open("w", encoding="utf-8") as stdout,
        (root / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    duration = time.monotonic() - started
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_run_record(
        root,
        tier=plan.name,
        command=command,
        returncode=completed.returncode,
        duration_seconds=duration,
        git_commit=git_commit,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        slurm_nodes=int(os.environ.get("SLURM_NNODES", "0")),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("tier", choices=sorted(_TIERS))
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--tier", choices=sorted(_TIERS), required=True)
    run_parser.add_argument("--artifact-dir", type=Path, required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", action="append", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)

    if args.action == "plan":
        print(json.dumps(tier_plan(args.tier), indent=2, sort_keys=True))
        return 0
    if args.action == "run":
        return _run_tier(args)
    if args.action == "finalize":
        finalize_evidence_bundle(args.run_dir, args.output)
        return 0
    evidence = load_evidence_bundle(args.bundle)
    print(
        json.dumps(
            {
                "git_commit": evidence.git_commit,
                "capabilities": evidence.capabilities,
                "axes": evidence.axes,
                "runs": evidence.runs,
                "missing_blocking_tiers": evidence.missing_blocking_tiers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
