#!/usr/bin/env python3
"""Fail loudly if K3 JIT caches can race through a shared filesystem."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping


CACHE_PATHS = {
    "HOME": "home",
    "XDG_CACHE_HOME": "xdg",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "VLLM_CACHE_ROOT": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang-tmp",
    "PYTHONPYCACHEPREFIX": "pycache",
}


def assert_job_local_cache(env: Mapping[str, str]) -> Path:
    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required")
    job_root = Path(f"/tmp/k3-{job_id}")
    cache_dir: Path | None = None
    for name, suffix in CACHE_PATHS.items():
        raw_path = env.get(name)
        if not raw_path:
            raise RuntimeError(f"{name} is required")
        path = Path(raw_path)
        if path.name != suffix:
            raise RuntimeError(f"{name} has unexpected leaf: {path}")
        if path.parent.parent != job_root:
            raise RuntimeError(
                f"{name} must be isolated under {job_root}/<fingerprint>: {path}"
            )
        if cache_dir is None:
            cache_dir = path.parent
        elif path.parent != cache_dir:
            raise RuntimeError(
                f"{name} does not share the job fingerprint directory: {path}"
            )
    assert cache_dir is not None
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-self-test", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.negative_self_test:
        env["TORCHINDUCTOR_CACHE_DIR"] = "/lustre/shared/torchinductor"
        try:
            assert_job_local_cache(env)
        except RuntimeError as error:
            print(f"K3_JIT_CACHE_FAIL_LOUD_OK error={error}")
            return
        raise SystemExit("negative JIT cache self-test unexpectedly passed")
    cache_dir = assert_job_local_cache(env)
    print(f"K3_JIT_CACHE_LOCAL_OK root={cache_dir}")


if __name__ == "__main__":
    main()
