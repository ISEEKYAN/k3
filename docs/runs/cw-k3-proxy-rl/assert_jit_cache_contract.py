#!/usr/bin/env python3
"""Fail loudly unless K3 JIT caches share one stable fingerprint tree."""

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


def assert_stable_cache(env: Mapping[str, str]) -> Path:
    cache_root_raw = env.get("K3_CACHE_ROOT")
    fingerprint = env.get("K3_JIT_CACHE_FINGERPRINT")
    if not cache_root_raw:
        raise RuntimeError("K3_CACHE_ROOT is required")
    if not fingerprint or "/" in fingerprint:
        raise RuntimeError("K3_JIT_CACHE_FINGERPRINT must be one path component")
    cache_root = Path(cache_root_raw)
    if not cache_root.is_absolute() or cache_root.parts[:2] != ("/", "lustre"):
        raise RuntimeError(f"K3_CACHE_ROOT must be a stable Lustre path: {cache_root}")
    expected_cache_dir = cache_root / fingerprint
    cache_dir: Path | None = None
    for name, suffix in CACHE_PATHS.items():
        raw_path = env.get(name)
        if not raw_path:
            raise RuntimeError(f"{name} is required")
        path = Path(raw_path)
        if path.name != suffix:
            raise RuntimeError(f"{name} has unexpected leaf: {path}")
        if path.parent != expected_cache_dir:
            raise RuntimeError(f"{name} must be under {expected_cache_dir}: {path}")
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
        env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/k3-current/torchinductor"
        try:
            assert_stable_cache(env)
        except RuntimeError as error:
            print(f"K3_JIT_CACHE_FAIL_LOUD_OK error={error}")
            return
        raise SystemExit("negative JIT cache self-test unexpectedly passed")
    cache_dir = assert_stable_cache(env)
    print(f"K3_JIT_CACHE_STABLE_OK root={cache_dir}")


if __name__ == "__main__":
    main()
