#!/usr/bin/env python3
"""Add only the missing VERL dependency to the precedence-owning vLLM site."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ASSETS = ("tensordict", "tensordict-0.10.0.dist-info")


def ensure_links(source_site: Path, target_site: Path) -> None:
    for name in ASSETS:
        source = source_site / name
        target = target_site / name
        if not source.exists():
            raise RuntimeError(f"missing source overlay asset: {source}")
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(
                    f"unexpected existing overlay asset: {target} -> {target.resolve()}"
                )
            continue
        if target.exists():
            raise RuntimeError(f"unexpected existing overlay asset: {target}")
        try:
            os.symlink(source, target, target_is_directory=True)
        except FileExistsError:
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"unexpected existing overlay asset: {target}") from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-site", type=Path, required=True)
    parser.add_argument("--target-site", type=Path, required=True)
    args = parser.parse_args()
    ensure_links(args.source_site, args.target_site)
    print("K3_VLLM_OVERLAY_DEPS_OK tensordict=0.10.0", flush=True)


if __name__ == "__main__":
    main()
