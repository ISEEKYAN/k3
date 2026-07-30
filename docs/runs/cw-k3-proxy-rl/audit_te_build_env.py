"""Report every Transformer Engine build prerequisite before compilation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def command(name: str, *args: str) -> dict[str, object]:
    path = shutil.which(name)
    if path is None:
        return {"ok": False, "error": "not found"}
    completed = subprocess.run(
        [path, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": completed.returncode == 0,
        "path": path,
        "rc": completed.returncode,
        "output": (completed.stdout or completed.stderr).splitlines()[:4],
    }


def distribution(name: str) -> dict[str, object]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"ok": False, "error": "not found"}
    return {
        "ok": True,
        "version": dist.version,
        "root": str(dist.locate_file("")),
    }


def main() -> None:
    te_source = Path(os.environ["TE_SOURCE"]).resolve()
    sys.path.insert(0, str(te_source))
    from build_tools import utils

    checks: dict[str, object] = {
        "python": {
            "ok": sys.version_info >= utils.min_python_version(),
            "value": sys.version,
        },
        "torch": distribution("torch"),
        "setuptools": distribution("setuptools"),
        "wheel": distribution("wheel"),
        "pip": distribution("pip"),
        "pybind11": {
            "ok": utils.found_pybind11(),
            "distribution": distribution("pybind11"),
        },
        "cmake": {"ok": utils.found_cmake(), "command": command("cmake", "--version")},
        "ninja": {"ok": utils.found_ninja(), "command": command("ninja", "--version")},
        "git": {
            **command("git", "--version"),
            "required": False,
        },
        "gcc": command(os.environ.get("CC", "gcc"), "--version"),
        "gxx": command(os.environ.get("CXX", "g++"), "--version"),
        "nvcc": command("nvcc", "--version"),
        "nvidia-cuda-nvcc": distribution("nvidia-cuda-nvcc"),
        "nvidia-cudnn-frontend": distribution("nvidia-cudnn-frontend"),
    }
    probes = {
        "nvcc_path": utils.nvcc_path,
        "cuda_version": utils.cuda_version,
        "cuda_include_dirs": utils.get_cuda_include_dirs,
        "cudnn_frontend_include": utils.cudnn_frontend_include_path,
        "cuda_archs": utils.cuda_archs,
        "frameworks": utils.get_frameworks,
    }
    for name, probe in probes.items():
        try:
            value = probe()
            if isinstance(value, (list, tuple)):
                value = [str(item) for item in value]
            else:
                value = str(value)
            checks[name] = {"ok": True, "value": value}
        except BaseException as exc:
            checks[name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    submodule_lines = os.environ["TE_SUBMODULE_STATUS"].splitlines()
    checks["submodules"] = {
        "ok": bool(submodule_lines)
        and all(line and line[0] not in "-+" for line in submodule_lines),
        "lines": submodule_lines,
    }
    failed = sorted(
        name
        for name, result in checks.items()
        if isinstance(result, dict)
        and result.get("required", True)
        and not result.get("ok", False)
    )
    print(
        "K3_TE_BUILD_AUDIT="
        + json.dumps({"checks": checks, "failed": failed}, sort_keys=True),
        flush=True,
    )
    if failed:
        raise SystemExit(f"TE build prerequisite audit failed: {failed}")


if __name__ == "__main__":
    main()
