"""Report every Transformer Engine build prerequisite before compilation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
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
    output_lines = (completed.stdout or completed.stderr).splitlines()
    return {
        "ok": completed.returncode == 0,
        "path": path,
        "rc": completed.returncode,
        "output_head": output_lines[:8],
        "output_tail": output_lines[-20:],
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
    build_deps = Path(os.environ["TE_BUILD_DEPS"]).resolve()
    tmp_dir = Path(os.environ["TE_BUILD_TMPDIR"]).resolve()
    min_tmp_bytes = int(os.environ.get("TE_MIN_TMP_BYTES", str(20 * 1024**3)))
    tmp_usage = shutil.disk_usage(tmp_dir)
    sys.path.insert(0, str(te_source))
    from build_tools import utils

    cuda_include = Path(os.environ["CUDA_HOME"]) / "targets/x86_64-linux/include"
    cudnn_include = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cudnn/include"
    nvtx_source_include = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/include"
    nccl_include = Path(sysconfig.get_paths()["purelib"]) / "nvidia/nccl/include"
    frontend_include = build_deps / "include"
    profiler_source_include = build_deps / "nvidia/cu13/include"
    nvtx_include = frontend_include
    nvtx_source_dir = nvtx_source_include / "nvtx3"
    if nvtx_source_dir.is_dir():
        shutil.copytree(
            nvtx_source_dir,
            nvtx_include / "nvtx3",
            dirs_exist_ok=True,
        )
    profiler_header = profiler_source_include / "cuda_profiler_api.h"
    if profiler_header.is_file():
        shutil.copy2(profiler_header, frontend_include / profiler_header.name)
    header_groups = {
        "cuda_headers": (
            cuda_include,
            [
                "cuda.h",
                "cuda_fp4.h",
                "cuda_fp8.h",
                "cuda_runtime.h",
                "cublasLt.h",
                "cublas_v2.h",
                "nvrtc.h",
                "cccl/cuda/barrier",
            ],
        ),
        "cudnn_headers": (
            cudnn_include,
            ["cudnn.h", "cudnn_graph.h"],
        ),
        "cudnn_frontend_headers": (
            frontend_include,
            ["cudnn_frontend.h", "cudnn_frontend_utils.h"],
        ),
        "nvtx_headers": (
            nvtx_include,
            ["nvtx3/nvToolsExt.h"],
        ),
        "nccl_headers": (
            nccl_include,
            ["nccl.h"],
        ),
    }
    checks: dict[str, object] = {
        "python": {
            "ok": sys.version_info >= utils.min_python_version(),
            "value": sys.version,
        },
        "torch": distribution("torch"),
        "tmp_disk": {
            "ok": tmp_usage.free >= min_tmp_bytes,
            "path": str(tmp_dir),
            "free_bytes": tmp_usage.free,
            "required_bytes": min_tmp_bytes,
        },
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
        "nvidia-cudnn-frontend": distribution("nvidia-cudnn-frontend"),
        "nvidia-cuda-profiler-api": distribution("nvidia-cuda-profiler-api"),
        "nvtx_source": {
            "ok": nvtx_source_dir.is_dir(),
            "value": str(nvtx_source_dir),
        },
    }
    source_include_pattern = re.compile(
        r"^\s*#\s*include\s*<((?:cuda|cublas|cudnn|nvrtc|nvtx3|nccl)[^>]+)>"
    )
    source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    required_cuda_headers: set[str] = set()
    for source_path in (te_source / "transformer_engine").rglob("*"):
        if not source_path.is_file() or source_path.suffix not in source_suffixes:
            continue
        for line in source_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            match = source_include_pattern.match(line)
            if match:
                required_cuda_headers.add(match.group(1))
    system_include_roots = [
        cuda_include,
        cuda_include / "cccl",
        cudnn_include,
        frontend_include,
        nccl_include,
    ]
    disabled_optional_headers: set[str] = set()
    if os.environ.get("NVTE_WITH_CUBLASMP", "0") == "0":
        disabled_optional_headers.add("cublasmp.h")
    if os.environ.get("NVTE_WITH_NCCL_EP", "1") == "0":
        disabled_optional_headers.add("nccl_ep.h")
    required_build_headers = required_cuda_headers - disabled_optional_headers
    missing_source_headers = [
        header
        for header in sorted(required_build_headers)
        if not any((root / header).is_file() for root in system_include_roots)
    ]
    checks["te_source_cuda_headers"] = {
        "ok": not missing_source_headers,
        "count": len(required_build_headers),
        "disabled_optional": sorted(required_cuda_headers & disabled_optional_headers),
        "missing": missing_source_headers,
        "roots": [str(root) for root in system_include_roots],
    }
    for name, (root, headers) in header_groups.items():
        missing = [header for header in headers if not (root / header).is_file()]
        checks[name] = {
            "ok": not missing,
            "root": str(root),
            "headers": headers,
            "missing": missing,
        }
    smoke_source = tmp_dir / "te_cuda_header_smoke.cu"
    smoke_object = tmp_dir / "te_cuda_header_smoke.o"
    smoke_source.write_text(
        """
#include <cuda/barrier>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nvrtc.h>
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_frontend.h>
#include "nvtx.h"
#include "util/logging.h"
__global__ void te_cuda_header_smoke() {}
""",
        encoding="utf-8",
    )
    checks["cuda_header_smoke"] = command(
        "nvcc",
        "-std=c++17",
        "-I",
        str(cudnn_include),
        "-I",
        str(frontend_include),
        "-I",
        str(nvtx_include),
        "-I",
        str(nccl_include),
        "-I",
        str(te_source / "transformer_engine/common"),
        "-I",
        str(te_source / "transformer_engine/common/include"),
        "-c",
        str(smoke_source),
        "-o",
        str(smoke_object),
    )
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
