import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2] / "docs/runs/cw-k3-proxy-rl/assert_jit_cache_contract.py"
)


def load_contract_module():
    spec = importlib.util.spec_from_file_location("jit_cache_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_accepts_one_job_scoped_node_local_cache_tree():
    module = load_contract_module()
    root = "/tmp/k3-14776736/fingerprint"
    env = {"SLURM_JOB_ID": "14776736"}
    env.update(
        {name: f"{root}/{suffix}" for name, suffix in module.CACHE_PATHS.items()}
    )

    assert module.assert_job_local_cache(env) == Path(root)


@pytest.mark.parametrize(
    "bad_path",
    (
        "/lustre/shared/jit-cache/torchinductor",
        "/tmp/k3-older-job/fingerprint/torchinductor",
    ),
)
def test_rejects_shared_or_cross_job_cache_paths(bad_path):
    module = load_contract_module()
    root = "/tmp/k3-14776736/fingerprint"
    env = {"SLURM_JOB_ID": "14776736"}
    env.update(
        {name: f"{root}/{suffix}" for name, suffix in module.CACHE_PATHS.items()}
    )
    env["TORCHINDUCTOR_CACHE_DIR"] = bad_path

    with pytest.raises(RuntimeError, match="TORCHINDUCTOR_CACHE_DIR"):
        module.assert_job_local_cache(env)
