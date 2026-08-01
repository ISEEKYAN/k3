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


def test_accepts_one_fingerprint_scoped_stable_cache_tree():
    module = load_contract_module()
    cache_root = "/lustre/shared/k3/jit-cache"
    fingerprint = "ae4bf8099e118ce4b3a9"
    root = f"{cache_root}/{fingerprint}"
    env = {"K3_CACHE_ROOT": cache_root, "K3_JIT_CACHE_FINGERPRINT": fingerprint}
    env.update(
        {name: f"{root}/{suffix}" for name, suffix in module.CACHE_PATHS.items()}
    )

    assert module.assert_stable_cache(env) == Path(root)


@pytest.mark.parametrize(
    "bad_path",
    (
        "/lustre/shared/k3/jit-cache/other-fingerprint/torchinductor",
        "/tmp/k3-current-job/fingerprint/torchinductor",
    ),
)
def test_rejects_paths_outside_the_declared_stable_fingerprint(bad_path):
    module = load_contract_module()
    cache_root = "/lustre/shared/k3/jit-cache"
    fingerprint = "ae4bf8099e118ce4b3a9"
    root = f"{cache_root}/{fingerprint}"
    env = {"K3_CACHE_ROOT": cache_root, "K3_JIT_CACHE_FINGERPRINT": fingerprint}
    env.update(
        {name: f"{root}/{suffix}" for name, suffix in module.CACHE_PATHS.items()}
    )
    env["TORCHINDUCTOR_CACHE_DIR"] = bad_path

    with pytest.raises(RuntimeError, match="TORCHINDUCTOR_CACHE_DIR"):
        module.assert_stable_cache(env)
