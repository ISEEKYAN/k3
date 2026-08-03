from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlite_k3.rl_finite_gate import find_non_finite_metric


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("line", "metric", "value"),
    [
        ("{'actor/ppo_kl': np.float32(nan)}", "actor/ppo_kl", "nan"),
        ('{"actor/pg_loss": Infinity}', "actor/pg_loss", "Infinity"),
        ("actor/grad_norm=tensor(-inf, device='cuda:0')", "actor/grad_norm", "-inf"),
        ("critic/vf_loss: +Inf", "critic/vf_loss", "+Inf"),
        ('{"loss": "NaN"}', "loss", "NaN"),
        ("ppo_kl=math.inf", "ppo_kl", "inf"),
    ],
)
def test_finds_non_finite_rl_metric(line: str, metric: str, value: str) -> None:
    assert find_non_finite_metric(line) == (metric, value)


@pytest.mark.parametrize(
    "line",
    [
        "actor/ppo_kl=0.0 actor/pg_loss=-1.25 actor/grad_norm=3.5e-4",
        "loss_mask=tensor([nan, nan])",
        "message='grad_norm may be inf in this explanation'",
        "rollout_corr/kl=nan",
    ],
)
def test_ignores_finite_or_unrelated_text(line: str) -> None:
    assert find_non_finite_metric(line) is None


def test_cli_streams_finite_lines_and_exits_nonzero_on_first_bad_metric() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mlite_k3.rl_finite_gate"],
        input=(
            "global_step=1 actor/ppo_kl=0.01 actor/pg_loss=0.5 "
            "actor/grad_norm=2.0\n"
            "global_step=2 actor/ppo_kl=nan actor/pg_loss=0.4\n"
            "global_step=3 actor/grad_norm=1.0\n"
        ),
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH")))
            ),
        },
    )

    assert result.returncode == 42
    assert result.stdout == (
        "global_step=1 actor/ppo_kl=0.01 actor/pg_loss=0.5 "
        "actor/grad_norm=2.0\n"
        "global_step=2 actor/ppo_kl=nan actor/pg_loss=0.4\n"
    )
    assert "K3_RL_NON_FINITE metric=actor/ppo_kl value=nan" in result.stderr
    assert "global_step=3" not in result.stdout


def test_cli_failure_propagates_through_documented_pipefail_pipeline() -> None:
    result = subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            "printf 'actor/grad_norm=inf\\n' | "
            "python -m mlite_k3.rl_finite_gate | tee /dev/null",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH")))
            ),
        },
    )

    assert result.returncode == 42
    assert "K3_RL_NON_FINITE metric=actor/grad_norm value=inf" in result.stderr
