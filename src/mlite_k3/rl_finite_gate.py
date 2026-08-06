"""Fail fast when streamed RL metrics contain NaN or infinity."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable


NON_FINITE_EXIT_CODE = 42

_NON_FINITE_METRIC = re.compile(
    r"""
    (?<![A-Za-z0-9_./-])
    (?P<quote>["']?)
    (?P<metric>[A-Za-z0-9_./-]*?(?:ppo_kl|loss|grad_norm))
    (?P=quote)
    \s*[:=]\s*
    (?:(?:(?:np\.)?float(?:16|32|64)?|tensor)\s*\(\s*)*
    (?P<value_quote>["']?)
    (?:(?:np|numpy|math)\.)?
    (?P<value>[+-]?(?:nan|inf(?:inity)?))
    (?P=value_quote)
    (?![A-Za-z0-9_])
    """,
    re.IGNORECASE | re.VERBOSE,
)


def find_non_finite_metric(line: str) -> tuple[str, str] | None:
    """Return the first non-finite PPO KL, loss, or grad-norm metric in *line*."""

    match = _NON_FINITE_METRIC.search(line)
    if match is None:
        return None
    return match.group("metric"), match.group("value")


def gate_lines(lines: Iterable[str]) -> int:
    """Copy metric lines to stdout and stop at the first non-finite value."""

    for line in lines:
        sys.stdout.write(line)
        sys.stdout.flush()
        failure = find_non_finite_metric(line)
        if failure is not None:
            metric, value = failure
            print(
                f"K3_RL_NON_FINITE metric={metric} value={value}",
                file=sys.stderr,
                flush=True,
            )
            return NON_FINITE_EXIT_CODE
    return 0


def main() -> int:
    """Run the streaming stdin/stdout gate."""

    return gate_lines(sys.stdin)


if __name__ == "__main__":
    raise SystemExit(main())
