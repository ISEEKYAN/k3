"""Isolate the production-shape FLA KDA forward/backward contract on one GPU."""

from __future__ import annotations

import datetime
import json

import torch

from mlite_k3.primitive.kda import kda


def mark(phase: str) -> None:
    print(
        "K3_KDA_PHASE="
        + json.dumps(
            {
                "phase": phase,
                "time": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20260730)
    shape = (1, 16, 96, 128)

    def tensor(tensor_shape, *, dtype=torch.bfloat16):
        return torch.randn(
            tensor_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )

    q = tensor(shape)
    k = tensor(shape)
    v = tensor(shape)
    gate = tensor(shape)
    beta = tensor(shape[:-1])
    a_log = tensor((shape[2],), dtype=torch.float32)
    dt_bias = tensor(shape[2:], dtype=torch.float32)

    mark("forward_start")
    output, final_state = kda(
        q,
        k,
        v,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        output_final_state=False,
        backend="fla",
    )
    mark("forward_done")
    if final_state is not None:
        raise AssertionError("FLA KDA unexpectedly returned a final state")

    mark("backward_start")
    output.float().square().mean().backward()
    mark("backward_done")
    gradients = (q, k, v, gate, beta, a_log, dt_bias)
    finite = all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in gradients
    )
    if not finite:
        raise RuntimeError("FLA KDA produced missing or non-finite gradients")
    print(
        "K3_KDA_BACKWARD_OK="
        + json.dumps(
            {
                "shape": shape,
                "output_max_abs": output.detach().float().abs().max().item(),
                "finite_grads": finite,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
