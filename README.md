# k3

## What this repository is

`mlite-k3` is an external Megatron Lite package for the public
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) model. It is
designed to keep model-specific composition outside Megatron Lite: importing
the package has no registration side effect, and applications explicitly call
`register_model()`.

The initial bootstrap covers packaging and the external registration contract.
Model construction, checkpoint loading, and numerical-parity claims will only
be added with executable tests and recorded public references.

The first release targets the `KimiLinearForCausalLM` text backbone. MoonViT-V2
and multimodal inputs are out of scope and must fail explicitly rather than
silently selecting the text path. Kimi K3 artifacts remain subject to the
[Kimi K3 License](LICENSE); this repository does not interpret its commercial
conditions.

## Quick start

Clone Megatron Lite next to this repository, create an isolated environment,
install the package, and run the CPU contract tests:

```bash
git clone https://github.com/ISEEKYAN/Megatron-LM.git
git clone https://github.com/ISEEKYAN/k3.git
cd k3

python -m venv .venv
. .venv/bin/activate
export PYTHONPATH="$PWD/../Megatron-LM/experimental/lite:$PWD/src"
python -m pip install -e '.[test]'
python -m pytest -q tests/unit
```

Register the external package before asking Megatron Lite to resolve Kimi K3:

```python
from mlite_k3 import register_model

register_model()
```

The same environment can verify the registration contract against the real
Megatron Lite registry:

```bash
python -m pytest -q tests/smoke/test_registry_integration.py
```

Both CPU commands pass without CUDA. The bootstrap intentionally does not yet
expose a buildable model protocol.

## The four-stage model-support workflow

1. Freeze the public Kimi K3 configuration, custom modeling code, checkpoint
   index, and quantization metadata.
2. Map KDA, gated MLA, LatentMoE, and parallel capabilities to validated
   Megatron Lite primitives.
3. Implement configuration, the model protocol, and exhaustive checkpoint
   mappings without modifying Megatron Lite.
4. Validate CPU contracts first, then independent tiny-model parity, real
   checkpoint IO, short training, and distributed combinations through a
   scheduler.

No later-stage capability is considered supported until its corresponding test
has passed on the stated execution path.

## Use this repository as a template

Keep model identity and composition in `src/mlite_k3/`, expose a single explicit
registration entry point, and keep fast CPU contracts in `tests/unit/`.
Architecture-specific code should depend on validated framework primitives
instead of copying shared attention, expert, parallel, or checkpoint logic.

When adding a capability, add the smallest failing test first and document only
the behavior that the test actually exercises.
