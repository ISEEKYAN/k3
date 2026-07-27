# Validation contract

This document separates verified behavior from planned validation. A green
packaging test is not evidence that the model, checkpoint, or distributed
paths work.

## Frozen bootstrap references

- Megatron Lite: `ISEEKYAN/Megatron-LM` commit
  `9a5d44e932587ae90489d23b782f0c3cd681aa46`
- External-package template: `ISEEKYAN/hy3` commit
  `77d99447f6726891d2fcd86f350378466f403783`
- Model configuration: `moonshotai/Kimi-K3`, whose public configuration uses
  `model_type=kimi_k3` and `text_config.model_type=kimi_linear`

The model revision, custom modeling files, checkpoint index, and quantization
metadata must be pinned together before numerical tests are accepted.

## Frozen first-release scope

- Support the `KimiLinearForCausalLM` text backbone. Reject image or video
  inputs explicitly at the configuration, processor, or model boundary.
- Implement KDA as a reusable primitive against the pinned public
  `modeling_kimi_linear.py` equations and the published kernel. Gated DeltaNet
  is not an equivalent numerical reference.
- Reuse the existing MLA dimensions, while making NoPE and the output gate
  explicit capabilities that are disabled by default for other models.
- Use reduced layer and expert counts for correctness and parallel proxy tests.
  Do not use the full model or a full-width layer as the first correctness gate.
- Parse the checkpoint index and compressed-tensors metadata before opening
  shards. Only routed-expert linear weights use MXFP4; attention, shared
  experts, dense MLP, output projection, and vision tensors remain unquantized.
- Preserve the upstream Kimi K3 License and notice without making compatibility
  or commercial-use claims.

## Verified CPU bootstrap

At repository commit `6355ae9`, the following checks passed:

```text
python -m pytest -q tests/unit
2 passed

PYTHONPATH=<Megatron-Lite experimental/lite>:src \
  python -m pytest -q tests/smoke/test_registry_integration.py
1 passed
```

These checks cover editable packaging, side-effect-free import, explicit
registration, both public Hugging Face model types, and resolution through the
real Megatron Lite registry. They do not cover model construction.

## Increasing-cost validation stages

### Stage 1: CPU model contracts

Use a reduced text-only configuration that retains one KDA layer, one gated
MLA layer, routed and shared experts, and the official layer-order rules. Check:

- every public configuration field is mapped or rejected explicitly;
- model construction reaches the registered protocol;
- tensor shapes and parameter names cover KDA, gated MLA, router, routed
  experts, shared experts, embeddings, normalization, and output projection;
- unsupported multimodal inputs fail explicitly instead of silently falling
  back to the text model;
- every test selected by the CPU quick start passes without a skip.

### Stage 2: independent tiny-model parity

Build matched tiny models from the pinned public implementation and Megatron
Lite. Copy weights through the public checkpoint mapping rather than assigning
internal tensors ad hoc. Compare:

- KDA and gated-MLA layer outputs;
- router indices and scores;
- final logits and loss;
- representative gradients for embeddings, KDA, MLA, router, experts, and the
  output projection.

Freeze seeds, input tokens, dtype, and tolerances before inspecting results.
The two compared paths must not share the implementation under test.

### Stage 3: checkpoint and quantization

Exercise the public checkpoint index and compressed-tensors metadata without
materializing the full release in one process. Require:

- complete source-key consumption with an explicit allowlist for excluded
  tensors;
- MXFP4 unpack/dequant comparison against an independent reference;
- quantized load into the tiny model followed by forward parity;
- non-quantized export and reload with the same named tensors and outputs;
- clear rejection of unsupported quantization layouts.

Synthetic round trips alone do not satisfy this stage.

### Stage 4: scheduled GPU paths

Run from a clean checkout using a pinned container and Megatron Lite revision.
Each selected test must execute rather than skip, complete forward and backward,
and assert the requested topology and communication path. The minimum matrix is:

- one-rank independent parity and real checkpoint IO;
- expert parallelism with an all-to-all path assertion;
- context parallelism with variable-length THD sequences;
- combined context and expert parallelism;
- pipeline parallelism with layer ownership and boundary assertions;
- a short fixed-batch train whose loss remains finite and decreases.

If a proxy topology is used, report it as proxy evidence. It must not be
described as a full-scale Kimi K3 run.

## Evidence and publication rules

For every stage, record the exact source revisions, clean-tree status, command,
exit status, selected and skipped test counts, dtype, topology, and numerical
metrics. A skipped target is not a pass, and a test that silently selects a
fallback implementation is not evidence for the requested path.

Public documentation may claim only the stages rerun for the exact published
commit. Planned stages remain unchecked until their evidence is reproducible.
