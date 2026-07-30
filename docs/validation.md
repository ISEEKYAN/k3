# Validation contract

This document separates verified behavior from planned validation. A green
packaging test is not evidence that the model, checkpoint, or distributed
paths work.

## Frozen bootstrap references

- Megatron Lite: `ISEEKYAN/Megatron-LM` commit
  `9a5d44e932587ae90489d23b782f0c3cd681aa46`
- External-package template: `ISEEKYAN/hy3` commit
  `77d99447f6726891d2fcd86f350378466f403783`
- Model release: `moonshotai/Kimi-K3` commit
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- Kimi-Linear reference: `MoonshotAI/Kimi-Linear` commit
  `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`
- FlashKDA reference: `MoonshotAI/FlashKDA` commit
  `d2ff19a6a0c82f39f796f637ebd1c36090b1268f`

The pinned Kimi K3 release has these SHA-256 digests:

| File | SHA-256 |
| --- | --- |
| `config.json` | `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` |
| `model.safetensors.index.json` | `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| `modeling_kimi_k3.py` | `b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2` |
| `modeling_kimi_linear.py` | `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a` |
| `configuration_kimi_k3.py` | `735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae` |

The 59,764,096-byte index contains 497,220 keys across 96 shards. Its
247,296 `weight_packed` and 247,296 `weight_scale` entries cover exactly
92 MoE layers × 896 routed experts × three projections. The remaining 2,628
tensors are unquantized. The package audits these counts and every mapped
text-backbone source name before it opens a weight shard.

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

### Verified first-release checkpoint proxy

The current single-rank CPU proxy is intentionally narrower than the complete
Stage 2 and Stage 3 gates below. It retains one KDA layer, one gated-MLA layer,
one LatentMoE layer, and two routed experts. It verifies:

- exact coverage of every real tiny-model parameter by the public weight map,
  including bias-free KDA depthwise convolutions;
- six routed-expert projection weights through MXFP4 pack/dequant, with all
  remaining parameters loaded through the unquantized path;
- independent equation-level KDA and gated-MLA-plus-LatentMoE layer outputs;
- maximum absolute differences of `0.0` for KDA,
  `2.384185791015625e-07` for MLA plus MoE, and
  `2.9802322387695312e-07` for final logits;
- quantized load followed by plain Hugging Face export into a fresh model,
  restoring every parameter and final logits bit-for-bit.

This proxy does not verify the full-width release, gradients against the public
model, distributed execution, GPU kernels, or short training.

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

#### Verified reduced functional proxy

The CPU proxy in `tests/parity/test_tiny_proxy_parity.py` uses seed
`20260727`, two layers, four routed experts, two selected experts per token,
float32, and fixed token/label batches. The production path is compared with a
separate functional oracle that reads copied weights but does not call the
production KDA, MLA, LatentMoE, decoder-layer, or model forwards.

Its measured maximum absolute differences are:

| Surface | Maximum absolute difference |
| --- | ---: |
| KDA layer output | `2.9802322387695312e-08` |
| gated-MLA/LatentMoE layer output | `8.940696716308594e-07` |
| logits | `3.5762786865234375e-07` |
| loss | `0.0` |
| six representative gradients | `2.384185791015625e-07` |

This satisfies the reduced tiny-config functional proxy only. It does not use
the full public model class or load weights through the public checkpoint
mapping, so the complete Stage 2 contract above remains not-done.

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

#### Complete-checkpoint validation command

Run the fail-closed validator against a shared, complete checkout of the pinned
release:

```bash
PYTHONPATH=<Megatron-Lite experimental/lite>:src \
  python -m mlite_k3.checkpoint_validation /shared/Kimi-K3 \
  --output ./k3-checkpoint-validation.json
```

The command verifies the frozen config and index SHA-256 values before opening
weights. It then visits every mapped logical parameter, applies the exact
public-to-native layout transform and its inverse, and compares shape, dtype,
and raw bytes. It retains one logical tensor group at a time. The JSON report
is atomically created only after the complete traversal succeeds.

The structural sample is deterministic: include the first and last layer,
every MLA layer and its adjacent KDA boundaries, and expert positions
`0, 223, 447, 671, 895`. This covers dense/MoE and KDA/MLA transitions across
the 93-layer schedule without claiming that a sample replaces the all-key
checkpoint traversal.

The report carries this independent structure-by-capability matrix:

| Structure | Load | Save | BF16 export | MXFP4 export | Canonical QAT | Shard rules |
| --- | --- | --- | --- | --- | --- | --- |
| dense | covered | covered | covered | BF16 passthrough | excluded by contract | plain |
| routed MoE | covered | covered | covered | packed + scale | covered | pair co-location |
| MLA | covered | covered | covered | BF16 passthrough | excluded by contract | plain |
| KDA | covered | covered | covered | BF16 passthrough | excluded by contract | plain |
| shared expert | covered | covered | covered | BF16 passthrough | excluded by contract | plain |
| router + expert bias | covered | covered | covered | BF16 passthrough | excluded by contract | plain |
| MTP | out of scope | out of scope | out of scope | out of scope | out of scope | out of scope |

Repository unit tests exercise the validator with real safetensors APIs and
small fixtures. They are regression coverage only: a report from the complete
1.56-TB release is required before claiming complete-checkpoint equality.

### Stage 4: scheduled GPU paths

**Deferred / not done in the first release.** Expert parallelism, context
parallelism, THD sequences, pipeline parallelism, and short training require
K3-owned sharding, parameter placement, state-transfer, and communication work
in this repository. They do not depend on a K3 primitive PR in Megatron Lite.

`ModelBundle.extras["validated_axes"]` is derived only from the following
machine-readable evidence manifest. Enabling a parallel dimension is not
evidence. Every entry must name a public scheduler test ID; the runtime and
this document are checked together and fail loudly if they drift.

<!-- K3_VALIDATED_AXIS_EVIDENCE_BEGIN -->
```json
{}
```
<!-- K3_VALIDATED_AXIS_EVIDENCE_END -->

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

Scheduler-backed proxy tests exist for one-rank parity, EP, CP, packed THD,
combined CP+EP, and PP, but they are not publication evidence until the unified
matrix is rerun for the exact release commit and its public test IDs are added
to the manifest above. Full-scale checkpoint training and a short
optimizer-backed train remain deferred. FlashKDA direct dispatch is also not
done; the approved training path is FLA `chunk_kda`.

## Evidence and publication rules

For every stage, record the exact source revisions, clean-tree status, command,
exit status, selected and skipped test counts, dtype, topology, and numerical
metrics. A skipped target is not a pass, and a test that silently selects a
fallback implementation is not evidence for the requested path.

Public documentation may claim only the stages rerun for the exact published
commit. Planned stages remain unchecked until their evidence is reproducible.
