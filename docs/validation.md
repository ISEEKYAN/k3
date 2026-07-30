# Validation contract

This document separates verified behavior from planned validation. A green
packaging test is not evidence that the model, checkpoint, or distributed
paths work.

## Frozen bootstrap references

- Megatron Lite: `ISEEKYAN/Megatron-LM` commit
  `85eacfbc1acbcfef9b003e0301d409be464d1377`
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

The command verifies the frozen config and index SHA-256 values, then visits
every mapped physical source through safetensors headers only. It checks the
complete key set, dtype and shape metadata, MXFP4 packed/scale pairing, fused
source compatibility, and the KDA convolution layout without materializing
tensor payloads. The JSON report is atomically created only after the metadata
traversal succeeds. This is a mapping/layout audit, not checkpoint-load or
numerical-parity evidence.

The report does not infer capability coverage from that traversal. Without an
execution-evidence manifest, every in-scope capability cell is emitted as
`not-covered`. A unified test must use the validation harness to produce a
fingerprinted evidence bundle from its run artifacts and Slurm accounting:

```bash
PYTHONPATH=<Megatron-Lite experimental/lite>:src \
  python -m mlite_k3.checkpoint_validation /shared/Kimi-K3 \
  --output ./k3-checkpoint-validation.json \
  --evidence-bundle ./k3-executed-evidence.json
```

Plain hand-written capability mappings are not accepted. The bundle verifier
checks the harness schema and fingerprint, run-record fingerprint, stdout and
stderr digests, the expected test and success marker, exact git commit, and a
`sacct` row with `COMPLETED` plus `0:0`. Capability and axis sources are then
derived from the registered tier rather than read from user-authored claims.

The structural sample is deterministic: include the first and last layer,
every MLA layer and its adjacent KDA boundaries, and expert positions
`0, 223, 447, 671, 895`. This covers dense/MoE and KDA/MLA transitions across
the 93-layer schedule without claiming that a sample replaces the all-key
checkpoint traversal.

With no evidence manifest, the matrix is deliberately:

<!-- K3_CAPABILITY_SCHEMA_BEGIN -->
```json
{
  "capabilities": [
    "load",
    "save",
    "export_bf16",
    "export_mxfp4",
    "qat_canonical",
    "shard_rules"
  ],
  "structures": [
    "dense",
    "moe",
    "mla",
    "kda",
    "shared_expert",
    "router_expert_bias"
  ]
}
```
<!-- K3_CAPABILITY_SCHEMA_END -->

| Structure | Load | Save | BF16 export | MXFP4 export | Canonical QAT | Shard rules |
| --- | --- | --- | --- | --- | --- | --- |
| dense | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |
| routed MoE | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |
| MLA | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |
| KDA | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |
| shared expert | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |
| router + expert bias | not-covered | not-covered | not-covered | not-covered | not-covered | not-covered |

MTP is outside this package scope and therefore has no matrix row rather than
a hard-coded coverage conclusion.

Repository unit tests include one on-disk safetensors fixture that exercises
the production header reader and atomic report path. Lower-level transform
tests use an in-memory metadata reader. Both are regression coverage only:
checkpoint-load equality requires the scheduler-backed distributed test and
cannot be inferred from this metadata report.

### Stage 4: scheduled GPU paths

**Deferred / not done in the first release.** Expert parallelism, context
parallelism, THD sequences, pipeline parallelism, and short training require
K3-owned sharding, parameter placement, state-transfer, and communication work
in this repository. They do not depend on a K3 primitive PR in Megatron Lite.

`ModelBundle.extras["validated_axes"]` is derived from
`ImplConfig.validation_evidence` for that exact build. Enabling a parallel
dimension is not evidence. Every entry must name a `test:` or `job:` execution
identifier; unknown axes, empty lists, and untraceable strings fail loudly.
When no per-run evidence is supplied, the following published default is used.
The repository contract test checks that this documented default and the
runtime default do not drift.

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

### Unified validation harness

The harness makes the reduced checkpoint import/export test the first-class
distributed gate. Its tiers deliberately do not require a four-node allocation
to establish TP/EP gather correctness:

| Tier | Nodes / tasks | Partition | Purpose | Blocking |
| --- | ---: | --- | --- | --- |
| `checkpoint_gather_1n` | 1 / 8 | `interactive` | single-node TP2/EP2/PP2 versus single-rank bitwise export, QAT key parity, persistent-buffer coverage | yes |
| `checkpoint_gather_2n` | 2 / 8 | `interactive` | the same topology distributed across two nodes, exercising cross-node collectives | yes |
| `checkpoint_scale_4n` | 4 / 8 | `batch_short` | scale confirmation only; queue delay does not block correctness evidence | no |

Inspect a tier before scheduling:

```bash
python -m mlite_k3.validation_harness plan checkpoint_gather_1n
```

Inside the matching Slurm allocation, wrap the exact checkpoint smoke command.
The harness owns stdout/stderr capture, wall duration, return code, git commit,
job ID, node count, partition, and the fingerprinted run record:

```bash
python -m mlite_k3.validation_harness run \
  --tier checkpoint_gather_1n \
  --artifact-dir ./artifacts/checkpoint_gather_1n \
  -- torchrun --standalone --nproc-per-node=8 \
  tests/gpu/test_checkpoint_load_smoke.py
```

After the job reaches a terminal state, finalize from a scheduler client. This
queries `sacct`; it refuses any state other than `COMPLETED`, any exit code
other than `0:0`, a mismatched partition/node count, a missing test success
marker, a changed artifact digest, or a run from a different git commit:

```bash
python -m mlite_k3.validation_harness finalize \
  --run-dir ./artifacts/checkpoint_gather_1n \
  --output ./artifacts/k3-evidence.json

python -m mlite_k3.validation_harness verify \
  ./artifacts/k3-evidence.json
```

Capability cells and axes are reported by the test itself only after its
assertions succeed; tiers contain scheduling and test-identity constraints,
not coverage claims. Finalization parses that test report, rejects unknown or
missing capabilities, and binds the claims to the fingerprinted run record
and `sacct` artifact. Copying or editing only the top-level JSON cannot
manufacture coverage. A checkpoint report additionally requires every tier
marked blocking (`checkpoint_gather_1n` and `checkpoint_gather_2n`); a partial
bundle may be inspected but cannot be published as complete evidence.

## Evidence and publication rules

For every stage, record the exact source revisions, clean-tree status, command,
exit status, selected and skipped test counts, dtype, topology, and numerical
metrics. A skipped target is not a pass, and a test that silently selects a
fallback implementation is not evidence for the requested path.

Public documentation may claim only the stages rerun for the exact published
commit. Planned stages remain unchecked until their evidence is reproducible.
