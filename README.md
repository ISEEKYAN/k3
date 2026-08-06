# k3

## What this repository is

`mlite-k3` is an external Megatron Lite package for the public
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) model. It is
designed to keep model-specific composition outside Megatron Lite: importing
the package has no registration side effect, and applications explicitly call
`register_model()`.

The current text-only scope includes the native configuration, 69 KDA + 24
gated-MLA layer schedule, Attention Residual composition, LatentMoE, and a
single-rank Megatron Lite model protocol. The MoonViT vision encoder,
distributed kernels, optimizer integration, and production checkpoint loading
are not included yet. The repository does include a streaming-index audit and
an MXFP4 routed-expert pair decoder plus a real tiny-model load/export/reload
proxy, but it does not claim full-checkpoint loading. Unsupported multimodal,
parallel, and optimizer inputs fail explicitly.

The first release targets the `KimiLinearForCausalLM` text backbone. MoonViT-V2
and multimodal inputs are out of scope and must fail explicitly rather than
silently selecting the text path. Kimi K3 artifacts remain subject to the
[Kimi K3 License](LICENSE); this repository does not interpret its commercial
conditions.

The implementation is anchored to public
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3/tree/301be1b88c89c0d3a763da6301352cb8fe399e90)
custom code and the
[`MoonshotAI/FlashKDA`](https://github.com/MoonshotAI/FlashKDA/tree/d2ff19a6a0c82f39f796f637ebd1c36090b1268f)
kernel interface, together with
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention/tree/0a9b9f222e86b9a895c2447767e9b4cce6c8d530).
The initial CUDA compute path is `fla.ops.kda.chunk_kda`. FlashKDA is a
forward-only optional acceleration backend behind FLA; it has not been
validated in the approved environment and this package makes no FlashKDA
execution or performance claim. Context parallelism is not available on the
initial FLA path. The bounded torch recurrence is used only by the tiny CPU
correctness path.

The model projection wrapper consumes the K3-owned operator in
`mlite_k3.primitive.kda`. KDA recurrence and backend dispatch therefore ship
with this package. Checkpoint MXFP4 dequantization likewise lives in
`mlite_k3.primitive.mxfp4`; neither path requires a K3-specific change in
Megatron Lite.

## Tutorial 1: install the package

Clone Megatron Lite next to this external package, then create an isolated
environment inside the K3 checkout:

```bash
git clone https://github.com/ISEEKYAN/Megatron-LM.git
git clone https://github.com/ISEEKYAN/k3.git
git -C Megatron-LM checkout 85eacfbc1acbcfef9b003e0301d409be464d1377
cd k3

python -m venv .venv
. .venv/bin/activate
export PYTHONPATH="$PWD/../Megatron-LM/experimental/lite:$PWD/src"
python -m pip install -e '.[test]'
```

## Tutorial 2: register K3 explicitly

Register the external package before asking Megatron Lite to resolve Kimi K3:

```python
from mlite_k3 import register_model

register_model()

from megatron.lite.model.registry import (
    get_train_runtime_module,
    resolve_model_type_from_hf,
)

assert resolve_model_type_from_hf({"model_type": "kimi_k3"}) == "k3"
protocol = get_train_runtime_module("k3")
```

Importing `mlite_k3` alone does not mutate the registry.

## Tutorial 3: run the tiny hybrid model on CPU

The single-rank reference path is deliberately small and readable. It exercises
a real KDA layer, a real gated-MLA layer, Attention Residuals, a dense first
MLP, LatentMoE, logits, loss, and backward:

```python
import torch

from mlite_k3.config import K3Config
from mlite_k3.lite.protocol import ImplConfig, build_model

config = K3Config(
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=32,
    intermediate_size=24,
    max_position_embeddings=16,
    q_lora_rank=8,
    kv_lora_rank=4,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    kda_head_dim=4,
    kda_num_heads=2,
    kda_short_conv_kernel_size=2,
    full_attention_layers=(2,),
    kda_layers=(1,),
    attn_res_block_size=2,
    first_k_dense_replace=1,
    moe_intermediate_size=6,
    routed_expert_hidden_size=8,
    num_experts=4,
    num_experts_per_token=2,
    num_shared_experts=2,
)
bundle = build_model(
    config,
    impl_cfg=ImplConfig(device="cpu", dtype="float32"),
)
output = bundle.chunks[0](
    input_ids=torch.tensor([[1, 2, 3, 4]]),
    labels=torch.tensor([[2, 3, 4, 5]]),
)
output["loss"].backward()
print(output["logits"].shape)
```

This path is a correctness reference, not a performance implementation. It
does not claim full-checkpoint or distributed parity.

## Tutorial 4: run the CPU verification suite

Run the package contracts and the real Megatron Lite registry/model-bundle
smokes:

```bash
python -m pytest -q \
  tests/unit \
  tests/parity/test_tiny_proxy_parity.py \
  tests/smoke/test_registry_integration.py \
  tests/smoke/test_tiny_model_bundle.py
```

These checks require no CUDA. The verified checkpoint and numerical scope is
the reduced checkpoint proxy and independent functional proxy below; GPU and
distributed claims require their own non-skipped tests.

## Fail closed on non-finite RL metrics

Place the CPU-only streaming gate in every RL launcher before `tee`. It exits
with status 42 as soon as a `ppo_kl`, `loss`, or `grad_norm` metric is reported
as NaN or infinity. `pipefail` makes that status fail the job and also prevents
the producer from continuing after the gate closes the pipe:

```bash
set -o pipefail
PYTHONUNBUFFERED=1 run_training 2>&1 \
  | k3-rl-finite-gate \
  | tee "${log_file}"
```

The matching is restricted to structured `key=value` and `key: value` metric
fields whose key ends in `ppo_kl`, `loss`, or `grad_norm`; prose and unrelated
KL metrics are passed through unchanged. The bad line is preserved in the log,
and `K3_RL_NON_FINITE` identifies the rejected metric on standard error.

## Attention Residual-aware pipeline layout

`K3ParallelModel` uses Megatron Lite's
`build_pipeline_chunk_layout(..., decoder_layer_groups=...)` primitive. It
turns each configured Attention Residual block into an indivisible decoder
group for the default automatic layout and validates the resulting local stage
before constructing layers. PP sizes that leave a decoder-empty stage, layouts
that do not cover every decoder exactly once, and PP larger than the decoder
count fail during model construction.

The public 93-layer configuration uses `attn_res_block_size=12`, so its groups
are `0..11`, `12..23`, ..., `72..83`, and the final `84..92`. Automatic PP8
therefore assigns `[12, 12, 12, 12, 12, 12, 12, 9]` decoder layers. The first
stage also owns the embedding and the shorter final stage owns the final
Attention Residual projection, norm, and language-model head.

Block alignment is a default performance policy, not a correctness
requirement. An explicit `ParallelState.pp_layout` opts out of grouped
auto-layout and emits a warning describing the tradeoff. The cumulative
Attention Residual snapshots already travel in the folded pipeline activation,
so a split block adds no distinct P2P call and remains numerically valid.
Boundary placement can still change the folded activation width and pipeline
bubble, so explicit layouts should be benchmarked.

No separate P2P call is added for Attention Residuals. The existing pipeline
activation packs the normal hidden state together with the cumulative residual
snapshots. After aligned blocks, the seven PP8 boundaries carry 1 through 7
snapshots. For sequence-local token count `T`, hidden size `H`, element size
`D`, and boundary snapshot count `K`, the forward payload is
`T * H * D * (1 + K)` bytes; the backward gradient has the same shape. With
K3's `H=7168` and BF16, that is 14 KiB per local token for each hidden-sized
component, or 28 through 112 KiB total per token at the seven boundaries.
Residual blending computes its score in FP32 and converts back to the
activation dtype; pipeline transport is BF16 when activations are BF16, not
unconditionally FP32.

The default aligned layouts and decoder-only balance estimates are:

| PP | Decoder layers per stage | Layer utilization |
|---:|---|---:|
| 2 | `[48, 45]` | 96.875% |
| 3 | `[24, 36, 33]` | 86.111% |
| 5 | `[12, 12, 24, 24, 21]` | 77.500% |
| 6 | `[12, 12, 12, 12, 24, 21]` | 64.583% |
| 7 | `[12, 12, 12, 12, 12, 12, 21]` | 63.265% |
| 8 | `[12, 12, 12, 12, 12, 12, 12, 9]` | 96.875% |

Layer utilization is `93 / (PP * max_stage_layers)` and excludes embedding,
head, attention/MoE heterogeneity, and fill/drain bubble.

For PP3, an explicit `[32, 32, 29]` layout has 96.875% decoder-layer
utilization and 3.125% imbalance bubble, versus 86.111% utilization and 13.889%
bubble for any aligned layout whose bottleneck has 36 layers (including
`[36, 36, 21]` and the default `[24, 36, 33]`). Under a uniform per-layer
first-order model, reducing the bottleneck from 36 to 32 layers lowers
bottleneck compute time by 11.1% and raises throughput by 12.5%; production
timing must account for heterogeneous layers.

The aligned `[36, 36, 21]` boundaries and split `[32, 32, 29]` boundaries both
carry snapshot counts `[3, 6]`. With K3's `H=7168` and BF16, both therefore
send 56 KiB and 98 KiB per local token across their two forward boundaries
(154 KiB total), with the same shapes for backward. Their communication
difference is zero.

Construct the model normally; the grouped layout is automatic:

```python
from megatron.lite.primitive.parallel import ParallelState
from mlite_k3.config import K3Config
from mlite_k3.lite.model import K3ParallelModel

rank = 7
parallel = ParallelState(
    tp_size=1,
    cp_size=8,
    dp_size=8,
    ep_size=32,
    pp_size=8,
    pp_rank=rank,
    pp_is_first=rank == 0,
    pp_is_last=rank == 7,
)
model = K3ParallelModel(K3Config(), parallel)
assert model.layer_indices == list(range(84, 93))
```

To request the balanced PP3 layout explicitly:

```python
rows = [
    ["embedding", *(["decoder"] * 32)],
    ["decoder"] * 32,
    [*(["decoder"] * 29), "loss"],
]
parallel = ParallelState(
    pp_size=3,
    pp_rank=rank,
    pp_is_first=rank == 0,
    pp_is_last=rank == 2,
    pp_layout=rows,
)
```

## Tutorial 5: save a public HF checkpoint

`save_hf_weights` streams tensors into bounded safetensors shards.  It publishes
each shard and then atomically replaces the HF index, so an index never names a
partially written shard.  Use `target="bf16"` for a lossless model export or
`target="mxfp4"` for the public routed-expert compressed-tensors layout:

```python
from mlite_k3.lite.checkpoint import save_hf_weights

save_hf_weights(
    model,
    "./k3-hf",
    config,
    bundle.parallel_state,
    target="mxfp4",
    max_shard_size_bytes=5 * 1024**3,
)
```

The resulting `model.safetensors.index.json` has the shard `weight_map`, total
tensor bytes, and the chosen export format.  MXFP4 `_packed` and `_scale` keys
are always co-located in a shard; persistent router `expert_bias` remains a
plain public tensor.

## Tutorial 6: validate the pinned complete checkpoint

The validator accepts only the frozen `moonshotai/Kimi-K3` release at revision
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`. It checks the pinned config and
index hashes, then streams every mapped logical tensor through the public-to-
native-to-public layout transforms. Shape, dtype, and raw tensor bytes must all
match; signed zero and NaN payload differences are failures.

```bash
k3-validate-checkpoint /shared/Kimi-K3 \
  --output ./k3-checkpoint-validation.json
```

The JSON file is atomically published only after every tensor passes. It also
contains the reproducible 93-layer/896-expert structural sample and the
structure-by-capability coverage matrix. A missing shard, wrong release, or
single tensor mismatch exits nonzero without publishing a new report.

## R3 replay and MXFP4 QAT contracts

The K3 protocol exports Megatron Lite's standard zigzag-THD replay helpers:
`router_replay_roots`, `pack_routed_experts`, `pack_r3_replay_mask`, and
`unpack_thd_forward_output`. The caller owns replay-mask semantics; K3 only
converts routes and the caller-provided mask through the same THD/CP/TP layout.

For packed THD with context parallelism, pass the runtime `PackedBatch` to the
bundle's `forward_step`. The shared protocol performs the packing-aware CP split
once and marks `packed_seq_params.local_cp_size`; the model consumes that local
layout without slicing hidden states, labels, or the loss mask again. Reduced
eight-rank CP1/2/4 forward/backward examples are executable with:

```bash
for cp in 1 2 4; do
  K3_CP_SIZE="${cp}" torchrun --standalone --nproc-per-node=8 \
    tests/gpu/k3_thd_cp_smoke.py
done
```

Weight-only MXFP4 fake quantization is an explicit model-build option:

```python
bundle = build_model(
    config,
    impl_cfg=ImplConfig(
        device="cpu",
        dtype="bfloat16",
        qat={"enabled": True, "format": "mxfp4"},
    ),
)
print(bundle.extras["qat"])
```

K3 narrows the generic QAT target to the public checkpoint contract: only
routed-expert linear weights are parametrized. Attention, shared experts,
dense MLPs, embeddings, residual projections, routers, and the language-model
head remain unquantized. Checkpoint load and BF16 export resolve QAT
`parametrizations.weight.original` names back to their logical public names.
For rollout resynchronization, `protocol.export_hf_weights(
bundle.chunks, config, bundle.parallel_state, target="mxfp4")` gathers the
shared TP/ETP/EP/PP state before emitting `_packed`/`_scale` pairs only for
routed-expert `w1`, `w2`, and `w3`.

This is a QAT graph/checkpoint/export contract. K3 still rejects a non-null
`ImplConfig.optimizer`, so it does not claim an optimizer-backed QAT training
loop.

## Verified first-release checkpoint proxy

The single-rank CPU proxy retains one KDA layer, one gated-MLA layer,
LatentMoE, and two routed experts. Six routed-expert projection weights use
MXFP4 pairs; every other parameter follows the unquantized path.

The proxy verifies complete native-parameter coverage, MXFP4 load into the real
tiny model, independent KDA and MLA-plus-MoE layer equations, final logits, and
plain Hugging Face export/reload. The two layer maximum absolute differences
are `0.0` and `2.384185791015625e-07`; final-logit maximum absolute difference
is `2.9802322387695312e-07`. Plain export/reload restores every parameter and
the logits bit-for-bit.

The independent functional tiny proxy retains one
KDA layer, one gated-MLA layer, one dense MLP, and one four-expert LatentMoE
layer. With seed `20260727`, its maximum absolute differences were
`8.940696716308594e-07` across layer outputs,
`3.5762786865234375e-07` for logits, `0.0` for loss, and
`2.384185791015625e-07` across the six representative gradients checked.
This is reduced CPU proxy evidence, not execution of the full public model
class or checkpoint-conversion parity.

These CPU proxies do not establish GPU, distributed, or short-training
support. Those capabilities require their own scheduler-backed
forward/backward tests and are documented separately below.

## The four-stage model-support workflow

1. Freeze the public Kimi K3 configuration, custom modeling code, checkpoint
   index, and quantization metadata.
2. Implement KDA, gated MLA, LatentMoE, and parallel capabilities in the
   K3-owned primitive layer against explicit Megatron Lite protocols.
3. Implement configuration, the model protocol, and exhaustive checkpoint
   mappings without modifying Megatron Lite.
4. Validate CPU contracts first, then independent tiny-model parity, real
   checkpoint IO, short training, and distributed combinations through a
   scheduler.

No later-stage capability is considered supported until its corresponding test
has passed on the stated execution path.

## First-release boundary

The first release is limited to the readable reference implementation and
reduced-layer, reduced-expert tiny-config proxy parity. Proxy parity remains
not done until its independent, non-skipped evidence is published.

Reduced distributed proxies cover expert parallelism, context parallelism,
packed THD sequences, and pipeline parallelism in `tests/gpu`. Full-scale
checkpoint training remains deferred; no full-scale distributed or training
claim is implied by the CPU reference path or the reduced GPU proxies.

## Use this repository as a template

Keep model identity and composition in `src/mlite_k3/`, expose a single explicit
registration entry point, and keep fast CPU contracts in `tests/unit/`.
Reusable KDA recurrence/backend selection belongs in
`mlite_k3.primitive.kda`; `kda.py` owns K3 projections and output gating.
Gated-MLA and LatentMoE behavior belongs in `primitives.py`; `model.py` owns
configuration mapping, layer scheduling, and model-specific composition.
K3-specific primitives remain self-contained in this repository and require no
K3 changes to the Megatron Lite repository.

When adding a capability, add the smallest failing test first and document only
the behavior that the test actually exercises.
