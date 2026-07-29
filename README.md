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
    target="mxfp4",
    max_shard_size_bytes=5 * 1024**3,
)
```

The resulting `model.safetensors.index.json` has the shard `weight_map`, total
tensor bytes, and the chosen export format.  MXFP4 `_packed` and `_scale` keys
are always co-located in a shard; persistent router `expert_bias` remains a
plain public tensor.

## R3 replay and MXFP4 QAT contracts

The K3 protocol exports Megatron Lite's standard zigzag-THD replay helpers:
`router_replay_roots`, `pack_routed_experts`, `pack_r3_replay_mask`, and
`unpack_thd_forward_output`. The caller owns replay-mask semantics; K3 only
converts routes and the caller-provided mask through the same THD/CP/TP layout.

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
For rollout resynchronization, `iter_hf_weights(model, spec, target="mxfp4")`
emits `_packed`/`_scale` pairs only for routed-expert `w1`, `w2`, and `w3`.

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

Expert parallelism, context parallelism, THD sequences, pipeline parallelism,
and short training are deferred and not done. They do not depend on a K3
primitive change in Megatron Lite. Their unlock path is to implement and
validate K3-owned sharding, parameter placement, state transfer, and
communication primitives in this repository, then exercise them through
scheduler-backed tests. No distributed or training claim is implied by the CPU
reference path.

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
