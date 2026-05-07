# CLAUDE.md — `python/pydnn/transformer/`

## Purpose

Modern Transformer / Mixture-of-Experts building blocks for LLM-style
NLP work, exposed as `from pydnn import transformer`. This subpackage
is **pure Python on top of NumPy** — no torch / jax / tf dependency —
and ships its own minimal reverse-mode autograd engine so the models
can actually train end-to-end. It composes with the rest of `pydnn`
in spirit (numpy in / numpy out, soft topology mutations, end-of-
training compaction).

**Optional native backend.** When the C++ extension is built, the hot
ops (matmul, softmax, gelu, silu, layernorm, rmsnorm, embedding,
cross_entropy) transparently route through `_dnn_core.transformer_ops`
via the `_backend` dispatcher. The numpy reference is preserved as a
fallback (and selectable via `PYDNN_TRANSFORMER_BACKEND=numpy`); the
public Tensor / Module / `DynamicTransformer` API does not change.
The kernels *back* the existing dynamic transformer — they don't
replace it. Grow/prune logic, parameter discovery, and compact
semantics are unchanged.

## Files

| File | Role |
|---|---|
| `__init__.py` | Public exports for the subpackage. |
| `autograd.py` | Minimal Tensor + reverse-mode autograd (matmul, softmax, gelu, silu, embedding, cross_entropy, …). `[invariant]` external surface for layer ops. |
| `module.py` | `Module` / `ModuleList` base classes (torch.nn.Module analogue). Owns parameter discovery and `train()`/`eval()` modes. |
| `layers.py` | `Linear`, `LayerNorm`, `RMSNorm`, `Dropout`, `FeedForward`, `SwiGLU`. |
| `embeddings.py` | `TokenEmbedding`, `SinusoidalPositionalEncoding`, `LearnedPositionalEncoding`, `RotaryEmbedding` (RoPE — applied inside attention). |
| `attention.py` | `scaled_dot_product_attention`, `MultiHeadAttention`, `GroupedQueryAttention`, `KVCache`. `[hot]` |
| `moe.py` | `Expert`, `MixtureOfExperts` (top-k gated routing + Switch-style aux load-balance loss). `[hot]` |
| `model.py` | `TransformerConfig`, encoder/decoder layers and stacks, `EncoderOnlyModel`, `DecoderOnlyModel`, `Seq2SeqModel` / `Transformer`, `generate()`. `[invariant]` |
| `tokenizer.py` | `WhitespaceTokenizer`, `CharTokenizer`, tiny `BPETokenizer`. |
| `training.py` | `AdamW`, `SGD`, `cosine_with_warmup`, `linear_with_warmup`, `clip_grad_norm`, `CausalLMTrainer`. |
| `dynamic.py` | `DynamicTransformer` + `AdaptiveDecoderBlock` + `TransformerHealth` — the dnn-flavoured grow/prune wrapper. |
| `_backend.py` | Op dispatcher: routes `matmul / softmax / gelu / silu / layernorm / rmsnorm / embedding / xent` through `_dnn_core.transformer_ops` (cpu/cuda) when available, with a NumPy fallback. `set_backend("numpy"\|"cpu"\|"cuda")` and `PYDNN_TRANSFORMER_BACKEND` for testing parity. |

## Invariants

- **Public surface.** `TransformerConfig`, `EncoderOnlyModel`,
  `DecoderOnlyModel`, `Seq2SeqModel` (alias `Transformer`),
  `MultiHeadAttention`, `GroupedQueryAttention`, `MixtureOfExperts`,
  `CausalLMTrainer`, `AdamW`, and `DynamicTransformer` are external
  API. Add fields, don't remove them.
- **NumPy in, NumPy out at the edges.** Token IDs are
  `np.int64` numpy arrays; `Tensor` only appears on the autograd
  side. Don't expose `Tensor` in user-facing forward signatures
  (input_ids stay as numpy; logits are returned as `Tensor` so
  `.backward()` works).
- **`Module.parameters()` walks all sub-modules.** `Module.__setattr__`
  registers anything that's a `Parameter`, `Module`, or list/tuple of
  `Module`s. Don't bypass this with `object.__setattr__` outside the
  base class; you'll silently lose parameters from optimizer iteration.
- **Autograd graph is built per forward.** Call `optimizer.zero_grad()`
  before `loss.backward()` each step; the engine accumulates grads.
- **MoE auxiliary loss is opt-in.** `MixtureOfExperts.last_aux_loss`
  is populated each forward; the `*Model.aux_loss()` helper sums them.
  `CausalLMTrainer` adds it to the total loss when the model exposes
  `aux_loss()` returning a non-None Tensor. Don't silently drop it —
  the load-balance term is what stops experts from collapsing.
- **`pos_encoding="rope"` means no additive positional embedding.**
  `_make_pos_encoding` returns `(None, RotaryEmbedding(...))` — RoPE
  is wired into Q/K *inside* attention, not added to the residual
  stream. Don't double up.
- **`KVCache` is for inference only.** It bypasses autograd
  (`Tensor(cached_k)` is a leaf). Don't try to backprop through a
  cache; reset it between training and generation.
- **`DynamicTransformer.compact()` mirrors `Network::compact()`.**
  It physically drops blocks marked `active=False`. Like the C++
  side, only call it after training (or at least outside the
  optimizer's awareness — it changes `parameters()` membership).

## Maintenance notes

- The autograd engine is intentionally tiny. If you add an op,
  follow the pattern in `autograd.py`: forward computes data,
  records `_children`, sets `_backward` closure that *adds* to
  parent grads (never replaces — multiple paths into a Tensor
  must accumulate). `_unbroadcast` handles broadcasting reductions.
- `LayerNorm` / `RMSNorm` use a hand-rolled `_backward` rather than
  composing primitive ops: it's both faster and avoids numerical
  drift through the variance-reduction chain.
- `MoE` routing uses numpy indexing for the discrete top-k decision
  (no grad through indices) but keeps gradients flowing via the
  renormalised `weights` Tensor and via `gate(probs).mean()` for the
  aux loss. If you change the routing, preserve both gradient paths.
- `RotaryEmbedding._rotate` writes into a fresh `np.empty_like`
  buffer using even/odd slicing. If you switch to the "complex" RoPE
  layout (interleaved halves vs. real/imag pairs), update both
  forward and backward in lockstep.
- The simple `BPETokenizer` is O(merges × tokens) per train step —
  fine for demos, not for serious corpora. Swap it for the
  `tokenizers` library if you need scale.
- `DynamicTransformer.adapt()` is meant to be called *between*
  training steps (not inside `loss.backward()`). It mutates the
  `ModuleList`; the optimizer iterates `model.parameters()` afresh
  each step, so newly-added blocks pick up gradients automatically,
  but their AdamW moments start from zero — that's intentional.
- Keep this subpackage **import-cheap**. No matplotlib, no scipy,
  no optional ML frameworks. NumPy only.

## Quick smoke test

```python
import numpy as np
from pydnn import transformer as T

cfg = T.TransformerConfig(
    vocab_size=128, dim=32, num_heads=4,
    num_decoder_layers=2, ffn_type="swiglu",
    pos_encoding="rope", norm_type="rmsnorm",
    max_seq_len=64,
)
model = T.DecoderOnlyModel(cfg)
ids = np.random.randint(0, cfg.vocab_size, size=(4, 16))
logits = model(ids)              # Tensor, shape (4, 16, 128)
loss = T.cross_entropy(logits, ids)
loss.backward()
out = model.generate(ids[:, :4], max_new_tokens=8, top_k=10)
assert out.shape == (4, 12)
```

## Cross-refs

- Parent package contract: [`../CLAUDE.md`](../CLAUDE.md).
- Project-wide invariants (soft-delete topology, end-of-training
  compaction): [root `CLAUDE.md`](../../../CLAUDE.md). The
  `DynamicTransformer.compact()` semantics intentionally mirror
  `Network::compact()`.
