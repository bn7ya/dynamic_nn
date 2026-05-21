# CLAUDE.md — `csrc/include/enn/modules/`

## Purpose

`torch::nn::Module` subclasses for the reversible topology. All of
them rely on libtorch's autograd and parameter machinery — we do not
implement backward passes.

**Python view.** Each `*Impl` class here is exposed through pybind11
under its `TORCH_MODULE` wrapper name (`ReversibleLinear`,
`ReversibleNetwork`, `AdaptiveConv2d`). On the Python side the
classes expose `forward`, `parameters`, and `buffers` — enough for
`torch.optim.Adam(net.parameters(), …)` and a standard
`for x, y in DataLoader: …` training loop. Full `torch.nn.Module`
Python API parity (`state_dict()`, `.to(device)`, `eval()`,
`train()`) is recorded as future work; today these are pybind11
classes, not `torch.nn.Module` subclasses on the Python side.
The soft-prune / compact / topology_version contracts described
below hold regardless of who is driving the training loop —
`PhaseController` or a user's own `for` loop.

## Files

| File | Role |
|---|---|
| `node_metrics.hpp` | Per-node EMA of variance / gradient / dead-ratio / contribution. EMA α and the warmup sample count live in `NodeMetricsConfig`. |
| `reversible_linear.hpp` | `ReversibleLinearImpl(in, out)` exposes `weight` / `bias` as parameters and `active_mask` as a buffer. Methods: `add_nodes`, `prune_nodes`, `compact`, `update_node_metrics`, `topology_version`. `[invariant]` `forward` = `linear(input, weight, bias) * active_mask`. |
| `reversible_network.hpp` | Composes `ReversibleLinear` layers; supports `insert_layer`, `remove_layer`, `compact`. Caches per-layer inputs during forward so the trainer can compute metrics on the actual layer-local tensors. |
| `adaptive_conv2d.hpp` | One `torch::nn::Conv2d` per candidate kernel size, mixed by `softmax(architecture_logits)`. Pruning freezes the smallest candidate; growing adds an outer ring to the active candidate without changing its output at the moment of growth. |

## Invariants

- **`active_mask` is a buffer (no gradient)**; the mask gates the
  output, not the parameter update.
- **`add_nodes` reactivates dormant slots before allocating new
  rows.** This is what makes prune→add a round-trip — see
  `tests/cpp/test_reversible_linear.cpp::PruneThenAddRoundTrip`.
- **`compact()` returns a non-negative count of rows / layers
  dropped** and bumps `topology_version()` even when nothing was
  dropped.
- **`rebuild_layer_preserving_`** copies the surviving weight
  sub-block when fan-in changes; new weight rows / cols are
  zero-initialised.
- **AdaptiveConv2d preserves its forward output when growing the
  active kernel** (outer ring starts zero).

## Cross-refs

- Trainer that drives these: [`../training/phase_controller.hpp`](../training/phase_controller.hpp).
- Implementations: [`../../../modules/`](../../../modules/).
