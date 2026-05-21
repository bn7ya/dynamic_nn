# CLAUDE.md — `csrc/`

## Purpose

C++17 sources for the `_enn_core` PyTorch extension. Every numeric op
routes through libtorch (`torch::Tensor`, `torch::nn::functional`,
`torch::optim::*`); we do not reinvent matmul, autograd, or
optimizers.

These translation units build into a single libtorch extension
(`_enn_core.so`) via `torch.utils.cpp_extension.CUDAExtension` — or
`CppExtension` when nvcc is absent — driven by `setup.py` at the repo
root. The pybind11 layer in `bindings/` exposes the
`torch::nn::Module` subclasses so Python sees them as ordinary
`torch.nn.Module` instances usable with `torch.optim.*`,
`torch.utils.data.DataLoader`, `state_dict()`, and `.to(device)`.

## Layout

| Path | Role |
|---|---|
| `include/enn/controllers/` | Header-only public surface of the controllers (PlateauDetector, StabilityMonitor, AdaptiveLR, LayerManager, TrainableScheduler). |
| `include/enn/modules/` | torch::nn::Module subclasses (ReversibleLinear, ReversibleNetwork, AdaptiveConv2d, NodeMetrics). |
| `include/enn/training/` | PhaseController, BatchManager, EarlyStopping, DataDrivenConfig. |
| `controllers/` | Controller implementations. |
| `modules/` | Module implementations. |
| `training/` | Training pipeline + data-driven derivation. |
| `bindings/` | pybind11 surface (`module.cpp` registers the three submodules). |
| `ops/` | Reserved for novel CUDA/CPU ops (none yet — every existing op routes through libtorch). |

## Invariants

- **Reversible topology (soft-delete).** `ReversibleLinear::prune_nodes`
  flips `active_mask` entries to zero; weights are preserved. Only
  `compact()` performs hard erasure.
- **`topology_version()` is strictly monotonic on every mutation
  call**, including no-op `compact()` calls.
- **All allocations go through libtorch's caching allocator.** No
  parallel pools.
- **No magic numbers in code.** Thresholds are populated either by
  `csrc/training/data_driven_config.cpp` from `DatasetStatistics`, or
  they live in a typed config struct whose docstring explains why
  the value cannot be derived (e.g. `min_layers = 2` is a structural
  network invariant, not a tunable).
- **GIL release.** Long-running C++ calls release the GIL in
  `bindings/training.cpp` (`PhaseController::fit`).

## Adding new ops

If you need a custom CUDA kernel:

1. First check whether `torch::nn::functional::*` or `torch::*`
   already does it. If yes, use that.
2. If not, the kernel goes in `ops/{name}.cpp` (CPU) and
   `ops/{name}_cuda.cu` (GPU). Register the new symbol in
   `bindings/module.cpp` only after a microbenchmark in `microbench/`
   shows the libtorch alternative is too slow on the target shape.

## Cross-refs

- Python wrappers: [`elasticneuralnetwork/CLAUDE.md`](../elasticneuralnetwork/CLAUDE.md).
- Tests: [`tests/cpp/`](../tests/cpp/) (gtest) and
  [`tests/python/`](../tests/python/) (pytest).
- Top-level overview: [`CLAUDE.md`](../CLAUDE.md).
