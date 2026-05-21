# CLAUDE.md — `elasticneuralnetwork/`

## Purpose

The user-facing Python package. Re-exports the C++ types from
`_enn_core` and adds an `ElasticNetwork` convenience class that
flattens images, dispatches to `PhaseController`, and matches the
shape of the old `pydnn.DynamicNetwork` public surface (with the
spec's renamed APIs).

## Files

| File | Role |
|---|---|
| `__init__.py` | Public exports. Imports `torch` first so libtorch is preloaded before `_enn_core`. |
| `controllers.py` | Re-exports `_enn_core.controllers`; adds the `AdaptiveLRController` Python convenience wrapper. |
| `modules.py` | Re-exports modules; defines `ElasticNetwork(input_shape, output_size, seed, hidden_depth)`. |
| `training.py` | Re-exports `_enn_core.training` (DatasetStatistics, derive_* helpers, PhaseController, configs). |
| `benchmarks/` | Five-dataset benchmark suite (MNIST, CIFAR-10, UCI Adult, Covertype, Higgs). See [`benchmarks/CLAUDE.md`](benchmarks/CLAUDE.md). |

## Invariants

- **`ElasticNetwork.fit(X, y)` returns a `TrainingResult`** with
  `cost_trajectory`, `utilization_trajectory`,
  `stability_state_trajectory`, `topology_version_final`,
  `parameter_count_final`.
- **`predict(X)` flattens image inputs by default.** Pass a 2-D
  tensor for tabular data.
- **`torch` is imported in `__init__.py` before `_enn_core`.** This
  preloads `libtorch.so` / `libc10.so`; without it the extension
  fails to resolve `c10::*` symbols on systems where torch's lib
  directory is not on `LD_LIBRARY_PATH`.
- **No anthropomorphic names re-exported.** The renamed APIs from
  the project plan (PlateauDetector / StabilityMonitor /
  AdaptiveLRController etc.) are the only ones exposed.

## Cross-refs

- C++ side: [`../csrc/CLAUDE.md`](../csrc/CLAUDE.md).
- Benchmark CLI: [`benchmarks/CLAUDE.md`](benchmarks/CLAUDE.md).
- Top-level overview: [`../CLAUDE.md`](../CLAUDE.md).
