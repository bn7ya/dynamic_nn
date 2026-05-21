# CLAUDE.md — `csrc/include/enn/training/`

## Purpose

The four-phase training pipeline and the data-driven config
derivation that replaces the legacy `MAPPING_TABLE`.

`PhaseController` is the C++ counterpart of the loop a PyTorch user
would write by hand: it constructs `torch::optim::Adam` per phase
(LR / epoch count come from `derive_phase_schedule(stats)`), runs
the standard `zero_grad` → `forward` → `cross_entropy` /
`mse_loss` → `backward` → `step` cycle, and pushes the per-epoch
cost and utilization into trajectories. Optimizer state lives inside
libtorch's standard machinery; we never reimplement the update rule.
`compute_dataset_statistics(X, y)` is itself built on libtorch ops —
`torch::var`, `at::linalg_svd` for the effective-rank entropy,
`torch::nn::functional::log_softmax` for the label-entropy
denominator.

## Files

| File | Role |
|---|---|
| `data_driven_config.hpp` | `DatasetStatistics` + `compute_dataset_statistics(X, y)` (variance / SNR / label entropy / effective rank / Fisher separability) + `derive_*` functions that populate `AdaptiveLRConfig`, `StabilityConfig`, `PruningConfig`, `PhaseScheduleConfig`, `PlateauConfig`. **Every derivation is a cited formula**; no bare multiplicative constants. |
| `phase_controller.hpp` | `PhaseController(net, cost)` runs the four phases: `topology_discovery`, `convergence_rate_estimation`, `adaptive_training`, `frozen_architecture_finetuning`. `fit(X, y)` orchestrates them and returns `TrainingResult`. |
| `batch_manager.hpp` | Mini-batch sizing with growth-rate schedule. |
| `early_stopping.hpp` | Patience-based + regression-based stopping. |

## Invariants

- **No magic numbers.** All thresholds enter via the structs in
  `data_driven_config.hpp` and get populated from
  `DatasetStatistics` at the start of `PhaseController::fit`.
- **Phase order is fixed.**
  TopologyDiscovery → ConvergenceRateEstimation → AdaptiveTraining
  → FrozenArchitectureFinetuning. Each phase pushes to the cost /
  utilization / stability_state trajectories.
- **`fit()` always calls `net.compact()` once at the end.** This
  is what makes soft-pruning actually save FLOPs at inference.
- **Optimizers come from `torch::optim::*`.** Do not reimplement SGD,
  Adam, RMSprop.
- **History trajectories cap at `kMaxHistory = 1024`** in
  `phase_controller.cpp::push_capped`.

## Cross-refs

- Controllers consumed: [`../controllers/CLAUDE.md`](../controllers/CLAUDE.md).
- Modules trained: [`../modules/CLAUDE.md`](../modules/CLAUDE.md).
- Implementations: [`../../../training/`](../../../training/).
