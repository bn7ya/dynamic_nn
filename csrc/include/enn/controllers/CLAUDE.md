# CLAUDE.md — `csrc/include/enn/controllers/`

## Purpose

Headers for the stateless / scalar-only controllers. They consume
metric histories and emit decisions; they never mutate the
ReversibleNetwork directly (LayerManager is the only one with a
mutating `execute(decision)` method, and it goes through the public
network surface).

## Files

| File | Role |
|---|---|
| `plateau_detector.hpp` | Slope/curvature gate on a rolling window. `[invariant]` `is_at_local_max()` requires warmup + cooldown + flat slope. |
| `stability_monitor.hpp` | Replacement for `HealthMonitor`. Scores growth-anomaly and capacity-loss; states are `Stable`, `ExcessiveGrowthRisk`, `PathologicalGrowth`, `ExcessivePruningRisk`, `PathologicalPruning`, `Critical`. |
| `adaptive_lr_controller.hpp` | `AdaptiveLRState` (`improvement_signals`, `regression_signals`) + `step_adaptive_lr_controller` (reward / penalty / reset). |
| `layer_manager.hpp` | Wraps the network and stability monitor; emits `LayerDecision`s (add/remove layer or nodes) based on utilization. CKA via `LayerManager::linear_cka`. |
| `trainable_scheduler.hpp` | Scheduling of trainable-fraction over epochs (linear / exponential / cosine / step). |

## Invariants

- **State containers cap at `kHistoryWindow = 500`** for the
  AdaptiveLR deques. Older entries fall off the front.
- **Stability scoring is read-only**: it never adjusts thresholds or
  network state. It only inspects topology counts and the recorded
  change history.
- **Renames are load-bearing.** No anthropomorphic naming may
  reappear (`cancer_score`, `alzheimer_score`, `EmotionalState`,
  `depression_ratio`, `excitement_ratio`).

## Cross-refs

- Population of these configs: [`../training/data_driven_config.hpp`](../training/data_driven_config.hpp).
- Implementations: [`../../../controllers/`](../../../controllers/).
