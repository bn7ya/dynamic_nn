# CLAUDE.md — `include/dnn/dynamics/`

## Purpose

The "what to add or remove, and when" half of the dynamic architecture.
`LayerManager` watches per-layer efficiency and decides whether to add
or remove nodes/layers. `HealthMonitor` runs the cancer/alzheimer
pathology scores that gate destructive actions. `TrainableScheduler`
adapts the fraction of trainable nodes over time. None of these
modify weights directly — they only call into `Layer<T>` /
`Network<T>` mutators (which are now soft).

## Files

| File | Role |
|---|---|
| `layer_manager.hpp` | `LayerManager`: `analyze_with_efficiency()`, `execute(decision)`, `add_nodes` / `remove_nodes` / `add_layer` / `remove_layer` (all soft). Hosts the local-max shrink gate via `tick_efficiency` + `EfficiencyTracker`. `[hot]` `[invariant]` |
| `efficiency_tracker.hpp` | `EfficiencyTracker` + `EfficiencyGateConfig`: sliding-window detector of local maxima of E(t) (slope ~ 0, curvature <= 0) used to defer shrink decisions. Pure host-side scalar math. |
| `health_monitor.hpp` | Cancer / Alzheimer scores, "should we allow more growth / removal?" gates. `[hot]` |
| `trainable_scheduler.hpp` | Schedules `Layer::set_trainable_fraction` over training. |

`src/dynamics/` contains:
- `layer_manager.cpp` — explicit instantiations.
- `node_efficiency.cpp` — per-node efficiency aggregation.
- `layer_efficiency.cpp` — per-layer aggregation across nodes.
- `health_monitor.cpp` — score implementations.
- `trainable_scheduler.cpp` — schedule logic.

## Invariants

- **`LayerManager::remove_layer(idx)` is soft.** It calls
  `network_.mark_layer_inactive(idx)`, not `network_.remove_layer(idx)`.
  Don't switch back to dense erasure mid-training. (`layer_manager.hpp`
  remove_layer; root rationale in [core CLAUDE.md](../core/CLAUDE.md).)
- **`add_nodes` / `remove_nodes` clamp by `min/max_nodes_per_layer_`.**
  Don't let a decision pass that would shrink below `min_nodes_per_layer_`.
  Same for `min_layers_` / `max_layers_`.
- **All mutations go through `LayerManager::execute(decision)`.** That's
  the chokepoint where the topology lock would be acquired in the
  concurrent runtime path. Don't expose paths that bypass it.
- **`HealthMonitor::allow_*` gates are advisory, not enforcement.**
  The manager calls them and skips the action when they return false.
  Honour the contract; don't bypass to "force" a mutation.
- **Shrink decisions go through the local-max gate.** When
  `LayerManagerConfig::shrink_requires_plateau == true` (default),
  `analyze_with_efficiency()` defers `RemoveNodes` / `RemoveLayer`
  decisions to `Action::None` unless the gate reports
  `is_at_local_max()` — i.e. the efficiency time-series E(t) has a
  flat slope (`|dE/dt| <= plateau_slope_epsilon`) and non-positive
  curvature (`d2E/dt2 <= 0`) AND a warmup
  (`min_epochs_before_shrink`) plus cooldown
  (`shrink_cooldown_epochs`) have elapsed. The trainer must call
  `LayerManager::tick_efficiency(eff)` once per epoch before the
  decision call so the gate window stays current. `execute()`
  notifies the gate on `RemoveNodes` / `RemoveLayer` so cooldowns
  reset. Both `train_phased` and `train_phased_runtime` are wired;
  preserve that. Setting `shrink_requires_plateau=false` recovers
  the pre-gate behaviour bit-for-bit (documented exception, same
  pattern as `dynamic_thresholds`).

## Maintenance notes

- `compute_layer_similarity(i, j)` is now functional: linear CKA of
  the two layers' linear responses (`Wx+b`) to a fixed deterministic
  random probe, cached per `(i, j, topology_version)`. It computes the
  response directly from `weights()`/`biases()` and never calls
  `forward()` — `forward()` records per-node activation metrics and
  would pollute the efficiency state these decisions read. Two
  same-size/same-efficiency but functionally different layers now
  score low and are no longer collapsed. `LayerManager::linear_cka`
  is a public test hook.
- `HealthMonitor`'s rolling window is now a constructor parameter
  surfaced as `TrainerConfig::health_history_window` (default 50).
  Raise it for long Phase-3 runs so slow chronic growth stays visible.
- `analyze_with_efficiency(threshold)` returns a single
  `LayerDecision`. If multiple layers want to mutate in the same
  epoch, only the highest-priority one fires. This is intentional
  (one structural change per epoch) — preserve it; the trainer
  assumes it.
- The `cancer_threshold` / `alzheimer_threshold` defaults (0.7 each)
  are now also surfaced as `RuntimeAdaptiveConfig::cancer_threshold`
  / `alzheimer_threshold` (see
  [runtime CLAUDE.md](../training/runtime/CLAUDE.md)). Until the
  runtime is wired into the legacy path, the static config values
  in `TrainerConfig` are still read directly.
- `LayerManager` now has a 3-arg constructor accepting a
  `LayerManagerConfig` (efficiency / saturation / redundancy
  thresholds). The 2-arg constructor delegates to the 3-arg one with
  defaults so existing call sites stay byte-identical. The dynamic-
  thresholds Python layer writes derived values into
  `TrainerConfig::layer_manager_config`, which `Trainer::Trainer`
  forwards into the LayerManager. The 3-arg constructor validates the
  config: each threshold must lie in `[0, 1]` and
  `efficiency_threshold` must be strictly less than
  `saturation_threshold` (otherwise the add-vs-remove decision logic
  collapses); violations throw `InvalidArgumentException`. Sigmoid
  shape constants (`sigmoid_k_`, `sigmoid_base_`, `sigmoid_range_`)
  intentionally remain hardcoded — they're functional-form parameters
  documented inline, not decision thresholds.
- `TrainableScheduler` schedules trainable fractions across layers
  but doesn't yet co-ordinate with `active_mask_` from
  [core](../core/CLAUDE.md). A node can be both inactive (mask=0)
  and "trainable" (`Node::is_trainable()`); the gating in
  `apply_gradients` makes this safe — inactive wins. Don't try to
  unify the flags without preserving that order.

## Memory & reliability notes

- `LayerDecision` is a small POD; cheap to pass by value. Keep it
  that way — don't grow it with vectors that allocate.
- `health_monitor_.record_change()` is called on every successful
  mutation. The history grows with epoch count; if you add modes
  with very long runs, cap the rolling window.
- `TrainableScheduler` reads `Layer::nodes()` and ranks by
  efficiency. That's an O(N log N) sort per scheduler call. Don't
  call it per-batch — it's intended per-epoch or rarer.

## Cross-refs

- Mutations land in [`include/dnn/core/CLAUDE.md`](../core/CLAUDE.md)
  via soft `add_nodes` / `remove_nodes` / `mark_layer_inactive`.
- The scheduler interacts with the trainer's adaptive config
  ([`include/dnn/training/runtime/CLAUDE.md`](../training/runtime/CLAUDE.md))
  and the `Trainer` itself ([`include/dnn/training/CLAUDE.md`](../training/CLAUDE.md)).
- Final hard-erasure of inactive nodes/layers is
  `Network::compact()` — see [core CLAUDE.md](../core/CLAUDE.md).

## Updating this file

When you add a new mutation type to `LayerDecision` or a new health
score to `HealthMonitor`, append a row / paragraph as appropriate. If
soft-delete is ever extended to layer-level skip-forward (currently
deferred — layer stays in the chain), update the invariant about
`mark_layer_inactive`. Keep this file ≤180 lines.
