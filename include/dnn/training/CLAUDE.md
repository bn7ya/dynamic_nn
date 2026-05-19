# CLAUDE.md — `include/dnn/training/`

## Purpose

The training subsystem. Hosts `Trainer<T>` (orchestrator), the optimizer
family, batch sizing, cost functions, early stopping, the reward/penalty
+ emotional-state machinery, and the four-phase pipeline that drives
the network's growth and pruning. The new concurrent runtime sits
under [`runtime/`](runtime/CLAUDE.md); this directory holds the legacy
sequential body and the helpers shared between both paths.

## Files

| File | Role |
|---|---|
| `trainer.hpp` | `Trainer<T>` + `TrainerConfig` + `TrainingResult`; both `train_phased()` (legacy) and `train_phased_runtime()` (concurrent). `[hot]` `[invariant]` |
| `optimizer.hpp` | SGD / Momentum / Adam / RMSprop. `OptimizerConfig` defaults. |
| `batch_manager.hpp` | `BatchManager`: starts at `min_batch_size`, grows by `growth_rate` every `growth_interval_epochs`. |
| `cost_functions.hpp` | MSE, MAE, CrossEntropy, BinaryCrossEntropy, Huber, LogCosh, KLDivergence, CosineSimilarity. `[invariant]` external API. |
| `early_stopping.hpp` | `EarlyStopping`: patience + min-improvement + window-based plateau detection. |
| `emotional_state.hpp` | `EmotionalState` + `RewardPenaltyConfig` + `apply_reward_penalty_system`. `[invariant]` Phase 3 LR adjustment. |
| `train_strategy.hpp` | `TrainStrategy<T>` extraction interface (3.12). `[scaffolding]` — interface only, intentionally not wired (see note below). |
| `runtime/` | Concurrent stage runtime — separate sub-feature with its own [CLAUDE.md](runtime/CLAUDE.md). |

`src/training/` provides the corresponding `.cpp` files plus
`runtime/runtime_compile_check.cpp`.

## Invariants

- **`Trainer<T>::train_phased()` is the public entry.** It dispatches
  to `train_phased_runtime()` when `config_.runtime_enabled == true`,
  otherwise runs the legacy sequential body. Both paths must return
  a populated `TrainingResult` with the same field semantics.
  (`trainer.hpp:352-355` for the dispatch.)
- **`runtime_enabled=False` is the default.** Don't flip without a
  feature-flag deprecation cycle. The legacy body is the
  reproducibility baseline for every existing notebook and saved
  model. (`trainer.hpp` `TrainerConfig::runtime_enabled = false`.)
- **The four phases stay phase 1→4 in name and order.** Even under
  the runtime path. Stage 2 may run *in parallel* as an observer,
  but the schedule is still Exploration → Estimation → Main →
  Standard with rewinds. Don't reorder.
- **`apply_reward_penalty_system` is shared.** Both paths call it.
  Its signature (`emotional_state.hpp:363`) must not change without
  updating both `train_phased` and `train_phased_runtime`.
- **`TrainerConfig` is a public surface.** All fields are bound to
  Python via `_bindings.cpp:349`. Adding a field is fine; renaming
  or removing one is a breaking change. New nested config sub-structs
  (`LayerManagerConfig`, `RewardPenaltyConfig`, `BatchConfig`) are
  also bound and writable from Python; the dynamic-thresholds Python
  layer (`python/pydnn/dynamic_thresholds.py`) writes into them
  before `Trainer` construction.
- **`min_epochs_for_early_stop` is a TrainerConfig field**
  (default 10). It used to be a hardcoded literal at the
  `EarlyStopping` constructor call site (`trainer.hpp:208`); it is
  now threaded through. Don't reintroduce the literal.
- **`batched_train_forward` is a TrainerConfig field, default `true`.**
  When true, `Trainer::train_epoch` issues one rank-2 forward and one
  rank-2 backward per batch instead of the legacy B rank-1 calls.
  The rank-2 path was already implemented in `Layer::forward` /
  `backward` (CPU and CUDA). Numerical equivalence to the per-sample
  loop holds modulo floating-point summation order. On CUDA the
  difference is dramatic — one `cuda_gemm` per layer instead of B
  `cuda_gemv`s collapses the per-sample CPU↔GPU round-trip cost.
  Don't flip the default without flagging it in this file. The
  legacy per-sample loop stays reachable via the same field for
  benchmarking and regression bisection.
- **`TrainingResult` field set is contractual.** Python's
  `_fit_cpp` reads specific attributes (`success`, `epochs_completed`,
  `final_cost`, `final_efficiency`, `best_cost`, `best_efficiency`,
  `stopping_reason`, `cost_history`, `efficiency_history`,
  `training_time`). Don't drop those.

## Maintenance notes

- The legacy `train_phased` body (lines ~366–688 in `trainer.hpp`)
  contains 30+ formerly-static numeric constants. The runtime path
  reads them from `RuntimeAdaptiveConfig` instead. When a constant
  changes, update *both* — or, preferably, route the legacy path
  through the same `RuntimeAdaptiveConfig` defaults.
- `train_phased_runtime` currently allows **one** rewind from Phase 3
  back to Estimation. If you want richer rewind policies (multiple
  rewinds, Phase 1 reactivation), the place is
  `trainer.hpp` `train_phased_runtime` and the observer in
  [runtime/CLAUDE.md](runtime/CLAUDE.md).
- `Optimizer<T>::resize(weight_size, bias_size)` preserves momentum/
  variance for the surviving parameter prefix and zero-fills growth
  (new params get no history — correct). SGD is a no-op (stateless).
  `Trainer::apply_gradients_step` calls `resize` per layer when the
  topology version changes but the layer count is unchanged (a layer
  grew via `add_nodes`); a layer-count change still triggers a full
  rebuild. Adam's `timestep_` (bias correction) is intentionally kept
  across resize.
- The optimizer family is now wired. `Trainer::apply_gradients_step()`
  replaces the direct `network_.apply_gradients(lr)` call in
  `train_epoch`. Default `TrainerConfig::use_optimizer=false` keeps the
  legacy inline-SGD path bit-for-bit. When true, each CPU layer's
  update routes through a per-layer `Optimizer<T>` built from
  `optimizer_config` (the optimizer's own clipping is disabled since
  `clip_gradients()` runs first). CUDA layers keep the on-device legacy
  step (device-path optimizer wiring is a CUDA-host follow-up). Per-
  layer optimizer state is rebuilt on `topology_version_` change (1.11
  adds overlap-preserving resize). New `TrainerConfig` fields
  `use_optimizer` / `optimizer_config` are additive — add the matching
  `_bindings.cpp` rows in a pybind-host session.
- `Trainer::apply_random_perturbation(fraction)` is now implemented
  (was a no-op that still incremented `result.perturbations_applied`).
  It picks `max(1, total_nodes*fraction)` random (layer, active-node)
  pairs via a Trainer-owned `perturb_rng_` (seeded from
  `network.config().seed`, deterministic) and adds zero-mean Gaussian
  noise (stddev `TrainerConfig::perturbation_scale`, default 0.01) to
  each chosen node's weight row, then marks those layers' GPU mirrors
  dirty. New field `perturbation_scale` is additive on `TrainerConfig`
  — add the matching row in `_bindings.cpp` in a pybind-host session.
  Phase 3 cost trajectory now genuinely changes when perturbation
  fires (expected, Tier 1).
- 3.12 (`TrainStrategy` extraction) is intentionally interface-only
  (`train_strategy.hpp`, `[scaffolding]`). `train`/`train_phased` are
  NOT rewired: `train_phased` is the bit-for-bit reproducibility
  baseline and parity can only be proven via the end-to-end
  Python/`_dnn_core` cost-trajectory smoke, which is unbuildable here
  (no pybind11). Wiring it unverified would risk a silent
  reproducibility regression. The header documents the safe path for a
  pybind/CUDA-capable session.
- There is now a single `compute_efficiency` definition — the
  windowed-sigmoid free function in `emotional_state.hpp`. The
  trainer's local 2-point member was deleted (3.2); unqualified calls
  resolve to the free function. Efficiency values (hence some
  decisions) differ from the old member formula — Tier-3 behaviour
  change, record a fresh baseline if bisecting.
- `apply_static_config` now also maps `phase4_lr_decay_rate`,
  `phase4_lr_decay_interval`, `phase4_batch_size`,
  `layer_adjustment_interval`, `enable_dynamic_layers` (0/1) into new
  `RuntimeAdaptiveConfig` `AdaptiveScalar`s (3.7) — previously silently
  dropped on the runtime path. New `TrainerConfig::health_history_window`
  (default 50) is additive — flag `_bindings.cpp` for pybind host.
- `Trainer::evaluate` now uses the batched rank-2 forward when
  `batched_train_forward` is true (no backward), matching `train_epoch`.
  Numerically equivalent to the per-sample loop modulo matmul summation
  order; the per-sample path stays reachable via the same flag.
- `compute_normalization_params` is single-pass Welford (mean + M2 +
  min/max in one traversal) instead of two passes. Same std values
  within fp tolerance, numerically stabler.
- `TrainerConfig::async_callback` (default false) runs the epoch
  callback on a background thread against a `network_.clone()` with
  single-slot drop semantics; `~Trainer()` joins any in-flight
  callback. Default false keeps the synchronous live-network behaviour
  bit-for-bit. Additive field — flag `_bindings.cpp` for pybind host.
- 2.3 (forward_cpu weights_t per-forward transpose) was intentionally
  NOT cached: the optimizer mutates `weights_` through the public
  `weights()` accessor, so no internal version counter can correctly
  invalidate a cached transpose, and changing the `SIMDOps::matmul`
  virtual interface risks Tier-2 numerical identity across SIMD
  backends. Left as-is for correctness.
- `Trainer::clip_gradients()` is now implemented (was a no-op). It
  iterates `network_.layers()` and calls `Layer::clip_gradients(value)`,
  which clamps the accumulated gradients in place to
  `[-gradient_clip_value, +gradient_clip_value]` — the GPU gradient
  mirrors on the CUDA path, the host accumulators otherwise. It runs
  after backward and before `apply_gradients`. `enable_gradient_clipping`
  defaults to `true`, so training output changed vs the old no-op
  baseline; record a fresh reference if you bisect on cost trajectory.
- `BatchManager`'s adaptive growth is gated by epoch and efficiency
  thresholds. If you add a new growth trigger, do it inside
  `BatchManager` so both training paths get it for free.
- `EarlyStopping` is currently used only by the legacy path. The
  runtime path embeds an inline early-stop check inside
  `train_phased_runtime`'s should-continue lambda. Keep them in sync;
  diverging them silently is a recipe for confusing regressions.
- Both training entry points (`train_phased` and
  `train_phased_runtime`) construct a `TrainingInProgressGuard`
  (defined as a private struct on `Trainer<T>`) at scope start.
  The guard flips `Network<T>::training_in_progress_` and clears it
  via RAII so `Network<T>::compact()` refuses to run mid-training
  even on exception paths. Mirror this guard in any new top-level
  training entry point.

## Memory & reliability notes

- Every `result.<name>_history` push happens inside the per-epoch
  step. Reservation goes through `Trainer::reserve_histories` with
  `Trainer::kDefaultHistoryReserve = 500`; the runtime path bumps
  the cap to `max(500, max_epochs)` so long opt-in runs don't thrash
  the heap. If you raise `max_epochs` past 500 on the legacy path,
  push the same change here.
- `EmotionalState::reward_history` / `penalty_history` /
  `depression_history` / `excitement_history` are `std::deque`s
  capped at `EmotionalState::kHistoryWindow = 500` via the
  `push_capped` helper. They're appended every epoch in Phase 3 and
  drop oldest-first when full, so memory stays bounded under
  indefinite Phase 3 epochs. The ratios use the running totals
  (`total_rewards` / `total_penalties`), not the histories, so they
  remain stable across the cap. (`emotional_state.hpp:18-56`.)
- The optimizer holds momentum / Adam state proportional to the
  number of parameters. Architecture growth means re-allocating that
  state. The current code recreates it lazily; if you add a new
  optimizer, mirror that pattern.
- Cost-function backward passes should not allocate for every batch.
  Reuse buffers as the existing implementations do.

## Cross-refs

- Network/Layer mutators are in
  [`include/dnn/core/CLAUDE.md`](../core/CLAUDE.md).
- Architecture decisions come from
  [`include/dnn/dynamics/CLAUDE.md`](../dynamics/CLAUDE.md).
- Concurrent runtime (StageController, observer, AdaptiveScalar):
  [`runtime/CLAUDE.md`](runtime/CLAUDE.md).
- Public Python surface:
  [`python/pydnn/CLAUDE.md`](../../../python/pydnn/CLAUDE.md).

## Updating this file

When you add a new optimizer, cost function, or training-side
configuration field, append a row to the file table or describe the
new field's defaults in **Maintenance notes**. When you change the
`TrainerConfig` / `TrainingResult` shape, update **Invariants** in the
same commit and check `_bindings.cpp` for the binding change. Keep
this file ≤220 lines.
