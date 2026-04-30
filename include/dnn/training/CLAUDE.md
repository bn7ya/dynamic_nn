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
- `BatchManager`'s adaptive growth is gated by epoch and efficiency
  thresholds. If you add a new growth trigger, do it inside
  `BatchManager` so both training paths get it for free.
- `EarlyStopping` is currently used only by the legacy path. The
  runtime path embeds an inline early-stop check inside
  `train_phased_runtime`'s should-continue lambda. Keep them in sync;
  diverging them silently is a recipe for confusing regressions.

## Memory & reliability notes

- Every `result.<name>_history` push happens inside the per-epoch
  step; with `reserve(500)` they grow but don't reallocate up to 500
  epochs. If you uncap the runtime's main-stage epoch budget,
  reservation should grow to match or you'll thrash the heap.
  `trainer.hpp:357-361`.
- `EmotionalState::reward_history` / `penalty_history` /
  `depression_history` / `excitement_history` are `std::deque`s.
  They're appended every epoch in Phase 3. If you allow indefinite
  Phase 3 epochs, add a `pop_front()` cap or memory will grow
  linearly with epoch count. (`emotional_state.hpp:18-56`.)
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
