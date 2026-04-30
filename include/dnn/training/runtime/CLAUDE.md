# CLAUDE.md — `include/dnn/training/runtime/`

## Purpose

The concurrent stage-worker runtime. Activated via
`TrainerConfig::runtime_enabled = true` (or the Python kwarg
`DynamicNetwork(runtime_enabled=True)`), this subsystem replaces the
strictly sequential phase loop with a controller-driven pipeline:
- four phases that share weights, run in time-sequence;
- a parallel **Estimation observer** thread that drains a metrics ring
  buffer and flags cost-trend stalls;
- topology mutations gated by a shared/exclusive lock so soft-delete
  semantics hold under concurrency;
- adaptive hyperparameter scalars the controller can nudge per epoch.

The legacy sequential body in `trainer.hpp` is preserved for
`runtime_enabled=False` (the default).

## Files

| File | Role |
|---|---|
| `adaptive_config.hpp` | `AdaptiveScalar` (atomic, clamped) + `RuntimeAdaptiveConfig` aggregating ~30 formerly-static learning constants. Also `apply_static_config(TrainerConfig)` (defined inline in `../trainer.hpp`) so caller-set values seed the adaptive scalars. `[invariant]` defaults. |
| `metrics_bus.hpp` | `MetricsBus`: bounded ring buffer of `MetricSample` published per-epoch by stages, drained by the controller / observer. Mutex-guarded. |
| `topology_lock.hpp` | `TopologyLock`: thin wrapper over `std::shared_mutex`. `read_lock()` for forward/backward; `write_lock()` for soft mutations. `[invariant]` |
| `stage_worker.hpp` | `StageWorker` long-lived-thread base with Standby/Active/Finished state machine. Used by the parallel observer; reserved for richer multi-thread futures. |
| `stage_controller.hpp` | `StageController`: drives stages via `run_stage(id, step, should_continue, max)`, owns the observer thread, exposes `rewind_requested()` / `clear_rewind()`. |

`src/training/runtime/runtime_compile_check.cpp` forces template
instantiation so latent type errors surface at link time.

## Invariants

- **The observer never mutates training state.** It only reads
  `MetricsBus`, computes a rolling improvement, and (a) nudges
  scalars in `RuntimeAdaptiveConfig`, (b) raises `rewind_` / sets
  `rewind_target_`. Don't introduce paths where the observer
  reaches into worker state directly. (`stage_controller.hpp`
  `observer_loop`.)
- **`AdaptiveScalar` writes come from a single controller thread.**
  Reads can come from any stage worker via `current()` (atomic).
  Don't write from multiple threads — the type intentionally allows
  it but the contract is single-writer for predictability.
- **`AdaptiveScalar::set/scale/nudge` clamp to `[min, max]`.** Don't
  add an unclamped setter; out-of-range values cascade into kernels
  that assume sanity (e.g. negative LR).
- **Soft topology gates are mandatory.** Inside a `run_stage` step
  body, any topology mutation must take `topology_lock_.write_lock()`
  before calling `LayerManager::execute(...)`. The shared lock is
  taken automatically by `run_stage` around the step. See
  `trainer.hpp` `train_phased_runtime` for the canonical pattern.
- **One rewind per `run` call by default.** The legacy invariant from
  `trainer.hpp:868` (`max_rewinds = 1`). Increasing it is fine, but
  it must remain bounded — otherwise a flapping observer can stall
  training forever.
- **`MetricsBus::publish` is single-producer per stage, MPSC overall.**
  The controller thread and observer thread are both readers. The
  ring is mutex-guarded; that's intentional and adequate for ~1
  publish per epoch.

## Maintenance notes

- The observer's stall heuristic is intentionally simple
  (`stage_controller.hpp` `kStallEpochs=12`, `kStallDelta=1e-5`).
  Replacing with Welford / Page–Hinkley / a learned change-point
  detector should drop into `observer_loop()` without touching
  callers — that's the design goal.
- The Tier-3 dynamic-thresholds nudges (also in `observer_loop`,
  guarded by `kTrendWindow=8`) refine the dataset-derived seeds set
  by `apply_static_config`. They only see the cost/efficiency stream
  on the bus — the controller stays type-agnostic, no Network
  reference. Per-iteration nudge magnitude is bounded (~1% of each
  scalar's range); `AdaptiveScalar::nudge`/`scale` clamp the result.
- `RuntimeAdaptiveConfig::apply_static_config(const TrainerConfig&)`
  is declared in `adaptive_config.hpp` (with `TrainerConfig`
  forward-declared) and defined inline in `../trainer.hpp` where
  `TrainerConfig` is fully visible. It first calls
  `reset_to_defaults()` and then overrides each scalar that has a
  matching field in `TrainerConfig` (cancer/alzheimer/patience/
  min_improvement/gradient_clip/batch/reward_penalty). With unmodified
  TrainerConfig defaults this produces the same scalar values as
  `reset_to_defaults()` did — that's the regression baseline.
- `AdaptiveScalar` is non-copyable / non-movable (atomic field).
  Don't add value-style assignment to `RuntimeAdaptiveConfig`; use
  per-field `set()` like `reset_to_defaults()` does.
- `StageWorker` (the long-lived thread base) is only used by the
  observer today. The original plan was to wrap each phase in its
  own worker thread; it's still feasible and the API is ready, but
  the current `train_phased_runtime` runs phases in the calling
  thread and uses workers only for the parallel observer.
- The CQRS bus in [`include/dnn/cqrs/CLAUDE.md`](../../cqrs/CLAUDE.md)
  is the planned dispatch backbone for this runtime — promotion is a
  follow-up.

## Memory & reliability notes

- `MetricsBus` capacity defaults to 2048 in `train_phased_runtime`.
  At 1 sample per epoch per active stage, that's ample for typical
  runs. If you increase it, remember each `MetricSample` is ~80
  bytes.
- `RuntimeAdaptiveConfig` is stack-allocated inside
  `train_phased_runtime`. Don't promote it to a member without also
  thinking about thread-safety across concurrent `train_phased`
  calls (which today are serialised through the trainer).
- The observer thread is **owned by the controller** and joined in
  the controller's destructor. `stop_observer()` is also called
  explicitly at the end of `train_phased_runtime`. Don't add
  exit paths that skip it; a leaked thread keeps the process alive.
- `TopologyLock`'s shared mutex permits concurrent readers but
  one writer. Long-running write holds will starve readers. Keep
  topology mutations short — they're `O(active_mask_.size())` for
  soft mutations, which is fine.

## Cross-refs

- The trainer route that invokes this is in
  [`include/dnn/training/CLAUDE.md`](../CLAUDE.md)
  (`train_phased_runtime`).
- Soft-topology semantics this runtime relies on:
  [`include/dnn/core/CLAUDE.md`](../../core/CLAUDE.md).
- The Python parallel observer that mirrors this on the
  pure-Python fallback:
  [`python/pydnn/CLAUDE.md`](../../../../python/pydnn/CLAUDE.md).
- Dormant CQRS infrastructure that may eventually back this:
  [`include/dnn/cqrs/CLAUDE.md`](../../cqrs/CLAUDE.md).

## Updating this file

When you add a new `AdaptiveScalar` field, document its default,
clamp, and feedback rule. When you change the rewind policy or the
observer's stall heuristic, revise the matching **Invariants** /
**Maintenance** entries in the same commit. If the runtime is ever
promoted to be the default (`runtime_enabled=true` baseline), update
the **Invariants** in [`../CLAUDE.md`](../CLAUDE.md). Keep this
file ≤220 lines.
