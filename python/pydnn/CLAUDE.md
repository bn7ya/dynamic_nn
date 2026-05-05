# CLAUDE.md — `python/pydnn/`

## Purpose

The Python user-facing package. Wraps the C++ core via pybind11
(`_dnn_core`) and adds a Python-side `DynamicNetwork` class that
provides the high-level `fit()` / `predict()` API, configuration
dataclasses, the pure-Python training fallback (`_fit_python`), the
`_CostTrendObserver` that mirrors the C++ runtime's parallel
observer, and `ThreeModelGenerator` for spawning multiple variants.

## Files

| File | Role |
|---|---|
| `__init__.py` | Public exports; tries to import `_dnn_core`, falls back to pure-Python types if unavailable. |
| `_bindings.cpp` | pybind11 wrappers for `Tensor`, `Network`, `Trainer`, `TrainerConfig`, `NetworkConfig`, cost functions, etc. `[invariant]` external surface. |
| `network.py` | `DynamicNetwork` (the user-facing class), `TrainingResult`, all `*Config` dataclasses, `_fit_cpp`, `_fit_python`, `_CostTrendObserver`. `[hot]` `[invariant]` |
| `dynamic_thresholds.py` | `compute_data_signals(X, y)` + `derive_thresholds(...)` + `MAPPING_TABLE`. Engine for the `dynamic_thresholds=True` path. |
| `model_generator.py` | `ThreeModelGenerator`: builds efficient / balanced / accurate variants from one template. |
| `visualization.py` | Matplotlib-based reports (cost curves, architecture history, health). Optional. |
| `transformer/` | Modern Transformer / MoE / LLM building blocks (own autograd, own Module base, own trainer). See [`transformer/CLAUDE.md`](transformer/CLAUDE.md). Pure-Python; does not touch `_dnn_core`. |

## Invariants

- **Public class surface.** `DynamicNetwork`, `TrainingResult`,
  `ThreeModelGenerator`, the various `*Config` dataclasses, and the
  module-level `COST_FUNCTIONS` constant are external API. Add
  fields, don't remove them.
- **`fit()` is forward-compatible with both backends.** When
  `_use_cpp` is true, `_fit_cpp` runs the C++ path; otherwise
  `_fit_python` runs the pure-Python fallback. Both must populate
  `TrainingResult` with the same fields. (`network.py:842-846`.)
- **`_fit_cpp` passes raw numpy arrays to the binding.**
  `PyTrainer::train` accepts `py::array_t<float>`, not lists of
  `Tensor`. Pre-existing API drift bit us once and is fixed in
  commit `dd8ba30`. Don't regress to `[Tensor(x) for x in X]`.
  (`network.py:894` and the binding at
  `_bindings.cpp:136-176`.)
- **`predict()` uses numpy arrays both directions.**
  `PyNetwork::predict` takes `py::array_t<float>` and returns a
  numpy array. Don't wrap inputs in `Tensor` or call `.numpy()` on
  the return. (`network.py:2528-2535` and the binding at
  `_bindings.cpp:69-77`.)
- **`predict_batch()` is the vectorised batch path.**
  `PyNetwork::predict_batch` accepts a 2-D `(N, F)` numpy array
  (any C-castable view; the binding force-casts to contiguous
  float32) and returns `(N, output)` in one call, with the GIL
  released around the per-row forward loop.
  `DynamicNetwork.predict()` prefers it via `hasattr(...)`, falling
  back to the per-sample loop if the binding is too old. Don't
  remove the fallback while older `.so` artefacts are still
  in circulation. (`network.py:2573-2594`,
  `_bindings.cpp` `predict_batch`.)
- **`fit()` calls `network.compact()` after `_fit_cpp` returns.**
  This is what makes soft-pruning actually save FLOPs at inference.
  Don't drop the call. (`network.py:892-901`.)
- **`runtime_enabled` is a constructor kwarg, default `False`.**
  When true, `_fit_cpp` builds a `_dnn_core.TrainerConfig` with
  `runtime_enabled=True`; the pure-Python fallback starts a
  `_CostTrendObserver`. Don't flip the default without flagging it
  in the root [`CLAUDE.md`](../../CLAUDE.md). (`network.py:482-501`.)
- **`dynamic_thresholds` is a constructor kwarg, default `True`.**
  This is the documented exception to the root CLAUDE.md "defaults
  are sticky" rule. When true, `_fit_cpp` calls
  `dynamic_thresholds.compute_data_signals(X, y)` once before
  building TrainerConfig and writes the derived values into the
  matching fields (cancer/alzheimer/patience/etc., plus the new
  `LayerManagerConfig`, `RewardPenaltyConfig`, `BatchConfig`).
  The signals dataclass is stashed on
  `self._last_data_signals` so callers can introspect what the
  variance and complexity scores were and which thresholds they
  produced. Setting `dynamic_thresholds=False` reproduces the
  legacy static-default path **bit-for-bit** — that's the
  regression baseline. C++ backend only; the pure-Python fallback
  ignores the flag.
- **GIL released in the binding around long C++ calls.**
  `PyTrainer::train` does `py::gil_scoped_release release` before
  calling `trainer_.train(...)` (`_bindings.cpp:164`). Preserve
  this on any new long-running binding so Python threads can
  observe progress.

## Maintenance notes

- `_CostTrendObserver` (`network.py:425-489`) is a daemon thread that
  reads `cost_history` (a regular list). Under the GIL both
  `list.append` and the slice `history[-window:]` are atomic — the
  slice creates a fresh list snapshot before any other thread runs —
  so the observer (read-only) and the fit loop (sole writer) need no
  explicit lock. If you ever add a second writer thread, introduce a
  `threading.Lock` in this class first.
- The observer raises `rewind_requested()` once and is cleared by the
  Phase 3 loop; the Python fallback then uses a simple "early break"
  rather than the C++ runtime's rewind-into-Estimation. Keep that
  divergence documented; if you implement true rewind in Python,
  remove the note.
- `_bindings.cpp` is the only place where C++ types touch Python.
  When you add a method on the C++ side, mirror the binding here in
  the same commit. Forgetting that is the most common cause of
  AttributeError in `network.py`.
- `model_generator.py` builds three independent `DynamicNetwork`
  instances. They train sequentially today; truly concurrent
  variant training is the obvious next perf win and was discussed
  in the original plan — see the root [`CLAUDE.md`](../../CLAUDE.md).
- `visualization.py` is optional and shouldn't be a hard dep.
  Imports of matplotlib / seaborn must stay lazy.

## Memory & reliability notes

- `TrainingResult.cost_history` and friends mirror the C++ histories.
  Each is a Python list, capped at module-level `MAX_HISTORY = 500`
  via the `_append_capped(history, value)` helper. Older entries
  drop off the front; the most recent 500 epochs are retained. The
  C++ `Trainer::kDefaultHistoryReserve` and
  `EmotionalState::kHistoryWindow` use the same value so the two
  backends report comparable windows. If you raise the cap, raise
  both sides.
- The pure-Python `_fit_python` allocates fresh numpy arrays per
  epoch in `_train_epoch` for shuffles and gradients. Don't try to
  micro-optimise this — the C++ path is the perf path.
- Numpy arrays passed across the boundary should be
  `np.float32` and `np.ascontiguousarray`. `_fit_cpp` and `predict()`
  both enforce this; do likewise in any new path that crosses the
  boundary.
- `_dnn_core.Network.compact()` may rebuild layers (resetting their
  weights) when fan-in changes. That's acceptable end-of-training,
  but don't call it mid-training from Python.

## Cross-refs

- C++ surface owners: [`include/dnn/training/CLAUDE.md`](../../include/dnn/training/CLAUDE.md)
  and [`include/dnn/core/CLAUDE.md`](../../include/dnn/core/CLAUDE.md).
- The C++ counterpart of `_CostTrendObserver`:
  [`include/dnn/training/runtime/CLAUDE.md`](../../include/dnn/training/runtime/CLAUDE.md).
- The smoke test recipe lives in the root [`CLAUDE.md`](../../CLAUDE.md).

## Updating this file

When you add a new pybind binding, append it to the file table
(or to **Invariants** if it's an external surface). When you add
or rename a `*Config` dataclass field in `network.py`, mention
defaults under **Maintenance notes**. If the C++/Python boundary
contract changes (numpy ↔ Tensor), update **Invariants** in the
same commit. Keep this file ≤220 lines.
