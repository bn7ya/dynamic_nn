# CLAUDE.md — dynamic_nn root

> Maintenance scaffolding for AI sessions. **Read this file first** when
> opening this repo. Per-directory CLAUDE.md files supply the local detail.

## Project overview

dynamic_nn is a neural network that grows and prunes itself during training.
A C++17 core (with optional CUDA + SIMD acceleration) is exposed to Python
via pybind11 as the `pydnn` package. The training pipeline runs four
stages — Exploration / Estimation / Main / Standard — that can be driven
either sequentially (legacy default) or as concurrent stage workers
behind a `StageController` with a parallel cost-trend observer
(`runtime_enabled=True`). Mid-training "remove node" / "remove layer"
flips an active mask without erasing weights; hard removal happens once
at end-of-training via `Network::compact()`.

## How to use these CLAUDE.md files

- **Each directory that owns a feature has its own CLAUDE.md.** It tells
  you what files do what, what's invariant, what's actively maintained,
  and how the directory connects to the rest.
- **Update them in the same commit as code changes.** When you add /
  remove / rename a file, edit the relevant CLAUDE.md so the table
  stays accurate. When you change a contract (e.g. soft-delete
  semantics), revise the invariants section.
- **Tags used in file tables:**
  - `[invariant]` — changing this without careful review breaks
    contracts other code relies on.
  - `[hot]` — runs in the per-batch / per-epoch loop. Watch for
    allocations and quadratic behaviour.
  - `[scaffolding]` — wired in but not yet driving anything; safe to
    extend, but don't delete without checking what depends on it.
  - `[fallback]` — only used when CUDA is off; keep behaviour
    equivalent to the GPU path.

## Maintenance principles

The user's three explicit goals — **stability, reliability, memory
efficiency** — translate into the following operational rules. Every
CLAUDE.md sub-file refers back here.

### Stability
- **Public surfaces don't break.** `DynamicNetwork.fit()`,
  `network.predict()`, `network.compact()`, `TrainerConfig`, and the
  pybind class layouts in `python/pydnn/_bindings.cpp` are external
  contracts. Add fields, don't remove them; add overloads, don't
  rename existing ones.
- **Both build configurations stay green** — `DNN_ENABLE_CUDA=ON` and
  `DNN_ENABLE_CUDA=OFF`. The CPU-only path is the smoke-test floor.
- **Defaults are sticky.** `runtime_enabled=False` keeps the legacy
  sequential `train_phased` body. Don't flip a default without
  flagging it in the relevant CLAUDE.md and the commit message.
- **Branch convention:** all work lands on
  `claude/<topic>-<id>` — never push directly to `main`.

### Reliability
- **Soft-delete topology.** Mid-training mutations *never* erase
  weights. Only `Network::compact()` (called once after training)
  is allowed to do dense erasure. See
  `include/dnn/core/CLAUDE.md`.
- **Gate every parallel write.** Topology mutations under
  `runtime_enabled=True` go through `TopologyLock` in write mode.
  Read-only forward/backward holds it in shared mode. See
  `include/dnn/training/runtime/CLAUDE.md`.
- **Observers don't mutate state they observe.** The Python and C++
  cost-trend observers only *read* metric history; they signal via
  flags, not by reaching into worker state.
- **Fail loudly at boundaries, gracefully inside.** Exceptions live
  at the C++/Python edge (`_bindings.cpp`); internal errors propagate
  as exceptions, not silent skipped epochs.

### Memory efficiency
- **Singleton pools.** All GPU allocations go through
  `cuda::CudaMemoryPool::instance()` (see
  `include/dnn/cuda/CLAUDE.md`). All large CPU allocations go through
  `dnn::memory::PoolAllocator` (see `include/dnn/memory/CLAUDE.md`).
  Don't introduce parallel allocator paths.
- **Compact at end-of-training.** `DynamicNetwork.fit()` calls
  `network.compact()` when the C++ path returns; this is what makes
  soft-pruning actually save FLOPs at inference time. Don't skip it.
- **Bounded histories.** `cost_history`, `efficiency_history`,
  `cancer_score_history`, `alzheimer_score_history`,
  `architecture_history`, `learning_rate_history`,
  `emotional_state_history` all grow per epoch. They have an
  implicit `reserve(500)` and currently no cap. If you add
  long-running modes, **add a window cap** rather than letting them
  grow without bound.
- **GIL released around long C++ work.**
  `python/pydnn/_bindings.cpp` does this in `PyTrainer::train` via
  `py::gil_scoped_release`. Preserve this on any new long-running
  binding.
- **No per-epoch allocations on the hot path** if avoidable. Layer
  forward/backward already reuses `cached_input_`,
  `cached_pre_activation_`, `cached_output_`. Don't replace these
  with per-call temporaries.

## Feature index

Every named feature in the project, the subsystem that owns it, and
how to engage it.

| Feature | Owner | Engage / opt-in |
|---|---|---|
| Soft topology (active masks, no mid-training erase) | [`include/dnn/core/`](include/dnn/core/CLAUDE.md) | Always on; transparent to callers |
| `Network::compact()` end-of-training cleanup | [`include/dnn/core/`](include/dnn/core/CLAUDE.md) | Auto-called from `fit()` |
| Concurrent stage pipeline (StageController + observer) | [`include/dnn/training/runtime/`](include/dnn/training/runtime/CLAUDE.md) | `DynamicNetwork(runtime_enabled=True)` |
| Adaptive learning hyperparameters | [`include/dnn/training/runtime/`](include/dnn/training/runtime/CLAUDE.md) | Active when `runtime_enabled=True` |
| Reward/penalty + emotional state | [`include/dnn/training/`](include/dnn/training/CLAUDE.md) | Always on in Phase 3 |
| Layer/node efficiency tracking | [`include/dnn/dynamics/`](include/dnn/dynamics/CLAUDE.md) | Always on; drives architecture mutation |
| Cancer / Alzheimer health monitoring | [`include/dnn/dynamics/`](include/dnn/dynamics/CLAUDE.md) | Always on; thresholds in `TrainerConfig` |
| SIMD CPU acceleration (AVX/AVX2/AVX512/SSE) | [`include/dnn/simd/`](include/dnn/simd/CLAUDE.md) | Auto-detected at runtime |
| OpenMP parallelism (CPU batch loops, matmul) | [`include/dnn/simd/`](include/dnn/simd/CLAUDE.md) | Compile-time `DNN_HAS_OPENMP` |
| CUDA GPU acceleration | [`include/dnn/cuda/`](include/dnn/cuda/CLAUDE.md) | `device="cuda"` + `DNN_ENABLE_CUDA` |
| Per-tensor CUDA stream binding | [`include/dnn/cuda/`](include/dnn/cuda/CLAUDE.md) | `CudaTensor::set_stream(s)` |
| Memory pooling (CPU + GPU) | [`include/dnn/memory/`](include/dnn/memory/CLAUDE.md), [`include/dnn/cuda/`](include/dnn/cuda/CLAUDE.md) | Always on |
| CQRS command/query dispatch | [`include/dnn/cqrs/`](include/dnn/cqrs/CLAUDE.md) | Scaffolding only — currently dormant |
| Dynamic batch sizing | [`include/dnn/training/`](include/dnn/training/CLAUDE.md) | `BatchConfig` in `TrainerConfig` |
| Early stopping | [`include/dnn/training/`](include/dnn/training/CLAUDE.md) | `enable_early_stopping` |
| Adaptive weight initialisation | [`include/dnn/core/`](include/dnn/core/CLAUDE.md) | Auto on first `fit()` |
| Cost functions | [`include/dnn/training/`](include/dnn/training/CLAUDE.md) | `cost_function="..."` |
| Optimizer types (SGD/Momentum/Adam/RMSprop) | [`include/dnn/training/`](include/dnn/training/CLAUDE.md) | `OptimizerConfig` |
| Architecture mutation (add/remove nodes/layers) | [`include/dnn/dynamics/`](include/dnn/dynamics/CLAUDE.md) | Always on; controlled by efficiency thresholds |
| ThreeModelGenerator (efficient/balanced/accurate) | [`python/pydnn/`](python/pydnn/CLAUDE.md) | `from pydnn import ThreeModelGenerator` |
| CPU/GPU device dispatch | [`include/dnn/core/`](include/dnn/core/CLAUDE.md) | `device="cpu"\|"cuda"` |
| Python parallel cost-trend observer | [`python/pydnn/`](python/pydnn/CLAUDE.md) | `runtime_enabled=True` (Python fallback path) |

## Build & smoke-test recipe

Verified working in this session:

```bash
# CPU-only build (no CUDA, no Python bindings)
rm -rf build_test
cmake -S . -B build_test \
    -DDNN_ENABLE_CUDA=OFF \
    -DDNN_BUILD_PYTHON=OFF \
    -DDNN_BUILD_EXAMPLES=OFF
cmake --build build_test -j

# Python extension build (requires pybind11)
PYBIND11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake -S . -B build_smoke \
    -DDNN_ENABLE_CUDA=OFF \
    -DDNN_BUILD_PYTHON=ON \
    -DDNN_BUILD_EXAMPLES=OFF \
    -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build_smoke -j
cp build_smoke/_dnn_core.cpython-*.so python/pydnn/

# End-to-end smoke test (the C++ runtime path through Python)
python3 -c "
import sys; sys.path.insert(0, 'python')
from pydnn import DynamicNetwork
import numpy as np
rng = np.random.default_rng(0)
X = rng.standard_normal((64, 16)).astype(np.float32)
y = np.eye(4)[rng.integers(0, 4, size=64)].astype(np.float32)
net = DynamicNetwork(input_shape=(16,), output_size=4, seed=42,
                    cost_function='CrossEntropy', runtime_enabled=True)
result = net.fit(X, y, verbose=True)
preds = net.predict(X[:4])
print('OK', result.epochs_completed, preds.shape)
"
```

If the smoke test stops working, the most likely break points are:
the `_fit_cpp` ↔ `_bindings.cpp` API contract (numpy arrays in,
numpy arrays out — see `python/pydnn/CLAUDE.md`), or the
`StageController` observer thread (see
`include/dnn/training/runtime/CLAUDE.md`).

## Hierarchy update protocol

- **Adding a new subdirectory under `include/dnn/` or `python/pydnn/`:**
  add a CLAUDE.md to it that follows the standard template. Add a
  row to the feature index above if it owns a user-facing feature.
- **Removing or collapsing a subsystem:** delete its CLAUDE.md and
  remove references from this file's feature index.
- **Renaming a feature:** update the feature-index row, the owner's
  CLAUDE.md, and any cross-references.
- **Promoting `[scaffolding]` to live code:** drop the tag in the
  owner's CLAUDE.md *file table*, and check whether any maintenance
  notes are now obsolete.

## Updating this file

When you change anything that affects the feature index, the
maintenance principles, or the build recipe, update this file in the
same commit. Keep the file under ~250 lines — push detail down into
per-subsystem CLAUDE.md files.
