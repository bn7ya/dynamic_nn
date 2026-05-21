# CLAUDE.md — elasticneuralnetwork root

> Maintenance scaffolding for AI sessions. **Read this file first** when
> opening this repo. Per-directory CLAUDE.md files supply the local detail.

## Project overview

`elasticneuralnetwork` (`enn`) is a dynamic neural network library built
on top of libtorch (PyTorch C++ API). It grows and prunes itself during
training: a `ReversibleNetwork` of `ReversibleLinear` layers holds an
`active_mask` buffer per layer, and topology mutations (add/prune nodes,
insert/remove layers, expand kernels) are reversible until end-of-training
`compact()` performs the only hard erasure.

Training runs four phases driven by a `PhaseController`:

1. **TopologyDiscovery** — start small, grow on plateau.
2. **ConvergenceRateEstimation** — measure the slope to estimate the
   remaining epoch budget.
3. **AdaptiveTraining** — `StabilityMonitor` + `AdaptiveLRController`
   throttle the learning rate based on data-driven thresholds.
4. **FrozenArchitectureFinetuning** — topology locked, only weights move.

All controller thresholds (`StabilityConfig`, `AdaptiveLRConfig`,
`PruningConfig`, `PhaseScheduleConfig`) are produced by
`derive_*` functions in `csrc/training/data_driven_config.cpp` from
`DatasetStatistics` computed at the start of `fit()`. No arbitrary
multiplicative constants are baked into the code.

## How to use these CLAUDE.md files

- **Each directory that owns a feature has its own CLAUDE.md.** It tells
  you what files do what, what is invariant, what is actively maintained,
  and how the directory connects to the rest.
- **Update them in the same commit as code changes.**
- **Tags used in file tables:**
  - `[invariant]` — changing this without careful review breaks contracts
    other code relies on.
  - `[hot]` — runs in the per-batch / per-epoch loop.
  - `[novel]` — the only kind of custom CUDA/CPU op we keep; everything
    else routes through libtorch.

## Maintenance principles

The user's three explicit goals — **stability, reliability, memory
efficiency** — translate into the following operational rules. Every
CLAUDE.md sub-file refers back here.

### Stability
- **Public Python surfaces don't break.** `ElasticNetwork.fit()`,
  `.predict()`, `.compact()`, and the dataclass configs are external
  contracts.
- **CPU and CUDA builds stay green** — `_can_use_cuda()` falls back to
  `CppExtension` when `nvcc` is absent.
- **No anthropomorphic naming.** All renames per the rename table in
  the project plan; if a name is unclear, rename.

### Reliability
- **Reversible topology (soft-delete).** Mid-training mutations never
  erase weights. Only `ReversibleLinear::compact()` and
  `ReversibleNetwork::compact()` perform hard erasure. See
  `csrc/include/enn/modules/CLAUDE.md`.
- **`topology_version()` monotonic.** Every add/prune/compact bumps the
  version; observers gate on it.
- **Observers don't mutate state they observe.** `PlateauDetector` and
  `StabilityMonitor` consume scalar histories and emit decisions; they
  do not touch `ReversibleNetwork` internals.

### Memory efficiency
- **Single allocator path.** All tensors come from libtorch's caching
  allocator; we do not introduce parallel pools.
- **Compact at end-of-training.** `PhaseController::fit()` calls
  `compact()` on the network when the final phase returns.
- **Bounded histories.** Cost / utilization / stability_state trajectories
  cap at `kHistoryWindow = 1024` entries; older entries fall off the
  front.

## Feature index

| Feature | Owner | Engage |
|---|---|---|
| Reversible topology (active_mask, no mid-training erase) | `csrc/modules/` | Always on |
| `compact()` end-of-training cleanup | `csrc/modules/` | Auto-called from `PhaseController::fit()` |
| Four-phase training | `csrc/training/phase_controller.cpp` | Always on |
| `StabilityMonitor` (growth_anomaly / capacity_loss) | `csrc/controllers/stability_monitor.cpp` | Always on |
| `AdaptiveLRController` (reward / penalty / reset) | `csrc/controllers/adaptive_lr_controller.cpp` | Always on |
| `PlateauDetector` (slope/curvature gate) | `csrc/controllers/plateau_detector.cpp` | Always on |
| Data-driven config derivation | `csrc/training/data_driven_config.cpp` | Always on |
| `AdaptiveConv2d` (architecture search + growing kernels) | `csrc/modules/adaptive_conv2d.cpp` | When input shape is image-like |
| Benchmark suite (MNIST, CIFAR-10, Adult, Covertype, Higgs) | `elasticneuralnetwork/benchmarks/` | `python -m elasticneuralnetwork.benchmarks.<dataset>` |

## Build & smoke-test

```bash
pip install torch
pip install -e .

python -c "from elasticneuralnetwork import ElasticNetwork; import torch
X = torch.randn(64, 16); y = torch.eye(4)[torch.randint(0, 4, (64,))]
model = ElasticNetwork(input_shape=(16,), output_size=4, seed=42)
r = model.fit(X, y); print('OK', r.epochs_completed)"
```

For C++ tests:
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For the full benchmark gate:
```bash
for d in mnist cifar10 tabular_adult tabular_covertype tabular_higgs; do
    python -m elasticneuralnetwork.benchmarks.$d --seed 42 --device cpu
done
python -m elasticneuralnetwork.benchmarks.report
```

## Out of scope (future work)

Distributed training, mixed precision (fp16 / bf16), quantization, mobile
deployment, pretrained model loading, ONNX export, recurrent /
attention layers.

## Updating this file

Keep under ~250 lines. Push detail into per-subsystem CLAUDE.md files.
