# elasticneuralnetwork — Developer Guide

`elasticneuralnetwork` (`enn`) is a **PyTorch C++ extension** that
adds a small set of `torch.nn.Module` subclasses for dynamic neural
networks — networks that grow and prune themselves during training.
The extension is built with
[`torch.utils.cpp_extension`](https://pytorch.org/docs/stable/cpp_extension.html);
the modules drop into any standard PyTorch training loop and work
with `torch.optim.*`, `torch.utils.data.DataLoader`, `state_dict()`,
and `.to(device)` unchanged.

This guide walks you through everything you need to use `enn`. Read
the sections in order the first time. After that, use it as a
reference.

---

## Table of contents

1. [What is a PyTorch extension, and why is this one?](#1-what-is-a-pytorch-extension-and-why-is-this-one)
2. [Installing the extension](#2-installing-the-extension)
3. [Architecture](#3-architecture)
   - [Modules — the `torch.nn.Module` subclasses](#modules--the-torchnnmodule-subclasses)
   - [Controllers — the observer layer](#controllers--the-observer-layer)
   - [Training — `PhaseController` and data-driven config](#training--phasecontroller-and-data-driven-config)
4. [Using the modules with `torch.optim` and `DataLoader`](#4-using-the-modules-with-torchoptim-and-dataloader)
5. [Using the `ElasticNetwork.fit()` convenience wrapper](#5-using-the-elasticnetworkfit-convenience-wrapper)
6. [Data-driven configuration derivation](#6-data-driven-configuration-derivation)
7. [Building from source](#7-building-from-source)
8. [FAQ](#8-faq)

---

## 1. What is a PyTorch extension, and why is this one?

A **PyTorch C++ extension** is a shared library, compiled against
libtorch, that registers C++ classes (typically `torch::nn::Module`
subclasses) and exposes them to Python through pybind11. The result
behaves exactly like any other `torch.nn.Module` from the Python
side — you can put it inside a `torch.nn.Sequential`, give its
parameters to `torch.optim.Adam`, and save it with `state_dict()` —
but the compute happens in optimized C++. PyTorch provides the
`torch.utils.cpp_extension` helpers so that `pip install` recompiles
the extension against your local libtorch.

`enn` is one such extension. Three things justify the C++ layer:

1. **Reversible topology requires a custom mask buffer.** The
   `active_mask` on `ReversibleLinear` is a non-trainable buffer
   (registered via `register_buffer`, no autograd) that gates the
   forward output. Soft-pruning a node is a `mask[i] = 0` write —
   the underlying weights stay intact. The Python `torch.nn.Linear`
   has no such buffer; we add one in C++ and expose the
   `prune_nodes` / `add_nodes` / `compact` mutations.
2. **Mid-training topology changes need to manage parameters
   correctly.** Adding rows to a layer's weight matrix means
   replacing the registered parameter and adjusting any optimizer
   state. The C++ layer keeps this contract tight; the Python
   wrapper does not need to know the details.
3. **The training-orchestration controllers are scalar and frequent.**
   `PlateauDetector`, `StabilityMonitor`, `AdaptiveLRController`
   run at every epoch boundary. Implementing them in C++ keeps the
   GIL untouched (`PhaseController.fit` releases the GIL for the
   whole training run) and matches the rest of the libtorch surface.

What `enn` does **not** reinvent: tensors, autograd, optimizers, or
common ops. Tensors are `torch::Tensor`. Optimizers are
`torch::optim::Adam`. Linear is `torch::nn::functional::linear`.
Cross-entropy is `torch::nn::functional::cross_entropy`. The custom
CUDA kernel slot in `csrc/ops/` is reserved for ops genuinely novel
to this project (it is currently empty — libtorch covers everything
we need so far).

## 2. Installing the extension

```bash
pip install torch
pip install -e .
```

`setup.py` invokes `torch.utils.cpp_extension.CUDAExtension` when
`nvcc` is on the path (CUDA build) and falls back to `CppExtension`
otherwise (CPU-only build). The compile step is driven by ninja
through `torch.utils.cpp_extension.BuildExtension`. Headers under
`csrc/include/enn/` are on the include path so other C++ projects can
link against the extension if needed.

Verify the install:

```python
import torch
from elasticneuralnetwork import ReversibleLinear
m = ReversibleLinear(in_features=8, out_features=4)
print(m.forward(torch.randn(2, 8)).shape)   # torch.Size([2, 4])
print([p.shape for p in m.parameters()])   # weight + bias
```

## 3. Architecture

### Modules — the `torch.nn.Module` subclasses

| Class | Role | Buffers / parameters |
|---|---|---|
| `ReversibleLinear(in, out)` | Like `torch.nn.Linear`, plus an `active_mask` buffer that gates the forward output and a `NodeMetrics` member that tracks per-node EMA statistics. | `weight`, `bias` (parameters); `active_mask` (buffer). |
| `ReversibleNetwork(config)` | A stack of `ReversibleLinear` layers with `insert_layer` / `remove_layer` / `compact` mutations at the network level. Caches per-layer inputs during forward so the trainer can compute metrics against layer-local tensors. | Owned children registered as `layer_0`, `layer_1`, …; rebuilt on every topology mutation. |
| `AdaptiveConv2d(in_ch, out_ch, kernel_sizes)` | Differentiable architecture search across `{3, 5, 7}` kernel sizes mixed by `softmax(architecture_logits)`. `prune_smallest_candidate` freezes one candidate; `grow_active_kernel` expands the live kernel by one ring without changing forward output at the moment of growth. | One `torch::nn::Conv2d` per candidate; `architecture_logits` (parameter); `candidate_mask` (buffer). |

Soft-delete contract: `prune_nodes` flips mask entries to zero and
preserves the weights; `compact` is the only operation that hard-
removes rows. `topology_version()` is strictly monotonic on every
mutation call, including no-op compacts.

### Controllers — the observer layer

These classes consume scalar histories and emit decisions. They never
touch torch parameters or call into autograd directly.

| Class | Role |
|---|---|
| `PlateauDetector(config)` | Least-squares slope + mean second difference over a rolling window. `is_at_local_max()` requires warmup epochs satisfied, cooldown elapsed, `|slope| < slope_epsilon`, and `curvature <= 0`. |
| `StabilityMonitor(initial_layers, initial_nodes, config)` | Records growth / prune events and emits one of `Stable` / `ExcessiveGrowthRisk` / `PathologicalGrowth` / `ExcessivePruningRisk` / `PathologicalPruning` / `Critical`. Used by `LayerManager` to gate add/remove decisions. |
| `AdaptiveLRController` | Wraps `AdaptiveLRState` and `step_adaptive_lr_controller`: reward when the cost and utilization trends are both positive, penalize when they regress, full LR reset when the running imbalance crosses `critical_imbalance_threshold`. |
| `LayerManager(network, monitor, config)` | Combines the above. `analyze_with_utilization(u)` returns a `LayerDecision`; `execute(decision)` calls back into the network's public `insert_layer` / `prune_nodes` / `compact` API. CKA available via `LayerManager.linear_cka(x, y)`. *Currently exposed in C++ only*; `LayerManagerConfig` is bound to Python but the class itself is constructed internally by `PhaseController`. |
| `TrainableScheduler(network, config)` | Per-epoch trainable-fraction schedule (linear / exponential / cosine / step). *Currently exposed in C++ only*; `TrainableConfig` is bound to Python. |

### Training — `PhaseController` and data-driven config

`PhaseController(net, cost)` runs four phases in sequence:

1. **`TopologyDiscovery`** — start small, watch the
   `PlateauDetector`, ask `LayerManager` to add nodes / layers when
   the network plateaus.
2. **`ConvergenceRateEstimation`** — pure training at a lower LR;
   used to estimate the remaining epoch budget.
3. **`AdaptiveTraining`** — `step_adaptive_lr_controller` adjusts
   the LR each epoch based on the cost and utilization trends.
4. **`FrozenArchitectureFinetuning`** — topology locked, only
   weights move.

Inside each phase the loop is the standard PyTorch pattern,
implemented in C++: construct `torch::optim::Adam`, then on every
epoch `opt.zero_grad()` → `net.forward(X)` →
`torch::nn::functional::cross_entropy` →
`loss.backward()` → `opt.step()`. After all four phases,
`net.compact()` performs the only hard erasure of soft-pruned nodes.

## 4. Using the modules with `torch.optim` and `DataLoader`

The simplest path — bypass `ElasticNetwork` entirely and treat the
modules as ordinary `torch.nn.Module` instances:

```python
import torch
from elasticneuralnetwork import ReversibleNetwork, ReversibleNetworkConfig

cfg = ReversibleNetworkConfig()
cfg.input_features = 28 * 28
cfg.output_features = 10
cfg.hidden_sizes = [128, 64]
net = ReversibleNetwork(cfg)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train.view(-1, 784), y_train),
    batch_size=64, shuffle=True,
)

for epoch in range(20):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            net.forward(x_batch), y_batch
        )
        loss.backward()
        optimizer.step()

# Soft-prune the lowest-weight rows in layer 0, then hard-erase.
net.layer(0).prune_nodes([0, 1, 2])
net.compact()

# Save the raw parameter tensors. Full state_dict() is future work.
torch.save([p.detach().clone() for p in net.parameters()], "model.pt")
```

This is the canonical "PyTorch extension" usage shape: the module is
what's special; everything around it is stock PyTorch.

## 5. Using the `ElasticNetwork.fit()` convenience wrapper

If you want the four-phase schedule, dataset-statistics-driven
configuration, and the cost / utilization / stability trajectories
without writing the loop yourself:

```python
import torch
from elasticneuralnetwork import ElasticNetwork

X = torch.randn(1000, 784)
y = torch.eye(10)[torch.randint(0, 10, (1000,))]

model = ElasticNetwork(
    input_shape=(784,),
    output_size=10,
    seed=42,
    hidden_depth=1,
    cost_function="CrossEntropy",
)
result = model.fit(X, y)

print(result.epochs_completed)
print(result.parameter_count_final)
print(result.topology_version_final)
preds = model.predict(X)
```

Under the hood this is the same `ReversibleNetwork` you'd build by
hand; `model.network` exposes it.

## 6. Data-driven configuration derivation

The legacy `pydnn` library used a `MAPPING_TABLE` of lambdas like
`lambda v, c: 0.5 + 0.3 * (1.0 - c)` to translate a "variance score"
and "complexity score" into thresholds. `enn` replaces that with
explicit, cited derivations in `csrc/training/data_driven_config.cpp`:

- `compute_dataset_statistics(X, y)` → `DatasetStatistics` with
  `feature_variance_mean`, `feature_variance_spread`,
  `signal_to_noise_ratio`, `label_entropy`, `effective_rank`,
  `fisher_separability` (`-1` for regression).
- `derive_adaptive_lr_config(stats)` →
  `AdaptiveLRConfig` with `baseline_learning_rate`, imbalance
  thresholds, trend window.
- `derive_stability_config(stats)`,
  `derive_pruning_config(stats)`,
  `derive_phase_schedule(stats)`,
  `derive_plateau_config(stats)` — same pattern.

Every threshold is either `mean + k·std` (k ∈ {1, 2, 3}, σ-coverage),
a named percentile, `ceil(sqrt(n))` (window-size rule of thumb), or a
sigmoid in `z`-space (`sigmoid(z, k=5)` maps `z ≈ ±2` to ≈ 0.9). No
bare multiplicative constants live in code.

## 7. Building from source

`pip install -e .` is the supported path. The full chain:

1. `pyproject.toml` declares `torch>=2.0`, `pybind11`, and
   `setuptools>=64` as build-system requirements.
2. `setup.py` imports `torch.utils.cpp_extension`, picks
   `CUDAExtension` if `cpp_extension.CUDA_HOME is not None`
   (otherwise `CppExtension`), and globs every `*.cpp` under `csrc/`
   plus `*.cu` if CUDA is available.
3. The `BuildExtension` cmdclass shells out to ninja, which compiles
   each TU and links them into `elasticneuralnetwork/_enn_core.so`.

For IDE / standalone C++ test convenience there is also a
`CMakeLists.txt` at the repo root that does
`find_package(Torch REQUIRED)` and exposes the same sources to gtest
in `tests/cpp/`.

## 8. FAQ

**Q: Can I use `enn` modules in any `torch.nn.Module` composition?**
Partially. The modules expose `forward` and `parameters` so they
work with `torch.optim.*` and a standard training loop. They are
*not yet* full `torch.nn.Module` subclasses on the Python side —
`isinstance(m, torch.nn.Module)` returns `False`, and
`state_dict()` / `.to(device)` / `eval()` / `train()` are not
bound. Wrapping them in a thin `torch.nn.Module` shim that forwards
`parameters`/`buffers` is straightforward; doing it natively in the
extension is recorded as future work.

**Q: Does `enn` reimplement Adam / SGD?**
No. `PhaseController` constructs `torch::optim::Adam` for each phase.
If you write your own loop, use any `torch.optim.*` optimizer.

**Q: Does `enn` have its own autograd?**
No. All forward passes route through libtorch ops
(`torch::nn::functional::linear`, `cross_entropy`, …). Backward is
the standard libtorch autograd engine.

**Q: What happens if I call `prune_nodes` then `forward`?**
The mask is multiplied into the output; pruned nodes contribute zero
to the next layer. Weights are preserved. A subsequent `add_nodes`
reactivates dormant slots first, only allocating new rows when no
dormant capacity is left — this is what makes prune→add a parameter-
preserving round trip.

**Q: When are weights truly removed?**
Only on `compact()`. Soft-pruning is reversible; compact is not.
`topology_version()` increments on every mutation call (including
no-op compacts).

**Q: Can I freeze the topology and keep training?**
Yes — that is what `FrozenArchitectureFinetuning` does. Or, in your
own loop, just stop calling the topology mutations.

**Q: How is this different from `pydnn`?**
`pydnn` had a custom `Tensor<T>`, custom CUDA kernels, hand-rolled
optimizers, and anthropomorphic naming (`Cancer`, `Alzheimer`,
`EmotionalState`). `enn` is a clean PyTorch extension: libtorch
tensors, libtorch optimizers, libtorch autograd, scientifically
named controllers. The dynamic-topology *intent* is preserved; the
implementation is entirely on top of PyTorch.

---

## Future work

Items not yet implemented; PRs welcome:

- **Full `torch.nn.Module` Python API parity** on the pybind11
  bindings — `state_dict()`, `load_state_dict()`, `.to(device)`,
  `eval()`, `train()`, `register_forward_hook`, recursive
  `named_modules()`. Today the bindings expose only `forward`,
  `parameters`, and `buffers`.
- **Native CUDA kernel for the row-0 D2H metric pattern** under
  `csrc/ops/`; currently every metric op routes through libtorch.
- **`AdaptiveConv2d` micro-benchmark** to confirm the
  sum-over-candidates form is acceptable on real image shapes
  before adding a fused masked-mixing kernel.
- Distributed training, mixed precision (fp16 / bf16),
  quantization, mobile deployment, pretrained model loading,
  ONNX export, recurrent / attention layers.
