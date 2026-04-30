"""Builds notebook.ipynb for the dynamic-NN vs standard-NN research study.

Run once to assemble the notebook:
    python notebooks/beanchmark/build_notebook.py

Then execute it to embed outputs:
    jupyter nbconvert --to notebook --execute notebook.ipynb --inplace \
        --ExecutePreprocessor.timeout=1800
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "notebook.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip("\n"))


def code(src: str):
    return new_code_cell(src.strip("\n"))


# ---------------------------------------------------------------------------
# Section 0 — Abstract & Table of Contents
# ---------------------------------------------------------------------------

def section_0():
    return [
        md(r"""
# Dynamic Neural Networks vs. Static Neural Networks — A Comprehensive Research Study

**Author**: Mohammed Al-Yahyai  &nbsp;|&nbsp;  **Project**: `dynamic_nn` / `pydnn`  &nbsp;|&nbsp;  **Language**: C++17 + CUDA + Python (pybind11)

---

## Abstract

We compare a *self-adapting* neural network — one that grows and prunes its own architecture during training based on per-neuron information density — against a conventional *static* (fixed-architecture) neural network, across five task families: regression, classification, unsupervised reconstruction, transformer-based text generation, and OCR on the MNIST digit dataset. The study also examines the applicability of dynamic adaptation inside Mixture-of-Experts (MoE) architectures. Every experiment is run with 3 random seeds and results are reported as mean ± standard deviation.

The dynamic network (`pydnn`) is implemented in C++17 with optional CUDA acceleration and exposed to Python via pybind11; the static baseline is PyTorch (with CUDA). We measure final loss / accuracy, parameter count, training time, inference latency, and — for `pydnn` — grow/prune event counts and network-health trajectory. Across tasks, the dynamic network converges to architectures that are **smaller than a generously-sized static baseline at comparable accuracy**, at the cost of a modest training-time overhead spent on structural mutation and health monitoring. Section 7 reports the full numbers; Section 8 concludes with concrete improvement paths.

---

## Table of Contents

- **Section 1** — Introduction: the cancer/alzheimer metaphor and the algorithm
- **Section 2** — Literature & motivation: why rethink static sizing
- **Section 3** — Design & implementation plan (C++/CUDA/SIMD architecture)
- **Section 4** — Implementation walkthrough (source code, layer by layer)
- **Section 5** — Dummy data generation
- **Section 6** — Experiments
  - 6.1 Regression (synthetic polynomial)
  - 6.2 Classification (synthetic multi-class)
  - 6.3 Unsupervised (autoencoder reconstruction)
  - 6.4 Text generation (small transformer)
  - 6.5 OCR (MNIST via PyTorch)
  - 6.6 MoE applicability (dynamic experts + gate)
- **Section 7** — Findings (master summary + Pareto plots)
- **Section 8** — Conclusion & future improvements
"""),
    ]


# ---------------------------------------------------------------------------
# Section 1 — Introduction
# ---------------------------------------------------------------------------

def section_1():
    return [
        md(r"""
## 1 · Introduction

### 1.1  A neural network in one paragraph

A feed-forward neural network is a stack of affine maps separated by non-linear activations: $h_{\ell+1} = \sigma(W_\ell h_\ell + b_\ell)$. Each scalar element of $h_\ell$ is a *neuron*; each $W_\ell$ row is its input receptive field. The **architecture** — how many layers, how wide each is — is a free choice the engineer makes *before* seeing what the data actually needs. Every subsequent design decision (optimizer, batch size, regulariser) is a compensation for how well or poorly that initial guess was made.

### 1.2  The motivating observation

After training a reasonably-sized network you can look at the weights and activations and notice something uncomfortable:

> **Most neurons end up carrying very little information.**

Their activations are almost constant across inputs, their gradients are essentially zero, they duplicate features that neighbouring neurons already encode, or they simply never specialise. You paid for them in memory, FLOPs, and training time, and they are contributing close to nothing. The symmetric failure mode is equally common: occasionally a layer becomes a **bottleneck** — it lacks the width to represent the signal the next layer needs — and no amount of additional training recovers what the architecture cannot express.

Static architectures force you to guess both problems away in advance, and then live with the guess.

### 1.3  The cancer / alzheimer metaphor

`dynamic_nn` borrows two words from pathology to name two ways a training run can go wrong:

- **Cancer** — uncontrolled, self-reinforcing growth. The network keeps adding capacity without corresponding improvement in loss; new neurons duplicate old ones; layer counts balloon. The algorithmic counterpart is *greedy growth without feedback*.
- **Alzheimer** — progressive, self-reinforcing loss of capacity. Dead ReLUs multiply, gradients vanish, previously-useful neurons atrophy, and pruning decisions remove yet more. The algorithmic counterpart is *greedy pruning without preservation*.

Both pathologies are easy to produce with naive grow-or-prune heuristics. The `dynamic_nn` *health monitor* gives each a quantitative score, and the *layer manager* refuses to act when either score is critical. The whole system is designed so that structural decisions are **local** (per neuron / per layer), but **health-gated** (globally supervised).

### 1.4  Per-neuron efficiency

The signal that drives mutation is a per-neuron *efficiency* score, a running estimate of how much of the loss reduction this particular unit is actually responsible for. It is **not** raw activation magnitude and **not** weight magnitude — two signals that Lottery-Ticket–style magnitude pruning relies on. Instead it combines:

- activation variance over a batch (a neuron that fires nearly constantly is low-information),
- gradient contribution during back-propagation,
- correlation with neighbouring neurons' activations (duplication penalty),
- saturation fraction (a tanh/sigmoid neuron pinned at one end is dead).

Layer-level efficiency is the mean of its neurons'. When a layer's efficiency rises above a sigmoid-adaptive *saturation threshold*, the layer is saturated and the manager adds neurons (up to 25% width growth); when efficiency falls below the *efficiency threshold* and the layer has dead neurons, the manager removes them. Both branches are vetoed by the health monitor if cancer or alzheimer risk is detected.

### 1.5  Adaptive saturation threshold

A key algorithmic ingredient is that the saturation threshold is **not constant** — it depends on current global efficiency through a sigmoid:

$$\tau_{\text{sat}}(e) = b + r \cdot \sigma(k\,(e - 0.5)), \qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

with defaults $b=0.3$, $r=0.5$, $k=5.0$. The effect is intuitive: when global efficiency is low ($e \ll 0.5$), $\tau_{\text{sat}}$ drops and the network becomes *more willing to mutate* — it's not working well, so it tries bigger structural changes; when efficiency is high ($e \gg 0.5$), $\tau_{\text{sat}}$ rises and the network preserves its hard-won architecture. This is a soft, differentiable analogue of an exploration / exploitation schedule baked into the structural-mutation mechanism itself. Source: `include/dnn/dynamics/layer_manager.hpp` lines 62–69.

### 1.6  Health scoring

The cancer and alzheimer scores are computed from a 50-epoch sliding window of structural-change events, each factored into four components:

| Factor | Cancer weight | Alzheimer weight |
|---|---|---|
| Rate of layer additions / removals | 0.30 | 0.30 |
| Max consecutive additions / removals | 0.25 | 0.25 |
| Size growth / shrink ratio vs initial | 0.25 | 0.25 |
| Node growth / minimum-approach factor | 0.20 | 0.20 |

Score ≥ 0.70 → **Risk**; score ≥ 0.90 → **Critical** (state becomes `Cancer` or `Alzheimer`, layer manager declines all mutations). Source: `include/dnn/dynamics/health_monitor.hpp` lines 208–304. Keeping the weights roughly symmetric between cancer and alzheimer is deliberate: the algorithm should be no more eager to grow than to prune.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 2 — Literature & motivation
# ---------------------------------------------------------------------------

def section_2():
    return [
        md(r"""
## 2 · Literature & Motivation

### 2.1  The "most neurons are empty" claim is old and well-supported

The observation that trained networks contain a great deal of dead weight has been made, quantified, and exploited repeatedly in the literature:

- **LeCun, Denker & Solla (1990), *Optimal Brain Damage*** — the first widely-cited showing that second-derivative saliency can identify and remove a large fraction of weights without accuracy loss.
- **Han, Pool, Tran & Dally (2015), *Deep Compression*** — showed that dense networks trained for ImageNet could be *9–13×* smaller after magnitude-based pruning + quantisation with no accuracy degradation.
- **Frankle & Carbin (2019), *The Lottery Ticket Hypothesis*** — showed that a trained dense network contains sparse sub-networks (≤10–20 % of weights) that, when *retrained from the original init*, match the full network's accuracy. Strong evidence that most parameters in a static net are, in the end, unused.
- **Liu et al. (2019), *Rethinking the Value of Network Pruning*** — a follow-up that argued the *architecture* (not the specific weights that survive pruning) is what carries the accuracy. Supports the case for *architecture search* rather than *weight pruning*.

What these works share: they all perform surgery **after** a static architecture has been trained, discovering in hindsight that the architecture was over-provisioned. `dynamic_nn` attempts to move that discovery into the training loop itself.

### 2.2  Size-vs-accuracy: double descent and the overparameterised regime

A more recent and more surprising body of work (Belkin et al., 2019; Nakkiran et al., 2020) established the *double descent* phenomenon: test error does not decrease monotonically with model size. It drops, then rises to a peak near the interpolation threshold, then drops again into the overparameterised regime. The regime most modern static nets operate in is the *far* right of that curve — lots of parameters, lots of waste. The dynamic network's goal is different: rather than outrunning the interpolation peak by overparameterising, it tries to **land on the left of the peak intentionally**, with just enough capacity to fit the data.

### 2.3  Biological analogy: synaptic pruning

Human brains overproduce synapses in early development and aggressively prune them during adolescence — roughly *half* of cortical synapses are removed by the end of puberty. The neurons that remain are those that carry information. Biology settled this design question long before deep learning did, and it settled it in favour of **start wide, prune hard**. `dynamic_nn` is a small algorithmic echo of that schedule, with the cancer and alzheimer pathologies standing in for the real biological failure modes (uncontrolled growth; progressive capacity loss).

### 2.4  Mixture-of-Experts as a *static approximation* of dynamic capacity

MoE architectures (Shazeer et al., 2017; Fedus et al., 2021 — *Switch Transformer*; Mistral AI, 2023 — *Mixtral 8×7B*) route each input through a *sparsely chosen* subset of expert sub-networks. The gate selects 1–2 experts per token out of 8–64, so *active parameters per token* is much smaller than total parameters. This is a clever way to decouple total capacity from per-forward-pass cost — but the experts themselves remain statically sized. Dynamic adaptation is complementary: each MoE expert could in principle be a dynamically grown/pruned sub-network, getting **both** the total-vs-active decoupling of MoE and the per-input right-sizing of dynamic_nn. Section 6.6 builds a small prototype of this.

### 2.5  The motive of `dynamic_nn`, stated plainly

> Reject the upfront architecture guess. Let the data *earn* the capacity it needs — no more, no less — and let the network refuse to grow or shrink when the mutation pattern looks pathological.

Every design decision in the project follows from that sentence.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 3 — Design and implementation plan
# ---------------------------------------------------------------------------

def section_3():
    return [
        md(r"""
## 3 · Design & Implementation Plan

### 3.1  Goals

1. **Run on CPU and GPU** without changing the public API.
2. **Zero dependency on PyTorch/TensorFlow** — the C++ core owns tensor math, the Python layer is thin.
3. **Architecture mutation is first-class** — the training loop is aware of it, the tensor shapes change between epochs, the optimiser state is transported across mutations.
4. **Health monitoring is mandatory, not optional** — no code path in the trainer skips the cancer/alzheimer check.
5. **Reproducible builds** — single `pip install .` invokes CMake, which auto-detects CUDA.

### 3.2  Module layout

| Path | Responsibility |
|---|---|
| `include/dnn/core/` | Tensor, Layer, Node, Network, Activations, Initializers, Random |
| `src/core/` | Implementation of the above |
| `include/dnn/dynamics/` | `LayerManager` (grow/prune decisions), `HealthMonitor` (cancer/alzheimer scoring), `TrainableScheduler` (which neurons get gradients this epoch) |
| `src/dynamics/` | Dynamics implementations |
| `include/dnn/training/` | `Trainer`, cost functions, early stopping, emotional-state reward/penalty |
| `src/training/` | Trainer loop; integrates dynamics + tensor ops |
| `src/simd/` | Runtime-dispatched AVX / AVX2 / AVX512 kernels for dense matmul & element-wise ops |
| `src/cuda/` | 7 CUDA modules: `cuda_runtime.cu`, `cuda_blas.cu`, `cuda_elementwise.cu`, `cuda_activations.cu`, `cuda_reductions.cu`, `cuda_memory_pool.cu`, `cuda_optimizers.cu` |
| `python/pydnn/` | Python package: `DynamicNetwork`, config dataclasses, visualization, model-generator helpers |
| `python/pydnn/_bindings.cpp` | pybind11 glue |

### 3.3  Build system

- `CMakeLists.txt` is the single source of truth.
- Auto-detects `nvcc`; if present, compiles CUDA kernels for compute capabilities **75 / 80 / 86 / 89 / 100** (Turing → Blackwell). With no CUDA toolkit available, falls back to CPU-only — which is how the current machine is configured.
- `setup.py` invokes CMake through a custom `build_ext`; `pip install .` is the supported install path.
- SIMD is on by default (`DNN_ENABLE_AVX=ON`, `DNN_ENABLE_AVX2=ON`); AVX-512 opt-in (`DNN_ENABLE_AVX512=ON`).

### 3.4  Training-loop shape (logical view)

```
for epoch in epochs:
    health = health_monitor.diagnose()
    if health.state != Critical:
        decision = layer_manager.analyze_with_efficiency(current_efficiency)
        network.apply(decision)            # grow / prune / no-op
        health_monitor.record_change(...)
    forward + backward + optimizer_step    # on (possibly mutated) architecture
    update per-neuron efficiency scores
    check early-stopping
```

That is: *decide, mutate, train, score.* The three subsystems (health, layer manager, trainer) communicate through a small number of POD structs (`HealthReport`, `LayerDecision`, `LayerMetrics`) rather than shared state — this is what lets the CPU and CUDA paths diverge internally without the training loop needing to care.

### 3.5  Environment of this run
"""),
        code(r"""
import platform, sys
import numpy as np
try:
    import torch
except ImportError:
    torch = None

import pydnn
env = pydnn.validate_environment()
print(f"Python : {sys.version.split()[0]}  ({platform.platform()})")
print(f"NumPy  : {np.__version__}")
print(f"PyDNN  : {pydnn.__version__}   (C++ extension: {env['cpp_available']})")
print(f"PyDNN CUDA available : {pydnn.cuda_available()}")
if torch is not None:
    print(f"Torch  : {torch.__version__}   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
"""),
        md(r"""
> **Note.** On this machine `nvcc` is not installed, so pydnn is built CPU-only; PyTorch has its bundled CUDA 13.x runtime and uses the GPU. All PyTorch baselines below therefore run on GPU; pydnn runs on CPU. Timing comparisons are reported transparently with this asymmetry stated — it is the *accuracy / parameter-count* comparisons that are the primary object of this study, not the wall-clock numbers, and the design discussion of CUDA kernels in §4 holds regardless of whether they are compiled on the current machine.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 4 — Implementation walkthrough
# ---------------------------------------------------------------------------

def section_4():
    return [
        md(r"""
## 4 · Implementation Walkthrough

### 4.1  The grow/prune decision — `LayerManager::analyze_with_efficiency`

This is the central piece of the whole project. Given the current global efficiency, it returns one of `{None, AddLayer, RemoveLayer, AddNodes, RemoveNodes, AddAndRemove}` together with a target layer index, the specific node IDs to remove, a natural-language reason, and a confidence. The full source is 80 lines; the decision tree is:

```text
adaptive_threshold = sigmoid_adaptive(current_efficiency)
health = health_monitor.diagnose()
if health.state == Critical:     return None                   # refuse to act

if should_add_layer(adaptive_threshold):
    return AddLayer at best_insertion_point, size = compute_new_layer_size()

if should_remove_layer():
    return RemoveLayer at least_efficient_layer

for each layer:
    if layer.avg_efficiency > adaptive_threshold and layer.num_nodes < max:
        return AddNodes at layer, count = layer.num_nodes // 4      # 25% grow

    if layer.avg_efficiency < efficiency_threshold and layer.dead_nodes > 0:
        nodes_to_remove = [n for n in layer.nodes if n.score < thr * 0.5]
        clip to keep at least min_nodes_per_layer
        if nodes_to_remove: return RemoveNodes

return None
```

Source: `include/dnn/dynamics/layer_manager.hpp` lines 91–179. The key invariants:
- **min/max nodes** (4 and 10,000) clamp growth and pruning.
- **Min layers = 2** — the manager never removes a layer if that would leave the network with a single hidden layer.
- **25% width growth** when a layer saturates — large enough to make progress, small enough that cancer scoring can catch a bad run before it spirals.

### 4.2  Cancer & alzheimer scoring — `HealthMonitor`

Scores are a weighted combination of four factors each, computed over the 50-epoch sliding window of structural-change events:

$$\text{cancer}(t) = 0.30 \cdot r_{\text{add}} + 0.25 \cdot \frac{\max\,\text{streak}_+}{|W|} + 0.25 \cdot \max\!\left(0, \tfrac{L(t)/L_0 - 1}{5}\right) + 0.20 \cdot \max\!\left(0, \tfrac{N(t)/N_0 - 1}{10}\right)$$

where $W$ is the 50-event window, $r_\text{add}$ is the layer-addition rate inside $W$, $L(t)/L_0$ is the ratio of current to initial layer count, and $N(t)/N_0$ for nodes. Alzheimer scoring is the *symmetric* formula with $1/r$ instead of $r-1$ for size, plus a hard term that maxes out as the network approaches the 2-layer minimum. Source: `include/dnn/dynamics/health_monitor.hpp` lines 208–304.

Thresholds: **≥ 0.70 → Risk**, **≥ 0.90 → Critical**. In `Risk` the manager may only perform *opposite* mutations (e.g. a cancer-risk network can still prune); in `Critical` all mutations are refused and the trainer either continues training on the frozen architecture or early-stops.

### 4.3  Python API — `DynamicNetwork`

The Python surface deliberately exposes only the *interface*, not the *architecture*:

```python
from pydnn import DynamicNetwork
network = DynamicNetwork(
    input_shape=(784,), output_size=10, seed=42,
    cost_function="CrossEntropy", device="cuda",  # or "cpu"
)
result = network.fit(X_train, y_train)      # architecture grows/prunes automatically
predictions = network.predict(X_test)
report = network.health_status()            # HealthReport(diagnosis, cancer_score, ...)
```

Advanced users can pass any subset of `{ArchitectureConfig, EfficiencyConfig, EarlyStoppingConfig, TrainingPhaseConfig, HealthScoreConfig, NormalizationConfig, PerturbationConfig, RewardPenaltyConfig}`; whatever they don't pass keeps its default. Source: `python/pydnn/__init__.py` and `python/pydnn/network.py`.

### 4.4  CUDA kernel surface (design target)

When CUDA is compiled (not on this machine), the kernels live in:

| File | Kernels |
|---|---|
| `cuda_runtime.cu` | Stream/device management, error handling |
| `cuda_blas.cu` | GEMM / GEMV (wraps cuBLAS) |
| `cuda_elementwise.cu` | Add, mul, broadcast, mask |
| `cuda_activations.cu` | ReLU, Leaky-ReLU, Tanh, Sigmoid, GELU — forward + backward |
| `cuda_reductions.cu` | Sum / mean / argmax / var (used by efficiency scoring) |
| `cuda_memory_pool.cu` | Slab allocator, avoids cudaMalloc/cudaFree per-op |
| `cuda_optimizers.cu` | SGD + Momentum, Adam step — fused |

The device-agnostic tensor (`core::Tensor<T>`) dispatches at construction time by `device` flag; the same forward pass compiles to CPU SIMD *or* CUDA depending on `Device::Cpu` / `Device::Cuda`. The layer manager never touches device memory directly — it only reads layer-level efficiency scalars — which is why mutation logic compiles once for both backends.

### 4.5  Why this architecture is *friendly* to dynamic mutation

A non-obvious point: the biggest engineering difficulty in a growing/pruning network is not the grow/prune *decision*, it is **transporting optimiser state (momentum, Adam moments) across the mutation**. The project handles this by:

1. Representing each layer's parameter block as a *logical view* on a pool-allocated tensor (not a separately-owned allocation).
2. On `AddNodes`, the pool grows and the new rows/columns are initialised with the *same* RNG stream the layer was created from.
3. On `RemoveNodes`, surviving rows are scatter-gathered into a fresh contiguous buffer; optimiser state is masked and rebuilt in the same pass.
4. The health monitor is kept *outside* the tensor path entirely — it only reads `LayerMetrics` POD, which is cheap to compute (single-pass stats over the last forward activation batch).

This decoupling is what makes the whole system fast enough to be on-by-default rather than a debug feature.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 5 — Dummy data generation
# ---------------------------------------------------------------------------

def section_5():
    return [
        md(r"""
## 5 · Dummy Data Generation

All synthetic datasets are produced here with fixed seeds so the rest of the notebook can reproduce numbers exactly. MNIST is loaded through `torchvision` (PyTorch's standard "number data" source) and kept in NumPy.
"""),
        code(r"""
import numpy as np

MASTER_SEED = 20260417
rng_master = np.random.default_rng(MASTER_SEED)
SEEDS = [42, 43, 44]            # three seeds per experiment
print('Master seed:', MASTER_SEED, ' Per-experiment seeds:', SEEDS)
"""),
        md(r"""
### 5.1  Regression — noisy multivariate polynomial

$y = 0.3 x_0^2 - 0.5 x_1 x_2 + 0.2 x_3 + 0.1 \sum_{i=4}^{9} x_i + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 0.1^2)$.
2000 samples, input dim 10. Standardised.
"""),
        code(r"""
def make_regression_data(n=2000, d=10, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (0.3 * X[:, 0] ** 2
         - 0.5 * X[:, 1] * X[:, 2]
         + 0.2 * X[:, 3]
         + 0.1 * X[:, 4:10].sum(axis=1)
         + noise * rng.standard_normal(n)).astype(np.float32)
    return X, y[:, None]

X_reg, y_reg = make_regression_data(seed=MASTER_SEED)
print('Regression:', X_reg.shape, y_reg.shape, '   target mean/std:',
      f'{y_reg.mean():.3f} / {y_reg.std():.3f}')
"""),
        md(r"""
### 5.2  Classification — 5 Gaussian clusters in ℝ²⁰
"""),
        code(r"""
def make_classification_data(n=3000, d=20, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3.0
    labels = rng.integers(0, n_classes, size=n)
    X = centers[labels] + rng.standard_normal((n, d)).astype(np.float32)
    X = X.astype(np.float32)
    y_onehot = np.eye(n_classes, dtype=np.float32)[labels]
    return X, y_onehot, labels

X_cls, y_cls, labels_cls = make_classification_data(seed=MASTER_SEED)
print('Classification:', X_cls.shape, y_cls.shape, '   class counts:',
      np.bincount(labels_cls))
"""),
        md(r"""
### 5.3  Unsupervised — autoencoder reconstruction target

Blob-like data in ℝ⁵⁰ embedded on a low-dim manifold; the autoencoder should compress it.
"""),
        code(r"""
def make_autoencoder_data(n=2000, d=50, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, rank)).astype(np.float32)
    M = rng.standard_normal((rank, d)).astype(np.float32)
    X = U @ M + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return X.astype(np.float32)

X_ae = make_autoencoder_data(seed=MASTER_SEED)
print('Autoencoder:', X_ae.shape, '   effective rank target:', 5)
"""),
        md(r"""
### 5.4  Text corpus — tiny character-level dataset for the transformer

We use a short, self-contained passage (~3 KB) to keep transformer experiments fast while still showing perplexity drop. The character vocabulary is whatever appears in the corpus.
"""),
        code(r"""
TEXT_CORPUS = (
    'A neural network that grows and prunes itself during training '
    'must decide, for every neuron it keeps or discards, whether that '
    'unit is actually carrying useful information. Most are not. Some '
    'are bottlenecks. The healthy ones lie in between, and the goal of '
    'the algorithm is to find them without falling into either cancer '
    'or alzheimer. ' * 40
)
vocab = sorted(set(TEXT_CORPUS))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
data_ids = np.array([stoi[c] for c in TEXT_CORPUS], dtype=np.int64)
print('Text corpus chars:', len(TEXT_CORPUS), '   vocab:', len(vocab), '   ids:', data_ids.shape)
"""),
        md(r"""
### 5.5  MNIST — loaded through `torchvision`

Standard 28×28 grayscale digits. We take a **5 000 train / 1 000 test** subset to keep per-seed runtime reasonable; this still provides enough data to distinguish model quality and is far above the low-data failure regime.
"""),
        code(r"""
import torch
from torchvision import datasets, transforms

_transform = transforms.Compose([transforms.ToTensor()])
_mnist_root = './mnist_data'
_train_full = datasets.MNIST(_mnist_root, train=True,  download=True, transform=_transform)
_test_full  = datasets.MNIST(_mnist_root, train=False, download=True, transform=_transform)

def _subset(ds, n, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    X = torch.stack([ds[i][0] for i in idx]).numpy().astype(np.float32)
    y = np.array([ds[i][1] for i in idx], dtype=np.int64)
    return X, y

X_mnist_tr, y_mnist_tr = _subset(_train_full, 5000, MASTER_SEED)
X_mnist_te, y_mnist_te = _subset(_test_full,  1000, MASTER_SEED + 1)
print('MNIST train:', X_mnist_tr.shape, y_mnist_tr.shape,
      '   test:', X_mnist_te.shape, y_mnist_te.shape)
"""),
    ]


# ---------------------------------------------------------------------------
# The rest of the sections are defined below in separate functions to keep
# each chunk readable. The main() at the bottom assembles everything.
# ---------------------------------------------------------------------------


def section_6_harness():
    return [
        md(r"""
## 6 · Experiments

Every experiment follows the same template:

1. Generate task data (from Section 5) with the current seed.
2. Train a pydnn `DynamicNetwork` (CPU, Python fallback — see note below).
3. Train a PyTorch static baseline (GPU when available).
4. Record {final loss or accuracy, param count, train time, inference latency, pydnn grow/prune counts}.
5. Repeat for 3 seeds; aggregate to mean ± std.

Results accumulate into the global `RESULTS` dict, which Section 7 reads to produce the master summary table.

> **Note on the pydnn code path.** This benchmark forces the pure-Python training backend via `net._use_cpp = False`. The C++ binding is functional (the old Tensor-vs-array drift was fixed in commit `dd8ba30`), but the C++ `TrainingResult` does not yet expose `nodes_added`, `nodes_removed`, `cancer_score_history`, or `architecture_history` — diagnostic fields that the experiments below rely on. Once those fields are surfaced through the binding, remove the `_use_cpp = False` override and the timing comparison becomes representative. All dynamic mutation, health scoring, and efficiency logic still runs — it is just orchestrated from Python rather than C++.
"""),
        code(r"""
import time, gc, json, math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydnn import (DynamicNetwork, TrainingPhaseConfig, ArchitectureConfig,
                   EarlyStoppingConfig)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch device:', DEVICE)

RESULTS = defaultdict(list)   # experiment -> list of per-seed dicts

def count_params_torch(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_params_pydnn(net):
    # Use live layer weights when the Python backend is active.
    # The old approach read architecture_history[-1] = (n_layers, total_nodes)
    # and misinterpreted those as layer widths, producing incorrect counts.
    if getattr(net, '_layers', None):
        total = 0
        prev = net.input_shape[0]
        for layer in net._layers:
            out = layer["W"].shape[0]
            total += prev * out + out   # W + b
            prev = out
        return total
    return 0  # C++ path: detailed layer info not yet exposed

def time_infer_torch(model, X, n_reps=3, batch=256):
    model.eval()
    X_t = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        for _ in range(2):                 # warm-up
            _ = model(X_t[:batch])
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = model(X_t)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times))

def time_infer_pydnn(net, X, n_reps=3):
    for _ in range(2):
        _ = net.predict(X[:min(64, len(X))])
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = net.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def fit_pydnn(X, y, seed, cost, max_layers=4, hidden=32, budget=30):
    tp = TrainingPhaseConfig(
        exploration_epochs=5, estimation_epochs=5,
        min_estimated_epochs=budget, max_estimated_epochs=budget,
        phase4_enabled=False,
    )
    arch = ArchitectureConfig(
        max_layers=max_layers, min_initial_hidden_size=hidden,
        max_nodes_per_layer=max(128, hidden * 4),
    )
    net = DynamicNetwork(
        input_shape=(X.shape[1],), output_size=y.shape[1],
        seed=seed, cost_function=cost, device='cpu',
        training_phase=tp, architecture=arch,
    )
    net._use_cpp = False  # keep Python path; C++ result lacks mutation/health fields
    t0 = time.perf_counter()
    res = net.fit(X, y, verbose=False)
    train_time = time.perf_counter() - t0
    return net, res, train_time

def aggregate(label, key_order):
    xs = RESULTS[label]
    if not xs:
        return None
    out = {}
    for k in key_order:
        vals = [x[k] for x in xs if x.get(k) is not None]
        if not vals:
            out[k] = None
        elif isinstance(vals[0], str):
            out[k] = vals[0]
        else:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    out['n_seeds'] = len(xs)
    return out
"""),
    ]


def section_6_1_regression():
    return [
        md(r"""
### 6.1  Regression — synthetic noisy polynomial

**Setup.** 2 000 samples, 10 inputs, 1 output. Train/test split 80/20. Metric: test RMSE (lower is better).

**Contestants.**
- `pydnn` `DynamicNetwork(MSE)` — free to grow/prune up to 4 hidden layers.
- PyTorch static MLP with two hidden layers of 64 (~5 k params), comparable to a "default" guess an engineer would write.
"""),
        code(r"""
class TorchMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(64, 64), out_activation=None):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        if out_activation is not None:
            layers.append(out_activation)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def run_regression(seed):
    X, y = make_regression_data(seed=seed)
    n_tr = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    # ---- pydnn ----
    net, res, t_train_dnn = fit_pydnn(Xtr, ytr, seed, cost='MSE', hidden=32, budget=40)
    pred_dnn = net.predict(Xte).reshape(-1)
    rmse_dnn = float(np.sqrt(((pred_dnn - yte.reshape(-1))**2).mean()))
    t_inf_dnn = time_infer_pydnn(net, Xte)
    p_dnn = count_params_pydnn(net)

    # ---- torch MLP ----
    torch.manual_seed(seed)
    model = TorchMLP(10, 1, hidden=(64, 64)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt_tr = torch.from_numpy(Xtr).to(DEVICE); yt_tr = torch.from_numpy(ytr).to(DEVICE)
    Xt_te = torch.from_numpy(Xte).to(DEVICE); yt_te = torch.from_numpy(yte).to(DEVICE)
    bs = 64
    t0 = time.perf_counter()
    for ep in range(200):
        idx = torch.randperm(len(Xt_tr), device=DEVICE)
        for i in range(0, len(Xt_tr), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = F.mse_loss(model(Xt_tr[b]), yt_tr[b])
            loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_train_torch = time.perf_counter() - t0
    with torch.no_grad():
        rmse_torch = float(torch.sqrt(F.mse_loss(model(Xt_te), yt_te)))
    t_inf_torch = time_infer_torch(model, Xte)
    p_torch = count_params_torch(model)

    RESULTS['regression_pydnn'].append(dict(
        metric=rmse_dnn, params=p_dnn, train_time=t_train_dnn, infer_time=t_inf_dnn,
        nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
        layers_added=res.layers_added, layers_removed=res.layers_removed,
        final_efficiency=res.final_efficiency,
        arch_final=res.architecture_history[-1] if res.architecture_history else None,
        final_cancer=res.cancer_score_history[-1] if res.cancer_score_history else 0.0,
        final_alzheimer=res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0,
        eff_history=res.efficiency_history[::max(1, len(res.efficiency_history)//20)] if res.efficiency_history else [],
        label='pydnn',
    ))
    RESULTS['regression_torch'].append(dict(
        metric=rmse_torch, params=p_torch, train_time=t_train_torch, infer_time=t_inf_torch,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-MLP',
    ))
    print(f'seed {seed}:  pydnn RMSE={rmse_dnn:.4f}  params={p_dnn}  train={t_train_dnn:.2f}s  '
          f'grow+prune={res.nodes_added}+{res.nodes_removed}n/{res.layers_added}+{res.layers_removed}L   '
          f'|  torch RMSE={rmse_torch:.4f}  params={p_torch}  train={t_train_torch:.2f}s')

for s in SEEDS:
    run_regression(s)

keys = ['metric', 'params', 'train_time', 'infer_time',
        'nodes_added', 'nodes_removed', 'layers_added', 'layers_removed',
        'final_efficiency']
print('\\n--- Regression summary (mean ± std) ---')
for tag in ['regression_pydnn', 'regression_torch']:
    a = aggregate(tag, keys)
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_6_2_classification():
    return [
        md(r"""
### 6.2  Classification — 5-way Gaussian blobs

**Setup.** 3 000 samples, input dim 20, 5 classes. 80/20 split. Metric: test accuracy.
"""),
        code(r"""
def run_classification(seed):
    X, y_oh, labels = make_classification_data(seed=seed)
    n_tr = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:n_tr], y_oh[:n_tr], X[n_tr:], y_oh[n_tr:]
    labels_tr, labels_te = labels[:n_tr], labels[n_tr:]

    # pydnn
    net, res, t_train_dnn = fit_pydnn(Xtr, ytr, seed, cost='CrossEntropy', hidden=32, budget=40)
    pred_dnn = net.predict(Xte)
    acc_dnn = float((pred_dnn.argmax(axis=1) == labels_te).mean())
    t_inf_dnn = time_infer_pydnn(net, Xte)
    p_dnn = count_params_pydnn(net)

    # torch MLP
    torch.manual_seed(seed)
    model = TorchMLP(20, 5, hidden=(64, 64)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt_tr = torch.from_numpy(Xtr).to(DEVICE)
    yt_tr = torch.from_numpy(labels_tr).to(DEVICE)
    Xt_te = torch.from_numpy(Xte).to(DEVICE)
    yt_te = torch.from_numpy(labels_te).to(DEVICE)
    bs = 64
    t0 = time.perf_counter()
    for ep in range(150):
        idx = torch.randperm(len(Xt_tr), device=DEVICE)
        for i in range(0, len(Xt_tr), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt_tr[b]), yt_tr[b])
            loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_train_torch = time.perf_counter() - t0
    with torch.no_grad():
        acc_torch = float((model(Xt_te).argmax(1) == yt_te).float().mean())
    t_inf_torch = time_infer_torch(model, Xte)
    p_torch = count_params_torch(model)

    RESULTS['classification_pydnn'].append(dict(
        metric=acc_dnn, params=p_dnn, train_time=t_train_dnn, infer_time=t_inf_dnn,
        nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
        layers_added=res.layers_added, layers_removed=res.layers_removed,
        final_efficiency=res.final_efficiency,
        arch_final=res.architecture_history[-1] if res.architecture_history else None,
        final_cancer=res.cancer_score_history[-1] if res.cancer_score_history else 0.0,
        final_alzheimer=res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0,
        eff_history=res.efficiency_history[::max(1, len(res.efficiency_history)//20)] if res.efficiency_history else [],
        label='pydnn',
    ))
    RESULTS['classification_torch'].append(dict(
        metric=acc_torch, params=p_torch, train_time=t_train_torch, infer_time=t_inf_torch,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-MLP',
    ))
    print(f'seed {seed}:  pydnn acc={acc_dnn:.4f}  params={p_dnn}  train={t_train_dnn:.2f}s  '
          f'grow+prune={res.nodes_added}+{res.nodes_removed}n/{res.layers_added}+{res.layers_removed}L   '
          f'|  torch acc={acc_torch:.4f}  params={p_torch}  train={t_train_torch:.2f}s')

for s in SEEDS:
    run_classification(s)

print('\\n--- Classification summary (mean ± std) ---')
for tag in ['classification_pydnn', 'classification_torch']:
    a = aggregate(tag, keys)
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_6_3_autoencoder():
    return [
        md(r"""
### 6.3  Unsupervised — autoencoder reconstruction

**Setup.** 2 000 samples, dim 50, underlying rank 5. Autoencoder reconstructs `X → X`. Metric: test MSE on reconstruction. `pydnn` is trained with input = target (standard autoencoder setup); PyTorch baseline is a symmetric 50→32→8→32→50 AE.
"""),
        code(r"""
class TorchAE(nn.Module):
    def __init__(self, d, bottleneck=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, x):
        return self.dec(self.enc(x))

def run_autoencoder(seed):
    X = make_autoencoder_data(seed=seed)
    n_tr = int(0.8 * len(X))
    Xtr, Xte = X[:n_tr], X[n_tr:]

    # pydnn  (input = target, MSE)
    net, res, t_train_dnn = fit_pydnn(Xtr, Xtr, seed, cost='MSE', hidden=32, budget=40)
    pred_dnn = net.predict(Xte)
    mse_dnn = float(((pred_dnn - Xte) ** 2).mean())
    t_inf_dnn = time_infer_pydnn(net, Xte)
    p_dnn = count_params_pydnn(net)

    # torch AE
    torch.manual_seed(seed)
    model = TorchAE(50, bottleneck=8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt_tr = torch.from_numpy(Xtr).to(DEVICE)
    Xt_te = torch.from_numpy(Xte).to(DEVICE)
    bs = 64
    t0 = time.perf_counter()
    for ep in range(200):
        idx = torch.randperm(len(Xt_tr), device=DEVICE)
        for i in range(0, len(Xt_tr), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            out = model(Xt_tr[b])
            loss = F.mse_loss(out, Xt_tr[b])
            loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_train_torch = time.perf_counter() - t0
    with torch.no_grad():
        mse_torch = float(F.mse_loss(model(Xt_te), Xt_te))
    t_inf_torch = time_infer_torch(model, Xte)
    p_torch = count_params_torch(model)

    RESULTS['autoencoder_pydnn'].append(dict(
        metric=mse_dnn, params=p_dnn, train_time=t_train_dnn, infer_time=t_inf_dnn,
        nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
        layers_added=res.layers_added, layers_removed=res.layers_removed,
        final_efficiency=res.final_efficiency,
        arch_final=res.architecture_history[-1] if res.architecture_history else None,
        final_cancer=res.cancer_score_history[-1] if res.cancer_score_history else 0.0,
        final_alzheimer=res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0,
        eff_history=res.efficiency_history[::max(1, len(res.efficiency_history)//20)] if res.efficiency_history else [],
        label='pydnn-AE',
    ))
    RESULTS['autoencoder_torch'].append(dict(
        metric=mse_torch, params=p_torch, train_time=t_train_torch, infer_time=t_inf_torch,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-AE',
    ))
    print(f'seed {seed}:  pydnn MSE={mse_dnn:.4f}  params={p_dnn}   '
          f'|  torch MSE={mse_torch:.4f}  params={p_torch}')

for s in SEEDS:
    run_autoencoder(s)

print('\\n--- Autoencoder summary (mean ± std) ---')
for tag in ['autoencoder_pydnn', 'autoencoder_torch']:
    a = aggregate(tag, keys)
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_6_4_transformer():
    return [
        md(r"""
### 6.4  Text generation — small transformer

**Setup.** Character-level language model on the tiny text corpus from §5.4. We train a 2-block PyTorch transformer (4 heads, d_model=64, ffn=128) as the **static baseline** and report perplexity. We then build a *hybrid* variant where the transformer's feed-forward sub-layer is replaced with a `DynamicNetwork`, to probe whether dynamic adaptation can plug into a transformer block. The hybrid is **exploratory** — it wraps the pydnn net in a per-token MLP wrapper; the point is not to beat the static transformer but to verify that dynamic FFNs can be trained inside an attention block.

Metrics: validation perplexity (lower is better), total parameter count, training time.
"""),
        code(r"""
import math

# --- static transformer baseline ---
class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, nlayers=2, ffn=128, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(x)

def _lm_batches(ids, seq_len, batch, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.from_numpy(ids)
    n = len(ids) - seq_len - 1
    while True:
        starts = torch.randint(0, n, (batch,), generator=g)
        x = torch.stack([ids[s:s+seq_len] for s in starts])
        y = torch.stack([ids[s+1:s+seq_len+1] for s in starts])
        yield x, y

def run_transformer(seed):
    torch.manual_seed(seed)
    seq_len, batch, steps = 32, 16, 400
    ids = data_ids
    n_train = int(0.9 * len(ids))
    ids_tr, ids_va = ids[:n_train], ids[n_train:]
    model = TinyTransformer(len(vocab), max_len=seq_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    it = _lm_batches(ids_tr, seq_len, batch, seed)
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = next(it); x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_train = time.perf_counter() - t0

    # validation perplexity
    model.eval()
    with torch.no_grad():
        it_v = _lm_batches(ids_va, seq_len, batch, seed + 1000)
        losses = []
        for _ in range(50):
            x, y = next(it_v); x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1)).item())
        ppl = math.exp(float(np.mean(losses)))
    p = count_params_torch(model)

    # inference latency: generate 50 tokens greedily
    start = torch.from_numpy(ids[:seq_len]).to(DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(start)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t1 = time.perf_counter()
        ctx = start
        for _ in range(50):
            logits = model(ctx[:, -seq_len:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ctx = torch.cat([ctx, nxt], dim=1)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t_inf = time.perf_counter() - t1

    RESULTS['transformer_torch'].append(dict(
        metric=ppl, params=p, train_time=t_train, infer_time=t_inf,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-tinyGPT',
    ))
    print(f'seed {seed}:  torch transformer  ppl={ppl:.3f}  params={p}  train={t_train:.1f}s')

for s in SEEDS:
    run_transformer(s)

# --- dynamic-FFN hybrid: replace transformer FFN with a pydnn DynamicNetwork, per-token ---
def run_transformer_dynamic(seed):
    torch.manual_seed(seed)
    # Pre-train static transformer briefly to get meaningful block outputs.
    model = TinyTransformer(len(vocab), d_model=64, nhead=4, nlayers=2, ffn=128, max_len=32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    it = _lm_batches(data_ids, 32, 16, seed)
    for step in range(200):
        x, y = next(it); x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward(); opt.step()

    # Collect hidden states after block 0 and after block 1 to train a pydnn-FFN mimic.
    model.eval()
    with torch.no_grad():
        xs, ys = [], []
        for _ in range(20):
            xb, _ = next(it); xb = xb.to(DEVICE)
            B, T = xb.shape
            pos = torch.arange(T, device=DEVICE)
            h0 = model.tok(xb) + model.pos(pos)[None]
            # After self-attn + FFN of first block:
            h1 = model.blocks.layers[0](h0, is_causal=True,
                                        src_mask=torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool())
            xs.append(h0.reshape(-1, 64).cpu().numpy())
            ys.append(h1.reshape(-1, 64).cpu().numpy())
    X_h = np.concatenate(xs).astype(np.float32)
    Y_h = np.concatenate(ys).astype(np.float32)
    print(f'  seed {seed} hybrid probe shape: X={X_h.shape} Y={Y_h.shape}')
    t0 = time.perf_counter()
    net, res, t_train_dnn = fit_pydnn(X_h, Y_h, seed, cost='MSE', hidden=64, budget=30)
    dt = time.perf_counter() - t0
    fit_mse = float(((net.predict(X_h) - Y_h) ** 2).mean())
    p = count_params_pydnn(net)
    RESULTS['transformer_dyn_ffn'].append(dict(
        metric=fit_mse, params=p, train_time=t_train_dnn, infer_time=None,
        nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
        layers_added=res.layers_added, layers_removed=res.layers_removed,
        final_efficiency=res.final_efficiency,
        arch_final=res.architecture_history[-1] if res.architecture_history else None,
        final_cancer=res.cancer_score_history[-1] if res.cancer_score_history else 0.0,
        final_alzheimer=res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0,
        eff_history=res.efficiency_history[::max(1, len(res.efficiency_history)//20)] if res.efficiency_history else [],
        label='pydnn-as-FFN',
    ))
    print(f'seed {seed}:  pydnn-FFN fit-MSE={fit_mse:.4f}  params={p}  '
          f'grow+prune={res.nodes_added}+{res.nodes_removed}n/{res.layers_added}+{res.layers_removed}L')

for s in SEEDS:
    run_transformer_dynamic(s)

print('\\n--- Transformer summary (mean ± std) ---')
for tag in ['transformer_torch', 'transformer_dyn_ffn']:
    a = aggregate(tag, keys)
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_6_5_ocr():
    return [
        md(r"""
### 6.5  OCR — MNIST handwritten digits (PyTorch number data)

**Setup.** 5 000 training / 1 000 test digits (loaded via `torchvision`). Three contestants:

- PyTorch MLP (784→128→64→10), ~110 k params
- PyTorch small CNN (2 conv blocks + MLP head), ~20 k params — flagged as apples-to-oranges because conv is a structural prior pydnn does *not* currently have
- `pydnn` `DynamicNetwork` on flattened 784-d inputs, MLP topology
"""),
        code(r"""
class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.c1(x), 2))
        x = F.relu(F.max_pool2d(self.c2(x), 2))
        x = x.flatten(1)
        return self.fc(x)

def _train_torch_classifier(model, X, y_int, epochs=15, bs=64):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y_int).to(DEVICE)
    t0 = time.perf_counter()
    for _ in range(epochs):
        idx = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    return time.perf_counter() - t0

def _eval_torch_accuracy(model, X, y_int):
    model.eval()
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y_int).to(DEVICE)
    with torch.no_grad():
        pred = model(Xt).argmax(1)
    return float((pred == yt).float().mean())

def run_mnist(seed):
    # Fixed MNIST split (not per-seed) so all three contestants see the same data;
    # seeded random init distinguishes runs.
    X_flat_tr = X_mnist_tr.reshape(len(X_mnist_tr), -1)
    X_flat_te = X_mnist_te.reshape(len(X_mnist_te), -1)
    y_tr_int = y_mnist_tr
    y_te_int = y_mnist_te
    y_tr_oh = np.eye(10, dtype=np.float32)[y_tr_int]

    # ---- pydnn on flattened 784 ----
    net, res, t_train_dnn = fit_pydnn(X_flat_tr, y_tr_oh, seed,
                                      cost='CrossEntropy', hidden=64,
                                      max_layers=4, budget=25)
    pred = net.predict(X_flat_te)
    acc_dnn = float((pred.argmax(axis=1) == y_te_int).mean())
    t_inf_dnn = time_infer_pydnn(net, X_flat_te)
    p_dnn = count_params_pydnn(net)

    # ---- torch MLP (flat) ----
    torch.manual_seed(seed)
    mlp = TorchMLP(784, 10, hidden=(128, 64))
    t_train_mlp = _train_torch_classifier(mlp, X_flat_tr, y_tr_int, epochs=15)
    acc_mlp = _eval_torch_accuracy(mlp, X_flat_te, y_te_int)
    t_inf_mlp = time_infer_torch(mlp, X_flat_te)
    p_mlp = count_params_torch(mlp)

    # ---- torch CNN (2-D) ----
    torch.manual_seed(seed)
    cnn = TorchCNN()
    t_train_cnn = _train_torch_classifier(cnn, X_mnist_tr, y_tr_int, epochs=15)
    acc_cnn = _eval_torch_accuracy(cnn, X_mnist_te, y_te_int)
    t_inf_cnn = time_infer_torch(cnn, X_mnist_te)
    p_cnn = count_params_torch(cnn)

    RESULTS['mnist_pydnn'].append(dict(
        metric=acc_dnn, params=p_dnn, train_time=t_train_dnn, infer_time=t_inf_dnn,
        nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
        layers_added=res.layers_added, layers_removed=res.layers_removed,
        final_efficiency=res.final_efficiency,
        arch_final=res.architecture_history[-1] if res.architecture_history else None,
        final_cancer=res.cancer_score_history[-1] if res.cancer_score_history else 0.0,
        final_alzheimer=res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0,
        eff_history=res.efficiency_history[::max(1, len(res.efficiency_history)//20)] if res.efficiency_history else [],
        label='pydnn',
    ))
    RESULTS['mnist_torch_mlp'].append(dict(
        metric=acc_mlp, params=p_mlp, train_time=t_train_mlp, infer_time=t_inf_mlp,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-MLP',
    ))
    RESULTS['mnist_torch_cnn'].append(dict(
        metric=acc_cnn, params=p_cnn, train_time=t_train_cnn, infer_time=t_inf_cnn,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-CNN',
    ))
    print(f'seed {seed}:  pydnn acc={acc_dnn:.4f} ({p_dnn}p)  |  '
          f'MLP acc={acc_mlp:.4f} ({p_mlp}p)  |  CNN acc={acc_cnn:.4f} ({p_cnn}p)')

for s in SEEDS:
    run_mnist(s)

print('\\n--- MNIST summary (mean ± std) ---')
for tag in ['mnist_pydnn', 'mnist_torch_mlp', 'mnist_torch_cnn']:
    a = aggregate(tag, keys)
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_6_6_moe():
    return [
        md(r"""
### 6.6  MoE applicability — K dynamic experts + a learned gate

**Setup.** A synthetic multi-modal regression: 3 000 samples, input dim 8. Each sample belongs to one of 3 regimes; each regime has a different true regression function. An oracle router would pick the right regime perfectly and each regime-expert would only need to model one function.

**Contestants.**
- Single `pydnn` DynamicNetwork trained on everything — has to represent the disjoint union of all three functions.
- `K=3` `pydnn` experts + a *learned* soft gate (small PyTorch linear layer trained with a straight-through-trick: experts are pretrained independently on *oracle* assignments per seed, then the gate is trained to reproduce the oracle). This mimics a real MoE training flow but keeps it tractable for a notebook.
- For reference, a single PyTorch MLP of matched total capacity.

Metrics: test MSE, total params, and *active* params per forward pass (for MoE: gate + top-1 expert only).
"""),
        code(r"""
def make_moe_data(n=3000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 3, size=n)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    m0 = regime == 0; m1 = regime == 1; m2 = regime == 2
    y[m0] = 0.7 * X[m0, 0] - 0.3 * X[m0, 1] ** 2
    y[m1] = np.sin(1.5 * X[m1, 2]) + 0.2 * X[m1, 3]
    y[m2] = 0.4 * X[m2, 4] * X[m2, 5] - 0.5 * X[m2, 6]
    y += 0.05 * rng.standard_normal(n).astype(np.float32)
    return X, y[:, None], regime

def run_moe(seed):
    X, y, regime = make_moe_data(seed=seed)
    n_tr = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    r_tr, r_te = regime[:n_tr], regime[n_tr:]

    # (a) single pydnn
    net_single, res_single, t_single = fit_pydnn(Xtr, ytr, seed, cost='MSE', hidden=32, budget=40)
    pred_single = net_single.predict(Xte).reshape(-1)
    mse_single = float(((pred_single - yte.reshape(-1)) ** 2).mean())
    p_single = count_params_pydnn(net_single)

    # (b) K=3 pydnn experts + learned gate
    experts = []; expert_params = 0; total_grow_prune = {}
    for k in range(3):
        mask = r_tr == k
        Xe, ye = Xtr[mask], ytr[mask]
        netk, resk, _ = fit_pydnn(Xe, ye, seed + k, cost='MSE', hidden=16, budget=30)
        experts.append(netk)
        expert_params += count_params_pydnn(netk)
        total_grow_prune[k] = (resk.nodes_added, resk.nodes_removed,
                               resk.layers_added, resk.layers_removed)

    # train a small torch gate on oracle labels
    torch.manual_seed(seed)
    gate = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3)).to(DEVICE)
    opt_g = torch.optim.Adam(gate.parameters(), lr=1e-2)
    Xt = torch.from_numpy(Xtr).to(DEVICE); rt = torch.from_numpy(r_tr).to(DEVICE)
    for ep in range(60):
        opt_g.zero_grad()
        loss = F.cross_entropy(gate(Xt), rt)
        loss.backward(); opt_g.step()
    with torch.no_grad():
        pred_k = gate(torch.from_numpy(Xte).to(DEVICE)).argmax(1).cpu().numpy()
    gate_acc = float((pred_k == r_te).mean())
    pred_moe = np.zeros(len(Xte), dtype=np.float32)
    for k in range(3):
        mk = pred_k == k
        if mk.any():
            pred_moe[mk] = experts[k].predict(Xte[mk]).reshape(-1)
    mse_moe = float(((pred_moe - yte.reshape(-1)) ** 2).mean())
    # Active params per forward pass: gate + 1 expert (average)
    gate_params = sum(p.numel() for p in gate.parameters())
    avg_expert_params = expert_params // 3
    p_active = gate_params + avg_expert_params
    p_total = gate_params + expert_params

    # (c) torch MLP of matched total capacity (~64-64)
    torch.manual_seed(seed)
    mlp = TorchMLP(8, 1, hidden=(64, 64)).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xtr).to(DEVICE); yt = torch.from_numpy(ytr).to(DEVICE)
    t0 = time.perf_counter()
    for ep in range(200):
        idx = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 64):
            b = idx[i:i+64]
            opt.zero_grad()
            loss = F.mse_loss(mlp(Xt[b]), yt[b])
            loss.backward(); opt.step()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_mlp = time.perf_counter() - t0
    with torch.no_grad():
        mse_mlp = float(F.mse_loss(mlp(torch.from_numpy(Xte).to(DEVICE)),
                                   torch.from_numpy(yte).to(DEVICE)))
    p_mlp = count_params_torch(mlp)

    RESULTS['moe_single_pydnn'].append(dict(
        metric=mse_single, params=p_single, train_time=t_single, infer_time=None,
        nodes_added=res_single.nodes_added, nodes_removed=res_single.nodes_removed,
        layers_added=res_single.layers_added, layers_removed=res_single.layers_removed,
        final_efficiency=res_single.final_efficiency,
        arch_final=None,
        final_cancer=res_single.cancer_score_history[-1] if res_single.cancer_score_history else 0.0,
        final_alzheimer=res_single.alzheimer_score_history[-1] if res_single.alzheimer_score_history else 0.0,
        eff_history=res_single.efficiency_history[::max(1, len(res_single.efficiency_history)//20)] if res_single.efficiency_history else [],
        label='single pydnn',
    ))
    RESULTS['moe_pydnn_experts'].append(dict(
        metric=mse_moe, params=p_total, train_time=None, infer_time=None,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='MoE (K=3 pydnn + gate)',
        active_params=p_active, gate_acc=gate_acc,
    ))
    RESULTS['moe_torch_mlp'].append(dict(
        metric=mse_mlp, params=p_mlp, train_time=t_mlp, infer_time=None,
        nodes_added=None, nodes_removed=None, layers_added=None, layers_removed=None,
        final_efficiency=None, arch_final=None, label='torch-MLP',
    ))
    print(f'seed {seed}:  single-pydnn MSE={mse_single:.4f}({p_single}p)   '
          f'MoE MSE={mse_moe:.4f}(total={p_total}p / active={p_active}p, gate acc={gate_acc:.3f})   '
          f'torch-MLP MSE={mse_mlp:.4f}({p_mlp}p)')

for s in SEEDS:
    run_moe(s)

print('\\n--- MoE applicability summary (mean ± std) ---')
for tag in ['moe_single_pydnn', 'moe_pydnn_experts', 'moe_torch_mlp']:
    a = aggregate(tag, keys + ['active_params', 'gate_acc'])
    print(tag, {k: (f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else v)
                for k, v in a.items() if v is not None})
"""),
    ]


def section_7_findings():
    return [
        md(r"""
## 7 · Findings

The master summary below collects every experiment's mean-and-std row. Rows are colour-coded by metric direction (lower-is-better for losses and perplexities; higher-is-better for accuracies).
"""),
        code(r"""
import pandas as pd

ROWS = [
    # (task, direction, label, result_key)
    ('Regression  (RMSE ↓)',        'lower',  'pydnn',         'regression_pydnn'),
    ('Regression  (RMSE ↓)',        'lower',  'torch-MLP',     'regression_torch'),
    ('Classification  (acc ↑)',     'higher', 'pydnn',         'classification_pydnn'),
    ('Classification  (acc ↑)',     'higher', 'torch-MLP',     'classification_torch'),
    ('Autoencoder  (MSE ↓)',        'lower',  'pydnn',         'autoencoder_pydnn'),
    ('Autoencoder  (MSE ↓)',        'lower',  'torch-AE',      'autoencoder_torch'),
    ('Transformer  (ppl ↓)',        'lower',  'torch-tinyGPT', 'transformer_torch'),
    ('Transformer  (fit-MSE ↓)',    'lower',  'pydnn-as-FFN',  'transformer_dyn_ffn'),
    ('MNIST  (acc ↑)',              'higher', 'pydnn',         'mnist_pydnn'),
    ('MNIST  (acc ↑)',              'higher', 'torch-MLP',     'mnist_torch_mlp'),
    ('MNIST  (acc ↑)',              'higher', 'torch-CNN *',   'mnist_torch_cnn'),
    ('MoE  (MSE ↓)',                'lower',  'single pydnn',  'moe_single_pydnn'),
    ('MoE  (MSE ↓)',                'lower',  'pydnn experts', 'moe_pydnn_experts'),
    ('MoE  (MSE ↓)',                'lower',  'torch-MLP',     'moe_torch_mlp'),
]

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 220)

def _fmt(pair, d=4):
    if pair is None: return ''
    if isinstance(pair, tuple): return f'{pair[0]:.{d}f} ± {pair[1]:.{d}f}'
    return str(pair)

def _build_row(task, direction, label, key):
    a = aggregate(key, keys + ['active_params', 'gate_acc', 'final_cancer', 'final_alzheimer'])
    if a is None: return None
    row = dict(task=task, contestant=label)
    row['metric']       = _fmt(a.get('metric'))
    row['params']       = _fmt(a.get('params'), d=0) if a.get('params') else ''
    row['train_time_s'] = _fmt(a.get('train_time'), d=2) if a.get('train_time') else ''
    row['infer_ms']     = _fmt(
        tuple(x*1000 for x in a['infer_time']) if isinstance(a.get('infer_time'), tuple) else a.get('infer_time'),
        d=2) if a.get('infer_time') else ''
    gp = []
    for g_k in ['nodes_added', 'nodes_removed', 'layers_added', 'layers_removed']:
        v = a.get(g_k)
        if isinstance(v, tuple):
            gp.append(f'{g_k[:4]}={v[0]:.1f}')
    row['mutation'] = ', '.join(gp) if gp else ''
    fc = a.get('final_cancer')
    fa = a.get('final_alzheimer')
    row['cancer']    = f'{fc[0]:.2%}' if isinstance(fc, tuple) else ''
    row['alzheimer'] = f'{fa[0]:.2%}' if isinstance(fa, tuple) else ''
    if a.get('active_params'):
        row['active_params'] = _fmt(a['active_params'], d=0)
    return row

rows_df = [r for r in (_build_row(*t) for t in ROWS) if r]
df = pd.DataFrame(rows_df)
print(df.to_string(index=False))
"""),
        code(r"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Pareto: accuracy vs params (higher-is-better tasks) and loss vs params (lower-is-better).
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

def _mean(x): return x[0] if isinstance(x, tuple) else x

higher_tasks = {
    'Classification':  [('pydnn', 'classification_pydnn'), ('torch-MLP', 'classification_torch')],
    'MNIST':           [('pydnn', 'mnist_pydnn'), ('torch-MLP', 'mnist_torch_mlp'),
                        ('torch-CNN', 'mnist_torch_cnn')],
}
lower_tasks = {
    'Regression':      [('pydnn', 'regression_pydnn'), ('torch-MLP', 'regression_torch')],
    'Autoencoder':     [('pydnn', 'autoencoder_pydnn'), ('torch-AE', 'autoencoder_torch')],
    'MoE':             [('single pydnn', 'moe_single_pydnn'),
                        ('pydnn experts', 'moe_pydnn_experts'),
                        ('torch-MLP', 'moe_torch_mlp')],
}

for ax, task_dict, ylabel, title in [
    (axs[0], higher_tasks, 'Accuracy', 'Higher-is-better (↑ = left-of-Pareto is good)'),
    (axs[1], lower_tasks,  'Loss',     'Lower-is-better (↓ = bottom-of-Pareto is good)'),
]:
    for tname, entries in task_dict.items():
        xs, ys, labels = [], [], []
        for label, key in entries:
            a = aggregate(key, keys + ['active_params'])
            if a is None or a.get('params') is None: continue
            xs.append(_mean(a['params']))
            ys.append(_mean(a['metric']))
            labels.append(f'{tname}:{label}')
        ax.scatter(xs, ys, s=80)
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('Parameter count (log)')
    ax.set_ylabel(ylabel); ax.set_title(title)

plt.tight_layout(); plt.show()
"""),
        code(r"""
# --- Efficiency evolution: one subplot per task, one line per seed ---
def _eff_evo(key, ax, title):
    pts = RESULTS.get(key, [])
    if not pts: return
    for i, p in enumerate(pts):
        eh = p.get('eff_history', [])
        if eh:
            ax.plot(eh, label=f'seed {i}', linewidth=1.2)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='0.5 baseline')
    ax.set_ylim(0, 1.05); ax.set_xlabel('checkpoint'); ax.set_ylabel('efficiency')
    ax.set_title(title); ax.legend(fontsize=7)

fig, axs = plt.subplots(2, 3, figsize=(14, 6))
_eff_evo('regression_pydnn',     axs[0,0], 'Regression — efficiency')
_eff_evo('classification_pydnn', axs[0,1], 'Classification — efficiency')
_eff_evo('autoencoder_pydnn',    axs[0,2], 'Autoencoder — efficiency')
_eff_evo('transformer_dyn_ffn',  axs[1,0], 'Transformer FFN — efficiency')
_eff_evo('mnist_pydnn',          axs[1,1], 'MNIST — efficiency')
_eff_evo('moe_single_pydnn',     axs[1,2], 'MoE single — efficiency')
plt.tight_layout(); plt.show()
"""),
        code(r"""
# --- Grow/prune mutation bar chart ---
def _evo(key, ax, title):
    pts = RESULTS.get(key, [])
    if not pts: return
    for i, p in enumerate(pts):
        heights = [p.get('nodes_added', 0) or 0, p.get('nodes_removed', 0) or 0,
                   p.get('layers_added', 0) or 0, p.get('layers_removed', 0) or 0]
        ax.bar(np.arange(4) + i*0.25, heights, width=0.2, label=f'seed {i}')
    ax.set_xticks(np.arange(4) + 0.25)
    ax.set_xticklabels(['nodes+', 'nodes-', 'layers+', 'layers-'])
    ax.set_title(title); ax.legend(fontsize=7)

fig, axs = plt.subplots(2, 3, figsize=(14, 6))
_evo('regression_pydnn',     axs[0,0], 'Regression — mutations')
_evo('classification_pydnn', axs[0,1], 'Classification — mutations')
_evo('autoencoder_pydnn',    axs[0,2], 'Autoencoder — mutations')
_evo('transformer_dyn_ffn',  axs[1,0], 'Transformer FFN — mutations')
_evo('mnist_pydnn',          axs[1,1], 'MNIST — mutations')
_evo('moe_single_pydnn',     axs[1,2], 'MoE single — mutations')
plt.tight_layout(); plt.show()
"""),
        code(r"""
# --- Health scores: final cancer & alzheimer per task per seed ---
health_keys = ['regression_pydnn', 'classification_pydnn', 'autoencoder_pydnn',
               'transformer_dyn_ffn', 'mnist_pydnn', 'moe_single_pydnn']
short_names = ['Reg', 'Cls', 'AE', 'TFN-FFN', 'MNIST', 'MoE']
n_tasks = len(health_keys)

fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for ax, score_key, title in [
    (axs[0], 'final_cancer',    'Final Cancer score per task'),
    (axs[1], 'final_alzheimer', 'Final Alzheimer score per task'),
]:
    for ti, (key, name) in enumerate(zip(health_keys, short_names)):
        pts = RESULTS.get(key, [])
        vals = [p.get(score_key, 0) for p in pts]
        xs = [ti + (i - 1)*0.25 for i in range(len(vals))]
        ax.scatter(xs, vals, s=60, zorder=3, label=name if ti == 0 else '')
    ax.axhline(0.7, color='red', linestyle='--', linewidth=0.9, label='threshold 0.7')
    ax.set_xticks(range(n_tasks)); ax.set_xticklabels(short_names, rotation=20, ha='right')
    ax.set_ylabel('score'); ax.set_ylim(0, 1.1); ax.set_title(title)
    ax.legend(fontsize=7)

plt.tight_layout(); plt.show()
"""),
        md(r"""
### 7.1  What the experiments say — reading the table honestly

The numbers above are from three seeds per experiment on the hardware configuration reported in §3.5 (CPU pydnn + GPU PyTorch). Reading them as data rather than as marketing, five patterns emerge:

1. **pydnn converges to smaller architectures.** On regression the dynamic net settles on **~193 parameters** vs the static baseline's **~4 900**, a 25× compression. On classification it's **~420 vs ~5 800** (14× smaller) — at identical 100 % accuracy. On MNIST the story is more nuanced: after pruning 72 of its initial 88 hidden nodes, pydnn settles on **~12 700 parameters vs the MLP's ~110 000** (8.6× smaller), but this is *larger* than the CNN's 9 100 parameters (see point 4). These numbers use corrected parameter counting that reads actual layer-weight shapes; earlier drafts of this notebook used an index-based approximation that underreported counts by 2–7×.
2. **Smaller does not mean better.** On regression the smaller pydnn has RMSE **0.39 vs 0.16** for the static baseline — roughly 2.4× worse loss. On the autoencoder, pydnn MSE is **0.018 vs 0.003** for the static AE. These are real quality gaps, not noise. The dynamic network is doing what it was asked to do (find the smallest architecture that satisfies its efficiency targets), and the efficiency targets clearly do not imply "match the loss of an overparameterised baseline."
3. **Classification is the clean win.** Both models reach **100 % accuracy** on the synthetic 5-blob task, and pydnn does it with 14× fewer parameters. When the task is intrinsically easy, dynamic sizing discovers that and stops there, while the static net allocates a default capacity anyway.
4. **MNIST is a nuanced loss.** pydnn reaches **91.4 % ± 0.6 %** on MNIST with **~12 700 parameters** (after pruning 72 of its 88 initial hidden nodes down to 16 remaining), versus 93.97 % for the MLP (110 k params) and 95.80 % for the CNN (9.1 k params). The CNN is *more* parameter-efficient than pydnn on this task — it achieves higher accuracy with fewer parameters — because its convolutional prior matches image structure, a prior pydnn does not currently have. **Dynamic sizing and the right inductive bias are orthogonal axes**, and the table shows clearly that missing the prior cannot be compensated by re-sizing.
5. **The MoE prototype under-performs.** The K=3 dynamic-expert MoE reaches MSE **0.72 ± 0.09**, *worse* than a single pydnn (0.44) and the static MLP (0.63). The cause is the gate: its test-set routing accuracy is only **~32 %**, essentially chance on a 3-way problem. The experts each learned a clean regime-specific function on their oracle-assigned training subset, but the gate, trained only on inputs (no loss signal from expert outputs), could not recover the regime labels. This is an **honest negative result for the prototype as built** and a precise pointer at the missing piece: the gate must be trained jointly with the experts through the downstream loss, not supervised on oracle labels.
6. **The transformer hybrid is a plumbing check, not a competition.** The "pydnn-as-FFN" fit-MSE (0.034) is against the static transformer's own FFN output — it says the dynamic network can be trained to reproduce a transformer FFN, at roughly **12 % of the FFN's parameters**. It does not say pydnn can replace an FFN trained end-to-end. That is §8's headline future-work item.
7. **Grow is rare, prune is frequent.** In the current hyperparameter settings, pydnn prunes **16–72 nodes** per task but adds none except on the autoencoder (where it added 10 nodes and 1 layer on top of the prune). This asymmetry is consistent with the adaptive-threshold algorithm: the initial architecture tends to be over-provisioned, efficiency scores quickly drop below the prune threshold, and the sigmoid-adaptive saturation threshold is not crossed often enough to trigger additions. Whether that is the *right* behaviour depends on whether the initial default size is generous (prune-heavy is correct) or conservative (grow-heavy would be correct).
8. **Train-time is CPU-vs-GPU, not dynamic-vs-static.** pydnn trained on CPU in 0.1–0.9 s, PyTorch on GPU in 0.5–2.6 s. The order is often *reversed* from what one might expect — because the datasets are small enough that the PyTorch per-step overhead (kernel launch, host-device copies for per-step callbacks) dominates, and pydnn is running a tighter Python loop on CPU. Neither number is a fair wall-clock comparison of the *algorithms* — that requires the CUDA-enabled pydnn build discussed in §4.4 and §8.

### 7.2  Where the static NN wins in this study

- **Low-loss regression and reconstruction.** When the target is a continuous function and quality is measured on a fine scale, the static baseline's extra capacity pays for sharper fits. pydnn's efficiency-based pruning tolerates residual error that the static network can still reduce.
- **MNIST accuracy.** Both MLP and CNN beat the flat pydnn on 5 000 training images — the MLP through brute capacity, the CNN through inductive bias.
- **MoE as built here.** Because the gate is the limiter, a single dynamic network beats the three-expert MoE.

### 7.3  Where the dynamic NN wins in this study

- **Parameter efficiency on easy tasks.** 8–25× smaller at matched (classification) or within-a-few-points (MNIST vs MLP) accuracy.
- **Architecture discovery without a-priori guessing.** No hidden-size was specified by hand; the final width/depth came from the data.
- **Pruning behaved.** Cancer and alzheimer scores stayed sub-threshold throughout — the structural mutation mechanism did not spiral. This is non-trivial; many naive grow/prune schemes do.

The overall story is that **dynamic sizing delivers radical parameter compression at a cost in peak quality**, and that closing that quality gap is the main research target — either by letting the network use its saved parameters for more depth (currently capped at 4 layers), or by making the efficiency threshold itself a function of loss rather than of activation statistics.
"""),
    ]


def section_8_conclusion():
    return [
        md(r"""
## 8 · Conclusion & Future Improvements

### 8.1  What this study established

- A *principled* dynamic network — health-gated grow/prune with a sigmoid-adaptive saturation threshold — trains **stably** across regression, classification, reconstruction, language modelling (as a block-level component), digit recognition, and MoE routing. Cancer and alzheimer scores stayed sub-threshold throughout; the structural-mutation mechanism does not spiral.
- Dynamic sizing yields **significant parameter compression**: 8–25× smaller than a sensible static baseline on the tabular and MNIST tasks (vs-MLP comparison; the CNN on MNIST is more parameter-efficient than the flat pydnn).
- On **easy tasks** (the 5-blob classification) the compression is **free** — both models reach 100 % accuracy and the dynamic net simply stops allocating.
- Dynamic adaptation is **compositional**: a dynamic feed-forward network trains on the shape signature required by a transformer FFN and by an MoE expert. The plumbing is verified even where the joint-training loss story is not.

### 8.2  What this study did *not* establish

- **Parity on fine-grained-loss tasks.** On regression and autoencoder reconstruction the dynamic net's pruned architecture paid a 2–6× loss tax vs the static baseline. The quality gap is real, not a measurement artefact.
- **Parity on MNIST.** Flat pydnn lost to an MLP and lost by a wider margin to a CNN. A convolutional prior matters; dynamic sizing does not compensate for its absence.
- **Wall-clock training-time wins for dynamic adaptation.** pydnn ran on CPU via the Python fallback; PyTorch ran on GPU. Neither direction of the resulting comparison is informative about the algorithms themselves.
- **A working end-to-end dynamic MoE.** The gate, trained on oracle regime labels only, reached chance-level routing on held-out data and dragged the MoE below even a single dynamic network. The prototype is a negative result; §8.3 item 5 is what would actually test the idea.
- **Beating a jointly-trained transformer.** The §6.4 hybrid mimics the static FFN; end-to-end training with a dynamic FFN is the experiment that would actually settle this.

### 8.3  Improvements, in rough priority order

1. **Expose mutation/health fields in the C++ `TrainingResult` binding** (`python/pydnn/_bindings.cpp`). The Tensor-vs-array bug in `_fit_cpp` was fixed in commit `dd8ba30`; the C++ trainer now runs correctly from Python. The remaining blocker is that `nodes_added`, `nodes_removed`, `cancer_score_history`, and `architecture_history` are not yet bound. Once they are, remove `net._use_cpp = False` from the harness and all timing comparisons become representative.
2. **Ship a working CUDA build** on machines without `nvcc` via binary wheels or a conda recipe. Today the source build depends on a local CUDA toolkit.
3. **Convolutional grow-ops.** Extend `LayerManager` to manipulate Conv2D filter counts, channel splits, and kernel sizes. This is the single biggest step toward closing the gap with static CNNs on image data.
4. **Attention-aware grow-ops.** Per-head efficiency scoring and per-head growth/pruning inside a transformer block. Plug this into §6.4's hybrid and train end-to-end.
5. **MoE with dynamic experts, end-to-end.** Replace the two-stage gate-training scheme in §6.6 with load-balanced, jointly-trained sparse routing where each expert's architecture is itself dynamic.
6. **Efficiency-scoring robustness under noisy gradients.** Efficiency is currently a running average of activation- and gradient-derived quantities; with batch-size / learning-rate schedules that cause transiently noisy gradients, scoring variance can trigger spurious mutations. A small IIR-filter or uncertainty-aware threshold would help.
7. **Reproducible benchmark suite.** Convert this notebook into a proper CI-runnable benchmark, pinning seeds, hardware, and epoch budgets. Regressions in grow/prune behaviour or in efficiency scoring should be catch-able by a unit test, not by a human rereading notebook outputs.
8. **Documentation of mutation hyperparameters.** `ArchitectureConfig`, `EfficiencyConfig`, `HealthScoreConfig` currently live in code with short docstrings. A short design note mapping each threshold to the observable behaviour it controls would help users without needing them to read `src/dynamics/*.cpp`.

### 8.4  Closing remark

The strongest claim this study actually supports is narrower than the one the motivation section promised: **dynamic architecture discovery reliably produces significantly smaller networks (8–25× vs a static MLP baseline), at a quality cost that ranges from zero (easy classification) to significant (fine-grained regression and reconstruction)**. That is a useful primitive — for deployment, for MoE experts, for compute-constrained on-device inference — but it is not a drop-in replacement for a well-tuned static baseline on a problem where the baseline already fits. The path forward is to close that quality gap by letting the pruned parameter budget be *redeployed* (deeper architectures, attention-aware grow-ops, joint MoE training) rather than simply *saved*. The claim "stop guessing the architecture" survives this study. The claim "and get better accuracy for free" does not, and the findings above are clear about why.
"""),
    ]


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build():
    nb = new_notebook()
    cells = []
    cells += section_0()
    cells += section_1()
    cells += section_2()
    cells += section_3()
    cells += section_4()
    cells += section_5()
    cells += section_6_harness()
    cells += section_6_1_regression()
    cells += section_6_2_classification()
    cells += section_6_3_autoencoder()
    cells += section_6_4_transformer()
    cells += section_6_5_ocr()
    cells += section_6_6_moe()
    cells += section_7_findings()
    cells += section_8_conclusion()
    nb['cells'] = cells
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3',
    }
    nb['metadata']['language_info'] = {'name': 'python'}
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, 'w') as f:
        nbf.write(nb, f)
    print(f'Wrote {NB_PATH}   ({len(cells)} cells)')


if __name__ == '__main__':
    build()
