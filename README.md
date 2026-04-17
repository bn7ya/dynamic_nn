# pydnn — Dynamic Neural Network

A neural network that **grows and prunes itself during training**. It starts small, watches how much useful signal each neuron carries, and reshapes its own architecture as it learns: adding capacity where the network needs it, removing it where it doesn't.

C++17 core with SIMD (AVX/AVX2/AVX512) and optional CUDA acceleration, wrapped with a pybind11 Python API.

---

## Why a dynamic network?

In a conventional feed‑forward network, you pick a topology up front — *N* layers, *M* neurons per layer — and then train it. After training, if you inspect the weights, a hard truth shows up:

> **Most neurons end up holding very little information.**

Their activations are nearly constant, their gradients are tiny, they duplicate what neighbouring neurons already encode, or they simply never specialise. You paid for them in memory, FLOPs, and training time, and they're carrying almost no signal. The flip side is equally common: a few layers become bottlenecks — you under‑provisioned the part of the network that actually needed the capacity, and no amount of training recovers what the architecture can't represent.

Static architectures force you to guess both of these in advance, and then live with the guess.

**pydnn rejects the guess.** During training it continuously measures a per‑neuron and per‑layer *efficiency* signal — essentially "how much of the loss reduction is this unit actually responsible for?" — plus two structural health scores inspired by pathology:

- **Cancer score** — detects runaway, redundant growth (neurons over‑firing, duplicating each other).
- **Alzheimer score** — detects dying units (vanishing activations, dead gradients, forgotten features).

When the network sees low‑information neurons, it prunes them. When it sees a saturated layer that can't fit the signal, it grows more capacity there. The result is an architecture that is *earned* by the data rather than *assumed* by the engineer — smaller where smaller works, larger only where it has to be.

On top of that, the trainer adapts its own hyperparameters: learning rate, batch size (small → large), and the fraction of trainable nodes (100% → 1%) all shift automatically as training progresses through its phases.

---

## Installation

### Requirements

- Python **3.8+**
- A C++17 compiler (gcc ≥ 9, clang ≥ 10, or MSVC 2019+)
- CMake **3.16+**
- *(Optional)* CUDA Toolkit 11+ for GPU acceleration — auto‑detected at build time

### Install from source

```bash
git clone https://github.com/dnn/dynamic_nn.git
cd dynamic_nn
pip install .
```

That's it. `pip install .` will:

1. Install the PEP 517 build deps (`setuptools`, `wheel`, `pybind11`, `cmake`) in an isolated env.
2. Invoke CMake on the root `CMakeLists.txt`, compiling the C++ core and the `_dnn_core` pybind11 module.
3. If `nvcc` is found on your PATH, also compile the CUDA kernels for compute capabilities 75/80/86/89/100 (Turing → Blackwell).
4. Place the compiled extension alongside the `pydnn/` Python package.

### Build knobs (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `DNN_ENABLE_CUDA` | auto | Set to `OFF` to force CPU‑only build even if `nvcc` is present |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `nproc` | Parallel compile jobs |
| `CMAKE_ARGS` | *(empty)* | Extra flags passed to CMake, e.g. `-DDNN_ENABLE_AVX512=ON` |
| `DEBUG` | `0` | Build the C++ core in `Debug` mode (asserts on, optimisations off) |

Example — force CPU‑only + AVX‑512:

```bash
DNN_ENABLE_CUDA=OFF CMAKE_ARGS="-DDNN_ENABLE_AVX512=ON" pip install .
```

### Editable install (for contributors)

```bash
pip install -e .
```

### Verify the install

```python
import pydnn
print(pydnn.__version__)                  # 0.0.1
print("CUDA:", pydnn.cuda_available())    # True / False
print(pydnn.validate_environment())       # detailed diagnostics
```

---

## Your first dynamic NN

This is the complete "hello world" — a 10‑class digit classifier. The network is given nothing more than the input shape, the output size, and a seed. Everything else (layer count, layer widths, learning rate, batch schedule, early stopping, pruning thresholds) is chosen by the network itself.

```python
import numpy as np
from pydnn import DynamicNetwork

# Fake data — swap for MNIST, your own dataset, etc.
rng = np.random.default_rng(0)
X_train = rng.standard_normal((1000, 784)).astype(np.float32)
y_train = np.eye(10)[rng.integers(0, 10, size=1000)].astype(np.float32)

# 1. Declare the *interface*, not the *architecture*
network = DynamicNetwork(
    input_shape=(784,),
    output_size=10,
    seed=42,
    cost_function="CrossEntropy",
    device="cuda" if __import__("pydnn").cuda_available() else "cpu",
)

# 2. Train. The architecture grows and prunes itself as it goes.
result = network.fit(X_train, y_train)

print(f"Final cost:       {result.final_cost:.4f}")
print(f"Final efficiency: {result.final_efficiency:.2%}")
print(f"Nodes added:      {result.nodes_added}")
print(f"Nodes removed:    {result.nodes_removed}")
print(f"Layers added:     {result.layers_added}")
print(f"Stopping reason:  {result.stopping_reason}")

# 3. Predict
X_test = rng.standard_normal((32, 784)).astype(np.float32)
predictions = network.predict(X_test)       # shape (32, 10)

# 4. Inspect the network's self‑reported health
health = network.health_status()
print(f"Diagnosis: {health.diagnosis}")

# 5. Save / load
network.save("./checkpoints", name="digits_v1")
# later...
from pydnn import DynamicNetwork
restored = DynamicNetwork.load("./checkpoints/digits_v1")
```

### What you should expect to see

During `fit()` the network prints per‑epoch telemetry — cost, efficiency, the current (layers, nodes) shape, and any growth/prune events. Expect the architecture to mutate many times over the course of training, especially in the first few epochs. That is not instability; it's the network doing the hyperparameter search that you would otherwise have done yourself.

### Shape conventions

- `X`: `(n_samples, *input_shape)` — for vectors, `(n, 784)`; for images fed as flat vectors, reshape first.
- `y`: `(n_samples, output_size)` — **one‑hot** for classification, raw targets for regression.
- `dtype`: `float32` is preferred everywhere.

---

## Going further

Everything past this point is optional — the defaults are designed to work without any of it.

### Pick a different loss

```python
from pydnn import COST_FUNCTIONS
print(COST_FUNCTIONS)
# ['MSE', 'MAE', 'CrossEntropy', 'BinaryCrossEntropy',
#  'Huber', 'LogCosh', 'KLDivergence', 'CosineSimilarity']
```

### Tune the dynamic behaviour

Every subsystem is exposed as a small config dataclass — pass only the ones you care about, the rest keep their defaults.

```python
from pydnn import (
    DynamicNetwork,
    ArchitectureConfig,
    EfficiencyConfig,
    EarlyStoppingConfig,
)

network = DynamicNetwork(
    input_shape=(784,),
    output_size=10,
    seed=42,
    architecture=ArchitectureConfig(),     # max layers/nodes, growth rates
    efficiency=EfficiencyConfig(),         # pruning thresholds
    early_stopping=EarlyStoppingConfig(),  # patience, min delta
)
```

### Generate three models at once (efficient / balanced / accurate)

```python
from pydnn import generate_models

efficient, balanced, accurate = generate_models(
    X_train, y_train,
    seed=42,
    output_dir="./models",
)
```

### Visualise training

```python
from pydnn import auto_generate_plots
auto_generate_plots(result, output_dir="./plots")
```

Requires `matplotlib` and/or `plotly`:

```bash
pip install ".[viz]"
```

### Optional extras

```bash
pip install ".[viz]"     # matplotlib, plotly
pip install ".[image]"   # pillow
pip install ".[video]"   # opencv-python
pip install ".[audio]"   # librosa
pip install ".[all]"     # all of the above
pip install ".[dev]"     # pytest, pybind11
```

---

## Project layout

```
dynamic_nn/
├── CMakeLists.txt          # C++/CUDA build definition
├── setup.py                # pip entry point (invokes CMake)
├── pyproject.toml          # PEP 517 build deps
├── include/dnn/            # public C++ headers
├── src/                    # C++ sources
│   ├── core/               # tensors, layers, network
│   ├── training/           # trainer, optimizers, early stopping
│   ├── dynamics/           # the self-adaptation engine (health, efficiency)
│   ├── simd/               # AVX/AVX2/AVX512 kernels
│   └── cuda/               # CUDA kernels (optional)
├── python/pydnn/           # Python package + pybind11 bindings
└── examples/, notebooks/   # usage examples
```

---

## License

MIT — see [LICENSE](./LICENSE). Redistributions and derivative works must credit Mohammed Al-Yahyai as the original author.

## Author

**Mohammed Al-Yahyai** — bn7ya@outlook.com
