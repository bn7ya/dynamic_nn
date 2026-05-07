# Dynamic Neural Network — Developer Guide

This guide walks you through everything you need to use `dynamic_nn`. It starts
with the concepts that make this library different from a standard neural
network, moves through a step-by-step tutorial for your first model, then covers
every public function in detail. At the end you will find technical notes on
the training pipeline, hardware acceleration, and configuration.

Read the sections in order the first time. After that, use it as a reference.

---

## Table of Contents

1. [What is a Dynamic Neural Network?](#1-what-is-a-dynamic-neural-network)
2. [Core Concepts](#2-core-concepts)
   - [Cancer — when the network grows without reason](#cancer)
   - [Alzheimer — when the network forgets too much](#alzheimer)
   - [Emotional State — how the network feels about its progress](#emotional-state)
   - [Architecture Mutations — growing and shrinking during training](#architecture-mutations)
3. [Installation and Setup](#3-installation-and-setup)
4. [Your First Dynamic Neural Network](#4-your-first-dynamic-neural-network)
5. [Checking Network Health After Training](#5-checking-network-health-after-training)
6. [Reading the Emotional State](#6-reading-the-emotional-state)
7. [Architecture Changes Since Training](#7-architecture-changes-since-training)
8. [Function Reference](#8-function-reference)
9. [Technical Details](#9-technical-details)
   - [The four training phases](#the-four-training-phases)
   - [Dynamic thresholds](#dynamic-thresholds)
   - [Cost functions](#cost-functions)
   - [Optimizers](#optimizers)
   - [CUDA and GPU acceleration](#cuda-and-gpu-acceleration)
   - [CPU acceleration with SIMD](#cpu-acceleration-with-simd)
   - [Concurrent training pipeline (`runtime_enabled=True`)](#concurrent-training-pipeline-runtime_enabledtrue)
   - [Transformer subpackage and native backend](#transformer-subpackage-and-native-backend)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. What is a Dynamic Neural Network?

Most neural networks have a fixed shape. You decide before training how many
layers and neurons to use, and that shape never changes. You guess whether 3
layers is better than 5, whether 128 neurons is enough or too many, and then
you commit.

Dynamic Neural Network does something different. It starts with a reasonable
default shape, observes how well each layer and node is actually contributing
to learning, and adjusts the structure mid-training. Nodes that are not pulling
their weight get deactivated. Layers that are saturated — every node working
at full capacity — prompt the network to grow a new layer. When training ends,
a cleanup step physically removes everything that was deactivated, leaving you
with a lean model that only contains what was actually useful.

This means you do not have to pick an architecture. You pick your data, your
cost function, and a seed. The network figures out the rest.

The library is written in C++17 with optional CUDA and SIMD support, and
exposed to Python via pybind11. You interact with it through the Python
`pydnn` package.

---

## 2. Core Concepts

Before you train your first model, you need to understand four ideas that make
this library behave differently from anything else you may have used. Each one
has a name borrowed from biology, because the network's behaviour genuinely
resembles what those names describe.

---

### Cancer

**The story.**

Imagine a startup that is growing fast. Every time a manager feels overwhelmed,
they hire two more people. Those new hires each get their own desks, their own
laptops, their own Slack channels. The headcount doubles every month. But after
a while, most of the desks are empty — the new hires are sitting around waiting
for work that never comes. The office is three times bigger than it needs to be,
the rent is enormous, and somehow things are slower than before.

That is cancer in a neural network. The network keeps adding new layers and
nodes even when the existing ones have nothing useful to do. The model grows
without becoming more capable. Training slows, memory fills up, and accuracy
either stays flat or gets worse.

**What it means technically.**

The library watches how often new layers and nodes are added over a sliding
window of recent epochs. It tracks four things:

- How frequently additions are happening in this window.
- Whether additions are happening in consecutive epochs (a bad sign — it means
  growth is not being evaluated, just happening automatically).
- How much bigger the network has grown compared to where it started.
- How much the total node count has grown compared to the start.

These four factors are combined into a single `cancer_score` between 0 and 1.
A score below 0.7 means the network is healthy. A score between 0.7 and 0.9
means it is at risk. A score at or above 0.9 means growth has become
pathological, and the library stops allowing new layers or nodes to be added
until the score comes back down.

You can read the cancer score at any point after training:

```python
health = network.health_status()
print(health.cancer_score)   # 0.0 to 1.0

result = network.fit(X, y)
print(result.cancer_score_history)  # one score per epoch
```

If cancer is chronically high, you can tune the thresholds using
`HealthScoreConfig`. The field `cancer_ratio_threshold` (default 2.0) controls
how much growth is considered excessive — a value of 2.0 means the network is
allowed to triple in size before concern is raised.

---

### Alzheimer

**The story.**

Now imagine the opposite startup. They are cutting costs aggressively. Every
month, the CFO reviews headcount and fires anyone whose output last quarter was
below the median. The problem is that when you fire the median performer, a new
person becomes the median, and they get fired next quarter. After a year, the
company has three employees left. Nobody remembers how the product works.
Customer support tickets go unanswered because the person who understood that
part of the system was let go six months ago.

That is Alzheimer in a neural network. The network keeps removing nodes and
layers to become more efficient, but at some point it removes knowledge it
actually needs. Performance collapses. The model has forgotten how to solve the
problem.

**What it means technically.**

The library watches removals with the same sliding window it uses for cancer.
It tracks:

- How frequently removals are happening.
- Whether removals are happening in consecutive epochs (layers disappearing one
  after another without pause is a red flag).
- How much smaller the network has become compared to where it started.
- Whether the network is dangerously close to the minimum viable size (2 layers).

These produce an `alzheimer_score` between 0 and 1. The same thresholds apply:
below 0.7 is healthy, 0.7 to 0.9 is at risk, 0.9 or above is pathological.
When the score reaches the at-risk threshold, the library blocks further
removals until the situation stabilises.

```python
health = network.health_status()
print(health.alzheimer_score)   # 0.0 to 1.0

result = network.fit(X, y)
print(result.alzheimer_score_history)  # one score per epoch
```

The field `alzheimer_ratio_threshold` in `HealthScoreConfig` (default 0.8)
controls how much node loss is considered excessive. A value of 0.8 means
losing 80% of the original nodes triggers high concern.

---

### Emotional State

**The story.**

Think about a student studying for an important exam. On the days when their
practice scores are going up, they feel excited. They work faster, take more
risks, try harder problems. On the days when every practice test comes back
worse than the last, they feel depressed. They slow down, second-guess
themselves, sometimes give up entirely.

Neither extreme is good. When the student is too excited, they skip the
fundamentals and rush ahead. When they are too depressed, they stop making
progress. A good teacher watches for both extremes and intervenes: "slow down,
you are getting sloppy" or "stop catastrophising, let's reset and try again
with fresh eyes."

The dynamic network has exactly this system. It calls it the emotional state.

**What it means technically.**

After each training epoch, the network evaluates whether it made progress:

- If the cost improved by more than 0.5%, and efficiency is not degrading, and
  the cost trend over the last 15 epochs is negative (going down), the network
  earns a **reward**. The learning rate is multiplied by 0.7 — take smaller
  steps because you are on the right track.
- If the cost worsened by more than 0.5%, or efficiency is degrading while the
  cost trend is negative, the network receives a **penalty**. The learning rate
  is multiplied by 1.5 — take larger steps to escape the bad spot.

Over time, the ratio of rewards to total adjustments is the **excitement ratio**.
The ratio of penalties to total adjustments is the **depression ratio**.

When the depression ratio rises above 0.6, the network is considered **depressed**
and the learning rate is partially corrected back toward the baseline (0.1).
When the depression ratio rises above 0.7, this becomes **extreme depression**
and the learning rate is fully reset to the baseline. The same logic applies in
reverse for excitement.

The five states the network can be in are:

| State | Condition | Action |
|---|---|---|
| `neutral` | Neither ratio above 0.6 | No change |
| `depressed` | Depression ratio 0.6–0.7 | LR partially corrected toward baseline |
| `excited` | Excitement ratio 0.6–0.7 | LR partially corrected toward baseline |
| `extreme_depression_reset` | Depression ratio above 0.7 | LR fully reset to 0.1 |
| `extreme_excitement_reset` | Excitement ratio above 0.7 | LR fully reset to 0.1 |

You can read all of this after training:

```python
result = network.fit(X, y)

print(result.total_rewards)          # how many epochs earned a reward
print(result.total_penalties)        # how many epochs earned a penalty
print(result.lr_reset_count)         # how many times LR was fully reset
print(result.emotional_state_history)  # state name for each epoch
print(result.depression_history)       # depression ratio per epoch
print(result.excitement_history)       # excitement ratio per epoch

health = network.health_status()
print(health.emotional_state)        # final state after training
print(health.depression_ratio)
print(health.excitement_ratio)
```

---

### Architecture Mutations

**The story.**

Picture a modern office building. As the company grows, the facilities team
adds new floors at the top. When a floor is consistently empty, they close it
and redirect the heating budget elsewhere. But here is the clever part: they do
not demolish the empty floors right away. They just turn off the lights and lock
the doors. The furniture stays. If the company grows again next year and needs
that space, reopening the floor costs almost nothing.

At the end of the lease — end of training — comes the real renovation. An
architect walks the building, counts which floors are permanently closed, and
removes them for real, along with everything inside. The building that comes out
the other side is compact, efficient, and ready for occupancy.

This is exactly how architecture mutations work.

**What it means technically.**

During training, when a node is removed, its weights are not erased. A flag
is flipped so the node outputs zero and receives no gradient. When a layer is
removed, a flag is flipped on the layer. The weights of both survive in memory.
This is called a soft delete.

At the end of training, `compact()` is called automatically. This is the hard
delete. It walks the network, physically erases every inactive node and layer,
and rebuilds the downstream layers to match the new sizes. The result is a
smaller, faster model that only contains what actually contributed to learning.

The reason for this two-phase approach is that the network might change its
mind mid-training. A node that looks useless at epoch 50 might become crucial
at epoch 100. Keeping the weights in memory means reactivating it is cheap.

---

## 3. Installation and Setup

### Building from source

You need CMake, a C++17 compiler, and pybind11.

**CPU-only build (no Python bindings — C++ tests and examples only):**

```bash
rm -rf build
cmake -S . -B build \
    -DDNN_ENABLE_CUDA=OFF \
    -DDNN_BUILD_PYTHON=OFF \
    -DDNN_BUILD_EXAMPLES=OFF
cmake --build build -j
```

**Python extension build (recommended):**

```bash
pip install pybind11 numpy

PYBIND11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S . -B build \
    -DDNN_ENABLE_CUDA=OFF \
    -DDNN_BUILD_PYTHON=ON \
    -DDNN_BUILD_EXAMPLES=OFF \
    -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build build -j
cp build/_dnn_core.cpython-*.so python/pydnn/
```

**With CUDA support:**

Replace `-DDNN_ENABLE_CUDA=OFF` with `-DDNN_ENABLE_CUDA=ON`. You need the CUDA
toolkit installed and a compatible GPU.

### Importing the library

```python
import sys
sys.path.insert(0, 'python')   # point to the python/ directory

from pydnn import DynamicNetwork
```

Or, if you installed via pip:

```python
from pydnn import DynamicNetwork
```

### Verifying the setup

```python
from pydnn import validate_environment

env = validate_environment()
print(env)
# Shows Python version, platform, whether C++ extension loaded,
# whether CUDA is available, and whether matplotlib is available.
```

---

## 4. Your First Dynamic Neural Network

This section walks you from raw data to trained model to predictions. Read it
once all the way through, then run it.

### Step 1 — Prepare your data

The library expects NumPy arrays. Inputs should be `float32`. For classification,
targets should be one-hot encoded. For regression, targets can be raw values.

```python
import numpy as np
from pydnn import DynamicNetwork

# --- Classification example ---
rng = np.random.default_rng(42)

# 200 samples, 16 features, 4 classes
X = rng.standard_normal((200, 16)).astype(np.float32)
y_labels = rng.integers(0, 4, size=200)
y = np.eye(4)[y_labels].astype(np.float32)  # one-hot encode

print(f"X shape: {X.shape}")    # (200, 16)
print(f"y shape: {y.shape}")    # (200, 4)
```

**Shape rules:**
- `X` must be `(n_samples, *input_shape)`. For a 1-D input of 16 features that
  is `(200, 16)`. For images it would be `(200, 28, 28)`.
- `y` must be `(n_samples, output_size)` for classification. For regression it
  can be `(n_samples, 1)` or `(n_samples,)`.
- Both are automatically converted to `float32` inside `fit()`, but it is good
  practice to do it yourself.

### Step 2 — Create the network

Only four arguments are required. Everything else has sensible defaults.

```python
network = DynamicNetwork(
    input_shape=(16,),
    output_size=4,
    seed=42,
    cost_function="CrossEntropy"
)
```

The `seed` controls all randomness: weight initialisation, data shuffling, node
selection during mutations. The same seed on the same data gives the same result
every run.

### Step 3 — Train

```python
result = network.fit(X, y, verbose=True)
```

You will see output like:

```
Training Dynamic Neural Network
  Input shape: (16,)
  Output size: 4
  Seed: 42
  Cost function: CrossEntropy
  Samples: 200

  [dynamic_thresholds] variance=0.512 complexity=0.341
    cancer_threshold = 0.7
    alzheimer_threshold = 0.7
    ...

Training complete!
  Epochs: 127
  Final cost: 0.241803
  Final efficiency: 0.7234
  Stopping reason: efficiency_target_reached
```

The `[dynamic_thresholds]` line tells you that the library analysed your dataset
and adjusted its internal thresholds accordingly. This is on by default. Pass
`dynamic_thresholds=False` to the constructor if you want the fixed defaults.

### Step 4 — Make predictions

```python
predictions = network.predict(X[:10])
print(predictions.shape)   # (10, 4)
print(predictions[:3])     # probabilities for each class
```

For classification, each row is a probability distribution over classes. To
get the predicted class:

```python
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y[:10], axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy on first 10 samples: {accuracy:.0%}")
```

### Step 5 — Read the training summary

The `result` object contains everything that happened during training.

```python
print(f"Success:           {result.success}")
print(f"Epochs completed:  {result.epochs_completed}")
print(f"Final cost:        {result.final_cost:.6f}")
print(f"Best cost ever:    {result.best_cost:.6f}")
print(f"Final efficiency:  {result.final_efficiency:.4f}")
print(f"Training time:     {result.training_time_ms} ms")
print(f"Why it stopped:    {result.stopping_reason}")
print()
print(f"Architecture changes:")
print(f"  Nodes added:     {result.nodes_added}")
print(f"  Nodes removed:   {result.nodes_removed}")
print(f"  Layers added:    {result.layers_added}")
print(f"  Layers removed:  {result.layers_removed}")
```

### Full working example — classification

```python
import sys
import numpy as np

sys.path.insert(0, 'python')
from pydnn import DynamicNetwork

rng = np.random.default_rng(0)
X = rng.standard_normal((500, 20)).astype(np.float32)
y_labels = rng.integers(0, 3, size=500)
y = np.eye(3)[y_labels].astype(np.float32)

network = DynamicNetwork(
    input_shape=(20,),
    output_size=3,
    seed=7,
    cost_function="CrossEntropy"
)

result = network.fit(X, y, verbose=True)
preds = network.predict(X[:5])
print("Predicted classes:", np.argmax(preds, axis=1))
print("True classes:     ", np.argmax(y[:5], axis=1))
```

### Full working example — regression

```python
import sys
import numpy as np

sys.path.insert(0, 'python')
from pydnn import DynamicNetwork

rng = np.random.default_rng(1)
X = rng.standard_normal((300, 8)).astype(np.float32)
y = (X[:, 0] * 2.5 + X[:, 1] - X[:, 2] * 0.5).reshape(-1, 1).astype(np.float32)

network = DynamicNetwork(
    input_shape=(8,),
    output_size=1,
    seed=99,
    cost_function="MSE"
)

result = network.fit(X, y, verbose=True)
preds = network.predict(X[:5])
print("Predicted:", preds.flatten()[:5])
print("True:     ", y[:5].flatten())
```

---

## 5. Checking Network Health After Training

After `fit()` returns, you can ask the network to describe its own health.

```python
health = network.health_status()

print(f"State:          {health.state}")
print(f"Cancer score:   {health.cancer_score:.3f}")
print(f"Alzheimer score:{health.alzheimer_score:.3f}")
print(f"Overall health: {health.overall_health:.3f}")
print()
print(f"Diagnosis:")
print(f"  {health.diagnosis}")
print()
print(f"Recommendations:")
for rec in health.recommendations:
    print(f"  - {rec}")
print()
print(f"Current network:")
print(f"  Layers: {health.current_layers}")
print(f"  Nodes:  {health.current_nodes}")
```

**What each field means:**

`state` is one of these strings:

| Value | Meaning |
|---|---|
| `Healthy` | All scores are in the normal range. |
| `CancerRisk` | Cancer score is rising toward the danger zone (0.7–0.9). |
| `Cancer` | Growth is pathological. Score at or above 0.9. |
| `AlzheimerRisk` | Alzheimer score is rising toward the danger zone (0.7–0.9). |
| `Alzheimer` | Pruning is pathological. Score at or above 0.9. |
| `Critical` | The network has shrunk to the minimum viable size (2 layers). |

`cancer_score` and `alzheimer_score` are values between 0 and 1. Higher means
worse. Think of anything below 0.3 as normal operation, 0.3 to 0.7 as
something to watch, and above 0.7 as a problem worth addressing.

`overall_health` combines both scores into a single value where 1.0 is perfect
health and 0.0 is a network in serious trouble.

`diagnosis` is a human-readable sentence summarising the state.

`recommendations` is a list of suggested actions. If the network is healthy,
this list is empty.

### Watching health over time

The `result` object from `fit()` stores the history of both scores:

```python
result = network.fit(X, y)

# A list with one entry per epoch (capped at the last 500 epochs)
print(result.cancer_score_history[:10])
print(result.alzheimer_score_history[:10])
```

Plotting these with matplotlib gives you a clear picture of how the network's
health evolved:

```python
import matplotlib.pyplot as plt

epochs = range(len(result.cancer_score_history))
plt.plot(epochs, result.cancer_score_history, label="Cancer")
plt.plot(epochs, result.alzheimer_score_history, label="Alzheimer")
plt.axhline(y=0.7, color='orange', linestyle='--', label="At-risk threshold")
plt.axhline(y=0.9, color='red', linestyle='--', label="Pathological threshold")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Network Health Over Training")
plt.show()
```

### What to do if the scores are high

**High cancer score:** The network is growing faster than it can learn. Consider
passing a lower `cancer_ratio_threshold` to `HealthScoreConfig` (the default is
2.0, meaning the network is allowed to triple in size before concern is raised):

```python
from pydnn import DynamicNetwork, HealthScoreConfig

network = DynamicNetwork(
    input_shape=(16,),
    output_size=4,
    seed=42,
    cost_function="CrossEntropy",
    health_score=HealthScoreConfig(cancer_ratio_threshold=1.0)  # stricter
)
```

**High alzheimer score:** The network is pruning itself too aggressively. Consider
raising the `alzheimer_ratio_threshold` (default 0.8) to allow more removal
before concern is raised, or reduce the `removal_efficiency_multiplier` in
`EfficiencyConfig` to make removals less frequent.

---

## 6. Reading the Emotional State

The emotional state tells you about the relationship between the network's
learning rate and its progress. It is most useful for diagnosing why training
stalled or behaved erratically.

### Quick read after training

```python
result = network.fit(X, y)

print(f"Total rewards:   {result.total_rewards}")
print(f"Total penalties: {result.total_penalties}")
print(f"LR resets:       {result.lr_reset_count}")

# What was the network feeling at the end?
health = network.health_status()
print(f"Final state:     {health.emotional_state}")
print(f"Depression ratio:{health.depression_ratio:.2f}")
print(f"Excitement ratio:{health.excitement_ratio:.2f}")
```

### Reading the epoch-by-epoch history

```python
result = network.fit(X, y)

for epoch, state in enumerate(result.emotional_state_history[:20]):
    lr = result.learning_rate_history[epoch]
    print(f"Epoch {epoch:3d}: {state:<30} LR={lr:.6f}")
```

A healthy training run should show mostly `neutral` with occasional `depressed`
or `excited` states. Seeing many `extreme_depression_reset` entries means the
network was struggling and the learning rate kept needing to be rescued. This
can indicate that your data is very noisy, your batch size is too small, or the
problem is genuinely hard.

### Plotting depression and excitement over time

```python
import matplotlib.pyplot as plt

epochs = range(len(result.depression_history))
plt.plot(epochs, result.depression_history, label="Depression ratio", color="red")
plt.plot(epochs, result.excitement_history, label="Excitement ratio", color="green")
plt.axhline(y=0.6, color='orange', linestyle='--', label="Moderate threshold")
plt.axhline(y=0.7, color='red', linestyle=':', label="Extreme threshold")
plt.xlabel("Epoch")
plt.ylabel("Ratio")
plt.legend()
plt.title("Emotional State Over Training")
plt.show()
```

### What to do if the network is chronically depressed

Chronic depression means the network earns more penalties than rewards, epoch
after epoch. The learning rate keeps getting bumped up and reset. This usually
means one of:

- The problem is hard and needs more epochs. Increase `max_estimated_epochs` in
  `TrainingPhaseConfig`.
- The reward threshold is too tight. Loosen it:

```python
from pydnn import DynamicNetwork, RewardPenaltyConfig

network = DynamicNetwork(
    input_shape=(16,),
    output_size=4,
    seed=42,
    cost_function="CrossEntropy",
    reward_penalty=RewardPenaltyConfig(
        cost_improvement_threshold=0.001,  # reward at 0.1% improvement (was 0.5%)
        extreme_threshold=0.8,             # reset only at 80% depression (was 70%)
    )
)
```

### What to do if the network is chronically excited

Chronic excitement means the network earns mostly rewards. This sounds good but
it usually means the learning rate is shrinking too fast and the network is
finding a local minimum rather than the global one. Try raising the baseline
learning rate:

```python
from pydnn import DynamicNetwork, RewardPenaltyConfig

network = DynamicNetwork(
    input_shape=(16,),
    output_size=4,
    seed=42,
    cost_function="CrossEntropy",
    reward_penalty=RewardPenaltyConfig(
        baseline_learning_rate=0.01,  # lower baseline (was 0.1)
        min_adjustment_factor=0.85,   # shrink LR less aggressively (was 0.7)
    )
)
```

---

## 7. Architecture Changes Since Training

One of the most interesting things about a dynamic network is that it tells
you exactly what structural decisions it made.

### Reading the totals

```python
result = network.fit(X, y)

print(f"Nodes added:   {result.nodes_added}")
print(f"Nodes removed: {result.nodes_removed}")
print(f"Layers added:  {result.layers_added}")
print(f"Layers removed:{result.layers_removed}")
```

A positive net change (nodes_added > nodes_removed) means the problem needed
more capacity than the starting architecture provided. A negative net change
means the starting architecture was over-provisioned.

### Reading the epoch-by-epoch architecture history

```python
# Each entry is (num_layers, num_nodes) at that epoch
for epoch, (layers, nodes) in enumerate(result.architecture_history[:10]):
    print(f"Epoch {epoch:3d}: {layers} layers, {nodes} nodes")
```

The architecture history is capped at the last 500 epochs. If your training
ran for longer, the earliest epochs will not appear.

### Finding the biggest structural changes

```python
arch = result.architecture_history

if len(arch) > 1:
    for i in range(1, len(arch)):
        prev_layers, prev_nodes = arch[i-1]
        curr_layers, curr_nodes = arch[i]
        if prev_layers != curr_layers or abs(curr_nodes - prev_nodes) > 10:
            print(f"Epoch {i}: {prev_layers}L/{prev_nodes}N -> {curr_layers}L/{curr_nodes}N")
```

### Plotting how the architecture evolved

```python
import matplotlib.pyplot as plt

layer_counts = [a[0] for a in result.architecture_history]
node_counts  = [a[1] for a in result.architecture_history]
epochs = range(len(layer_counts))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(epochs, layer_counts)
ax1.set_ylabel("Layers")
ax2.plot(epochs, node_counts)
ax2.set_ylabel("Nodes")
ax2.set_xlabel("Epoch")
fig.suptitle("Architecture Over Training")
plt.tight_layout()
plt.show()
```

### Why compact() runs automatically

When `fit()` finishes, it calls `compact()` on the underlying C++ network.
This physically removes every node and layer that was soft-deleted during
training. You do not need to call it yourself.

If you are using the C++ API directly and managing `Network` objects yourself,
you need to call `network.compact()` after training is complete. Do not call
it mid-training — it will raise an error by design.

---

## 8. Function Reference

This section covers every public function in the order you are most likely to
use them.

---

### `DynamicNetwork.__init__`

Creates a new dynamic network.

```python
DynamicNetwork(
    input_shape: tuple,
    output_size: int,
    seed: int,
    cost_function: str = "CrossEntropy",
    device: str = "cpu",
    training_phase: TrainingPhaseConfig = None,
    architecture: ArchitectureConfig = None,
    efficiency: EfficiencyConfig = None,
    sigmoid_threshold: SigmoidThresholdConfig = None,
    health_score: HealthScoreConfig = None,
    gradient: GradientConfig = None,
    perturbation: PerturbationConfig = None,
    early_stopping: EarlyStoppingConfig = None,
    reward_penalty: RewardPenaltyConfig = None,
    normalization: NormalizationConfig = None,
    runtime_enabled: bool = False,
    dynamic_thresholds: bool = True
)
```

**Required parameters:**

`input_shape` — A tuple describing the shape of one input sample, not including
the batch dimension. For a flat vector of 16 features: `(16,)`. For a 28-by-28
grayscale image: `(28, 28)`.

`output_size` — Number of outputs. For 10-class classification this is 10. For
a regression problem predicting one value this is 1.

`seed` — Integer seed for all randomness. The same seed on the same data always
produces the same result. Choose any integer you like.

`cost_function` — Which loss function to use. Available values:

| Name | Use for |
|---|---|
| `"CrossEntropy"` | Multi-class classification (default) |
| `"BinaryCrossEntropy"` | Binary classification (output size 1) |
| `"MSE"` | Regression — penalises large errors heavily |
| `"MAE"` | Regression — more robust to outliers |
| `"Huber"` | Regression — hybrid of MSE and MAE |
| `"LogCosh"` | Regression — smoother approximation of MAE |
| `"KLDivergence"` | Comparing output distributions |
| `"CosineSimilarity"` | Embedding or similarity tasks |

**Optional parameters:**

`device` — `"cpu"` (default) or `"cuda"`. CUDA requires the library to have
been compiled with `DNN_ENABLE_CUDA=ON`. Check with `pydnn.cuda_available()`.

`runtime_enabled` — Set to `True` to enable the concurrent training pipeline
where a parallel observer thread monitors cost trends and can trigger early
exits from training phases. This is the C++ `StageController` feature. Default
is `False`.

`dynamic_thresholds` — Set to `True` (default) to let the library analyse your
dataset and automatically adjust cancer/alzheimer thresholds, learning rate
bounds, and batch size based on the variance and complexity of your data. Set
to `False` to use the static built-in defaults, which reproduces the legacy
behaviour exactly.

All config objects (`training_phase`, `architecture`, etc.) use safe defaults
if not provided. See the Configuration Reference section for full details.

**Example:**

```python
from pydnn import DynamicNetwork, TrainingPhaseConfig

network = DynamicNetwork(
    input_shape=(784,),
    output_size=10,
    seed=42,
    cost_function="CrossEntropy",
    training_phase=TrainingPhaseConfig(
        exploration_epochs=20,        # longer exploration
        max_estimated_epochs=500      # allow more training time
    )
)
```

---

### `DynamicNetwork.fit`

Trains the network.

```python
result = network.fit(
    X: np.ndarray,
    y: np.ndarray,
    callback: callable = None,
    verbose: bool = True
) -> TrainingResult
```

`X` — Input data. Shape must be `(n_samples, *input_shape)`. Will be
automatically cast to `float32` if it is not already.

`y` — Target data. Shape must be `(n_samples, output_size)`. Will be
automatically cast to `float32`.

`callback` — An optional function you provide that gets called after each epoch.
It receives three arguments: `(epoch: int, cost: float, efficiency: float)`.
Use this for custom logging or to display progress in your own UI.

```python
def my_callback(epoch, cost, efficiency):
    if epoch % 10 == 0:
        print(f"  epoch={epoch} cost={cost:.4f} eff={efficiency:.4f}")

result = network.fit(X, y, callback=my_callback)
```

`verbose` — When `True` (default), prints a progress summary during training.
Set to `False` for silent operation.

**Returns** a `TrainingResult` dataclass with all training history. Key fields:

| Field | Type | Description |
|---|---|---|
| `success` | bool | True if training completed without errors |
| `epochs_completed` | int | Number of epochs that ran |
| `final_cost` | float | Cost at the last epoch |
| `best_cost` | float | Lowest cost achieved at any epoch |
| `final_efficiency` | float | Network efficiency at the last epoch |
| `stopping_reason` | str | Why training stopped |
| `cost_history` | list of float | Cost per epoch (last 500) |
| `efficiency_history` | list of float | Efficiency per epoch (last 500) |
| `learning_rate_history` | list of float | LR per epoch (last 500) |
| `training_time_ms` | int | Total training time in milliseconds |
| `nodes_added` | int | Total nodes added during training |
| `nodes_removed` | int | Total nodes removed during training |
| `layers_added` | int | Total layers added during training |
| `layers_removed` | int | Total layers removed during training |
| `architecture_history` | list of (int, int) | (layers, nodes) per epoch |
| `cancer_score_history` | list of float | Cancer score per epoch |
| `alzheimer_score_history` | list of float | Alzheimer score per epoch |
| `total_rewards` | int | Number of reward events |
| `total_penalties` | int | Number of penalty events |
| `lr_reset_count` | int | Number of learning rate full resets |
| `depression_history` | list of float | Depression ratio per epoch |
| `excitement_history` | list of float | Excitement ratio per epoch |
| `emotional_state_history` | list of str | Emotional state per epoch |
| `phase4_epochs` | int | Epochs run in Phase 4 fine-tuning |
| `phase4_cost_reduction` | float | Fraction of cost reduced in Phase 4 |

`stopping_reason` will be one of the following: `"efficiency_target_reached"`,
`"early_stopping"`, `"max_epochs_reached"`, `"cost_converged"`, or
`"observer_stall"` (only when `runtime_enabled=True`).

---

### `DynamicNetwork.predict`

Runs inference on input data.

```python
predictions = network.predict(
    X: np.ndarray,
    denormalize: bool = True
) -> np.ndarray
```

`X` — Input data. Same shape rules as `fit()`.

`denormalize` — When `True` (default), the output is transformed back to the
original scale of your target data. This is only meaningful for regression. For
classification the output is always probabilities and denormalization has no
effect. Pass `False` if you want the raw network output.

**Returns** a NumPy array of shape `(n_samples, output_size)`.

For classification the output is a probability distribution over classes. Use
`np.argmax(predictions, axis=1)` to get class indices.

For regression the output is the predicted value in the original scale of `y`
(assuming `denormalize=True`).

---

### `DynamicNetwork.health_status`

Returns a health report for the trained network.

```python
health = network.health_status() -> HealthReport
```

Can be called at any time after `fit()`. Returns a `HealthReport` dataclass:

| Field | Type | Description |
|---|---|---|
| `state` | str | One of Healthy, CancerRisk, Cancer, AlzheimerRisk, Alzheimer, Critical |
| `cancer_score` | float | 0–1, how much unproductive growth occurred |
| `alzheimer_score` | float | 0–1, how much excessive pruning occurred |
| `overall_health` | float | 0–1, combined health, 1.0 is perfect |
| `diagnosis` | str | Human-readable summary sentence |
| `recommendations` | list of str | Suggestions for addressing problems |
| `current_layers` | int | Layers in the network right now |
| `current_nodes` | int | Total active nodes right now |
| `emotional_state` | str | neutral, depressed, excited, extreme_depression, extreme_excitement |
| `depression_ratio` | float | Ratio of penalty epochs to total |
| `excitement_ratio` | float | Ratio of reward epochs to total |

---

### `DynamicNetwork.efficiency_report`

Returns a layer-by-layer efficiency breakdown.

```python
report = network.efficiency_report(include_nodes: bool = False) -> dict
```

`include_nodes` — When `True`, also returns per-node efficiency scores inside
each layer entry. Default is `False`.

Returns a dictionary with per-layer statistics. Fields vary by backend but
always include `avg_layer_efficiency`, `total_layers`, `total_nodes`, and
`total_parameters`.

```python
report = network.efficiency_report(include_nodes=True)
print(f"Average efficiency: {report['avg_layer_efficiency']:.3f}")
print(f"Total parameters:   {report['total_parameters']:,}")
```

---

### `DynamicNetwork.save`

Saves the trained network to disk.

```python
network.save(path: str, name: str = "model")
```

`path` — Directory to save into. Created automatically if it does not exist.

`name` — Base name for the saved files. Default is `"model"`.

Creates three types of files:
- `{name}.json` — architecture description and metadata
- `{name}_layer{i}_W.npy` — weight matrix for layer `i`
- `{name}_layer{i}_b.npy` — bias vector for layer `i`

```python
network.save("saved_models/my_experiment", name="classifier_v1")
```

---

### `DynamicNetwork.load`

Loads a previously saved network from disk.

```python
network = DynamicNetwork.load(path: str) -> DynamicNetwork
```

`path` — The directory that was passed to `save()`. The JSON and weight files
must all be present.

```python
network = DynamicNetwork.load("saved_models/my_experiment")
predictions = network.predict(X_new)
```

---

### `DynamicNetwork.to`

Moves the network to a different device.

```python
network.to(device: str) -> DynamicNetwork
```

`device` — `"cpu"` or `"cuda"`.

Returns `self` so you can chain it:

```python
network = DynamicNetwork(...).to("cuda")
```

Raises `RuntimeError` if CUDA is requested but not available.

---

### `DynamicNetwork.device` (property)

Returns the current device as a string: `"cpu"` or `"cuda"`.

```python
print(network.device)   # "cpu"
```

---

### `DynamicNetwork.num_layers` (property)

Returns the number of layers currently in the network.

```python
print(network.num_layers)  # integer
```

---

### `DynamicNetwork.num_parameters` (property)

Returns the total number of trainable parameters.

```python
print(f"Parameters: {network.num_parameters:,}")
```

---

### `ThreeModelGenerator`

Trains three variants of the same network: one optimised for efficiency (small
and fast), one balanced, and one optimised for accuracy (allowed to grow larger).
All three are trained on the same 80/10/10 train/validation/test split.

```python
from pydnn import ThreeModelGenerator

generator = ThreeModelGenerator(
    input_shape: tuple,
    output_size: int,
    seed: int,
    cost_function: str = "CrossEntropy",
    config: GeneratorConfig = None
)
```

**`ThreeModelGenerator.generate`**

Trains all three variants and returns them.

```python
variants = generator.generate(
    X: np.ndarray,
    y: np.ndarray,
    callback: callable = None
) -> list of ModelVariant
```

Each `ModelVariant` in the returned list has:
- `strategy` — `ModelStrategy.MOST_EFFICIENT`, `BALANCED`, or `HIGHEST_ACCURACY`
- `network` — the trained `DynamicNetwork`
- `training_result` — the `TrainingResult` from its `fit()` call
- `test_accuracy` — accuracy on the held-out test set
- `test_cost` — cost on the held-out test set
- `efficiency_score` — network efficiency at end of training
- `num_parameters` — total parameters
- `num_layers` — number of layers
- `description` — human-readable summary of this strategy

**`ThreeModelGenerator.select_best`**

Picks the variant that best meets a criterion.

```python
best = generator.select_best(
    variants: list,
    criterion: str = "balanced"
) -> ModelVariant
```

`criterion` options:
- `"accuracy"` — highest test accuracy
- `"efficiency"` — highest efficiency score
- `"balanced"` — best product of accuracy and efficiency (default)
- `"smallest"` — fewest parameters

**`ThreeModelGenerator.save_all`**

Saves all three variants and a comparison report.

```python
generator.save_all(
    variants: list,
    output_dir: str,
    include_comparison: bool = True
)
```

Creates a subdirectory for each strategy inside `output_dir`, saves the model
files, and optionally writes a `comparison_report.txt`.

**`ThreeModelGenerator.split_data`**

Splits data into train, validation, and test sets.

```python
(X_train, y_train), (X_val, y_val), (X_test, y_test) = generator.split_data(X, y)
```

Uses the ratios from `GeneratorConfig` (default: 80 / 10 / 10).

---

### `generate_models` (convenience function)

The easiest way to get three trained variants without setting up a generator.

```python
from pydnn import generate_models

efficient, balanced, accurate = generate_models(
    X: np.ndarray,
    y: np.ndarray,
    input_shape: tuple = None,     # inferred from X if not given
    output_size: int = None,       # inferred from y if not given
    seed: int = 42,
    cost_function: str = "CrossEntropy",
    output_dir: str = None,        # saves to disk if provided
    verbose: bool = True
)
```

Returns a tuple of three `ModelVariant` objects in order: efficient, balanced,
accurate.

```python
efficient, balanced, accurate = generate_models(X, y, seed=42)

best = max([efficient, balanced, accurate], key=lambda v: v.test_accuracy)
print(f"Best strategy: {best.strategy.value}")
print(f"Test accuracy: {best.test_accuracy:.4f}")
best.network.save("best_model/")
```

---

### `cuda_available`

Returns `True` if the library was compiled with CUDA support and a compatible
GPU is present.

```python
from pydnn import cuda_available

if cuda_available():
    network = DynamicNetwork(..., device="cuda")
```

---

### `validate_environment`

Returns a dictionary of environment diagnostics. Useful for debugging setup
problems.

```python
from pydnn import validate_environment

env = validate_environment()
print(env)
```

The dictionary includes: `python_version`, `platform`, `architecture`,
`numpy_version`, `pybind11_version`, `cpp_available`, `cpp_error`,
`expected_binary`, `binary_path`, `visualization_available`.

---

## 9. Technical Details

This section covers the internals for developers who want to understand what
the library is doing under the hood, tune its behaviour, or use the C++ API
directly.

---

### The four training phases

Every `fit()` call runs the same four-phase pipeline. Each phase has a different
purpose and different default hyperparameters.

**Phase 1 — Exploration (10 epochs)**

The network explores the architecture space aggressively. The learning rate is
0.05. Efficiency thresholds are loose (0.3), meaning the bar for adding nodes
is low. The network grows quickly to discover how much capacity the problem
actually needs.

At the end of Phase 1, the network has a rough shape and the trainer has
collected gradient statistics that will be used to calibrate Phase 3.

**Phase 2 — Estimation (10 epochs)**

Aggressive mutations stop. The learning rate drops to 0.01. The trainer
measures how fast the cost is improving and uses that rate to estimate how
many epochs Phase 3 will need. This estimate is clamped between 50 and 300
epochs by default.

You can see the estimate in `result.phase_metrics["estimated_epochs"]`.

**Phase 3 — Main Training (variable length)**

This is where most learning happens. The learning rate starts at 0.01 and is
continuously adjusted by the reward/penalty system described in the Core
Concepts section. Batch size grows over time from 32 toward 256 — small
batches early for exploration, larger batches later for stability.

During the first 20% of Phase 3 epochs, random perturbations are applied to
a small fraction of nodes (0.5% of nodes, at a scale of 0.01). This helps the
network escape local minima early.

Architecture mutations continue throughout Phase 3. The saturation threshold
rises smoothly from 0.3 to 0.8 using a sigmoid function, making mutations
progressively more conservative as training matures.

Early stopping can end Phase 3 if cost improvement drops below 1e-7 over a
window of 30 epochs.

**Phase 4 — Standard Training (50 to 200 epochs)**

The architecture is frozen. No nodes or layers are added or removed. The
learning rate starts at 0.005 and decays by a factor of 0.95 every 10 epochs.
The goal is to squeeze out an additional 50% cost reduction through fine-tuning
of the weights that survived Phase 3.

Phase 4 has its own early stopping with a patience of 35 epochs and a minimum
improvement of 1e-5.

Phase 4 can be disabled by setting `phase4_enabled=False` in
`TrainingPhaseConfig`.

---

### Dynamic thresholds

When `dynamic_thresholds=True` (the default), the library analyses your dataset
before training and derives custom thresholds for cancer/alzheimer detection,
batch sizes, gradient clipping, and early stopping patience.

It computes two signals:

`variance_score` — a value between 0 and 1 that describes how much variation
exists in your input features. High variance means the features carry a lot of
information but are also noisy. It is a weighted combination of:
- Mean feature variance (weight 0.5)
- Signal-to-noise ratio (weight 0.3)
- Spread of variance across features (weight 0.2)

`complexity_score` — a value between 0 and 1 describing how hard the problem
is. It is a weighted combination of:
- Label entropy — how evenly spread are the classes (weight 0.30)
- Effective rank of the feature matrix — how many independent directions exist
  in the data (weight 0.30)
- Fisher separability — how well the classes can be separated linearly (weight
  0.25, classification only)
- Dimensionality ratio (weight 0.15)

After training, you can inspect what was computed:

```python
result = network.fit(X, y, verbose=True)

signals = network._last_data_signals
if signals:
    print(f"Variance score:   {signals.variance_score:.3f}")
    print(f"Complexity score: {signals.complexity_score:.3f}")
    print(f"Derived thresholds:")
    for key, value in signals.derived_thresholds.items():
        print(f"  {key} = {value}")
    if signals.fallback_used:
        print("Note: fallback used (fewer than 50 samples)")
```

If your dataset has fewer than 50 samples, the dynamic threshold computation
is skipped and static defaults are used instead. This is the `fallback_used`
flag.

---

### Cost functions

| Name | Formula | When to use |
|---|---|---|
| `CrossEntropy` | `-sum(y * log(p))` | Multi-class classification. Pairs with softmax output. |
| `BinaryCrossEntropy` | `-[y*log(p) + (1-y)*log(1-p)]` | Binary classification, output size 1. |
| `MSE` | `mean((pred - y)^2)` | Regression. Penalises large errors heavily. Good default for regression. |
| `MAE` | `mean(|pred - y|)` | Regression. More robust when your targets have outliers. |
| `Huber` | MSE when error is small, MAE when large | Regression. Best of both: sensitive to small errors, robust to large ones. |
| `LogCosh` | `log(cosh(pred - y))` | Regression. Similar to Huber, fully smooth, no sharp corner. |
| `KLDivergence` | `sum(y * log(y/p))` | When outputs represent probability distributions. |
| `CosineSimilarity` | `1 - dot(a,b)/(|a||b|)` | Embedding tasks, semantic similarity. |

---

### Optimizers

The C++ backend uses Adam by default for all phases. The Python fallback uses
SGD with momentum. When using the C++ API directly, you can configure the
optimizer via `TrainerConfig`:

| Optimizer | Key parameters | Default values | When to use |
|---|---|---|---|
| `SGD` | `learning_rate` | 0.01 | Simplest baseline. Good for convex problems. |
| `SGDMomentum` | `learning_rate`, `momentum` | 0.01, 0.9 | Faster convergence than SGD, fewer oscillations. |
| `Adam` | `learning_rate`, `beta1`, `beta2`, `epsilon` | 0.01, 0.9, 0.999, 1e-8 | Default. Works well on most problems. |
| `RMSprop` | `learning_rate`, `decay_rate`, `epsilon` | 0.01, 0.99, 1e-8 | Good for recurrent-style problems or noisy gradients. |

---

### CUDA and GPU acceleration

To use a GPU, you need:

1. The CUDA toolkit installed on your machine.
2. The library compiled with `-DDNN_ENABLE_CUDA=ON`.
3. `device="cuda"` passed to `DynamicNetwork`.

```python
from pydnn import cuda_available, DynamicNetwork

if cuda_available():
    network = DynamicNetwork(
        input_shape=(784,),
        output_size=10,
        seed=42,
        cost_function="CrossEntropy",
        device="cuda"
    )
else:
    print("CUDA not available, using CPU")
    network = DynamicNetwork(
        input_shape=(784,),
        output_size=10,
        seed=42,
        cost_function="CrossEntropy"
    )
```

You can also move a network to the GPU after construction:

```python
network = DynamicNetwork(input_shape=(784,), output_size=10, seed=42)
network.to("cuda")
```

All memory management is handled automatically. GPU tensors use the
`CudaMemoryPool` singleton (20 size classes from 256 bytes to 256 MB). CPU
tensors use the `PoolAllocator` singleton (16 size classes from 64 bytes to
2 MB). You do not need to manage either of these.

---

### CPU acceleration with SIMD

On CPU, the library automatically detects the best SIMD instruction set
available on your machine and uses it for all matrix operations. The detection
happens once at startup through a CPUID probe.

Supported levels in order of preference:
- AVX-512 (512-bit vectors, available on recent Intel Xeon and some desktop CPUs)
- AVX2 and FMA (256-bit, most desktop CPUs from 2013 onward)
- AVX (256-bit, 2011 onward)
- SSE4.2 down through SSE2 (128-bit, nearly universal)
- Scalar fallback (no SIMD, always available)

You do not need to do anything to enable this. It is automatic.

If OpenMP was available at compile time, batch loops and matrix multiplications
are parallelised across available CPU cores for batch sizes larger than 4.

---

### Concurrent training pipeline (`runtime_enabled=True`)

When you pass `runtime_enabled=True`, the C++ training pipeline activates the
`StageController`. This adds a lightweight observer thread that runs in parallel
with the training loop.

The observer watches the cost history. When it detects that cost improvement
has stalled below a threshold over a 12-epoch window, it raises a signal. The
training loop picks up this signal between epochs and can exit the current
phase early. This mirrors the behaviour of the `_CostTrendObserver` in the
Python fallback.

For most use cases, `runtime_enabled=False` (the default) is sufficient and
produces equivalent results. Enable it if you want the concurrent architecture
and are training on a machine where the observer thread has meaningful breathing
room.

An important invariant: the observer thread only reads the cost history. It
never mutates network state directly. State changes happen only on the main
training thread. This design means no locks are needed on the history list.

All topology mutations (add/remove node/layer) under `runtime_enabled=True` go
through `TopologyLock` in write mode. Read-only forward and backward passes hold
the same lock in shared mode.

---

### Transformer subpackage and native backend

`pydnn.transformer` is a self-contained subpackage of building blocks for
LLM-style models — encoder, decoder, multi-head attention, grouped-query
attention, RoPE, SwiGLU, mixture-of-experts, a small `AdamW` and a
`CausalLMTrainer`. It ships its own minimal reverse-mode autograd engine on
top of NumPy (in `transformer/autograd.py`), so the models train end-to-end
without pulling in PyTorch / JAX / TF.

`pydnn.DynamicTransformer` is the dnn-flavoured wrapper over that stack: it
applies the same soft grow/prune principles that `DynamicNetwork` uses
(active masks mid-training, end-of-training compaction) to a stack of
`AdaptiveDecoderBlock`s. Calling `model.adapt(loss_history=...)` between
training steps inspects block utilisation and the loss plateau, then either
soft-prunes a low-utilisation block, grows a near-identity new block, or
re-initialises a dead expert in any MoE layer. `model.compact()` mirrors
`Network::compact()`: it physically drops blocks marked `active=False`
once training is finished. Use it the same way — call it after the last
training step, before saving or serving.

```python
import numpy as np
from pydnn import DynamicTransformer
from pydnn import transformer as T

cfg = T.TransformerConfig(
    vocab_size=2048, dim=128, num_heads=4,
    num_decoder_layers=4, ffn_type="swiglu",
    pos_encoding="rope", norm_type="rmsnorm",
    max_seq_len=256,
)
model = DynamicTransformer(cfg)
ids = np.random.randint(0, cfg.vocab_size, size=(8, 64))
logits = model(ids)               # autograd Tensor, shape (8, 64, 2048)
loss = T.cross_entropy(logits, ids)
loss.backward()
# After training:
model.compact()                   # drop blocks marked inactive
```

#### Native CPU and CUDA backend

The hot ops (`matmul`, `softmax`, `gelu`, `silu`, `LayerNorm`, `RMSNorm`,
`embedding`, fused softmax + cross-entropy) inside the transformer
subpackage transparently route through the C++ and CUDA kernels in
`include/dnn/transformer/ops.hpp` (sources under `src/transformer/cpu_ops.cpp`
and `src/cuda/transformer_ops.cu`). The kernels are exposed to Python as the
`_dnn_core.transformer_ops` pybind submodule and selected by the dispatcher
in `python/pydnn/transformer/_backend.py`.

- The kernels **back the existing `DynamicTransformer`** — they do not
  introduce a parallel implementation. The public `Tensor` / `Module` /
  `parameters()` / `adapt()` / `compact()` API is unchanged. Grow/prune
  logic stays in Python; only the inner numerics are accelerated.
- The CPU path uses OpenMP across the row dimension where it helps
  (norms, softmax, GELU/SiLU, embedding gather, batched matmul outer
  loop). The CUDA path uses cuBLAS for batched matmul and custom kernels
  with shared-memory reductions for the per-row ops; the `LayerNorm`
  and `RMSNorm` backwards reduce per-row scratch into the parameter
  gradients with a separate column-sum kernel.
- The dispatcher is automatic. If `_dnn_core` was built and a CUDA
  device is visible, the default backend is `cuda`; otherwise it is
  `cpu`; if `_dnn_core` is unavailable it falls back to `numpy`. The
  numpy path is preserved as the regression baseline.
- Override the choice with the `PYDNN_TRANSFORMER_BACKEND` environment
  variable (`numpy`, `cpu`, or `cuda`) or programmatically:

  ```python
  from pydnn.transformer import _backend as B
  B.available_backends()    # ('numpy', 'cpu') or ('numpy', 'cpu', 'cuda')
  B.current_backend()
  B.set_backend("numpy")    # for parity testing or determinism
  ```

- Numerics match the NumPy reference within fp32 noise. Per-op parity
  (max abs diff < 1e-4) and an end-to-end `DecoderOnlyModel` forward +
  backward parity check live in
  `python/tests/test_transformer_backend.py`. The same test exercises
  a `DynamicTransformer` grow → soft-prune → `compact()` cycle on the
  `cpu` backend, so the dynamic-NN integration stays under regression
  coverage.

When you build the Python extension, the transformer ops link into the
same `_dnn_core` shared module as the rest of the bindings. The static
`dnn_core` archive is compiled with `POSITION_INDEPENDENT_CODE` so the
shared-module link succeeds with or without CUDA. No separate package
or wheel is needed.

#### When to use which backend

- **`numpy`** — debugging numerical issues, comparing against a known
  reference, reproducing legacy behaviour bit-for-bit.
- **`cpu`** — default on a GPU-less machine. Faster than `numpy` for
  any non-trivial model size; OpenMP parallelism scales with the row
  count of the operation.
- **`cuda`** — default when a GPU is present and the library was built
  with `DNN_ENABLE_CUDA=ON`. Best when batch × sequence × hidden_dim is
  large enough to overcome the host↔device copy overhead; the host-side
  wrappers currently round-trip each tensor through `cudaMalloc` /
  `cudaMemcpy`, which is fine for typical training sizes but is the
  obvious target for a follow-up that keeps tensors resident on device
  across ops.

---

## 10. Configuration Reference

Every aspect of training can be fine-tuned by passing configuration objects
to the `DynamicNetwork` constructor. All of these are dataclasses with defaults
that work well for most problems.

---

### TrainingPhaseConfig

Controls the timing and learning rates of all four training phases.

```python
from pydnn import TrainingPhaseConfig

config = TrainingPhaseConfig(
    # Phase 1: Exploration
    exploration_epochs=10,
    exploration_learning_rate=0.05,
    exploration_batch_size=64,
    exploration_saturation_threshold=0.6,
    exploration_efficiency_threshold=0.2,

    # Phase 2: Estimation
    estimation_epochs=10,
    estimation_learning_rate=0.01,

    # Phase 3: Main Training
    main_learning_rate=0.01,
    main_initial_batch_size=32,
    main_max_batch_size=256,
    batch_size_growth_interval=25,
    perturbation_cutoff_ratio=0.2,
    target_efficiency=0.8,
    min_estimated_epochs=50,
    max_estimated_epochs=300,

    # Phase 4: Fine-tuning
    phase4_enabled=True,
    phase4_learning_rate=0.005,
    phase4_min_learning_rate=0.0001,
    phase4_lr_decay_rate=0.95,
    phase4_lr_decay_interval=10,
    phase4_batch_size=64,
    phase4_target_cost_reduction=0.5,
    phase4_min_epochs=50,
    phase4_max_epochs=200,
    phase4_patience=35,
    phase4_min_improvement=1e-5
)
```

---

### ArchitectureConfig

Controls the shape constraints of the network.

```python
from pydnn import ArchitectureConfig

config = ArchitectureConfig(
    min_nodes_per_layer=32,       # never let a layer shrink below this
    max_nodes_per_layer=1024,     # never let a layer grow above this
    min_nodes_to_keep=16,         # minimum nodes to keep when pruning
    max_layers=6,                 # maximum number of hidden layers
    min_layers=1,                 # minimum hidden layers
    exploration_growth_rate=0.10, # grow by 10% during Phase 1
    main_growth_rate=0.05,        # grow by 5% during Phase 3
    min_initial_hidden_size=64    # starting size for hidden layers
)
```

---

### EfficiencyConfig

Controls how node efficiency is computed and when removals happen.

```python
from pydnn import EfficiencyConfig

config = EfficiencyConfig(
    default_saturation_threshold=0.7,       # add nodes when layer is 70% saturated
    exploration_saturation_threshold=0.3,   # lower bar during Phase 1
    default_efficiency_threshold=0.5,       # remove nodes below this
    exploration_efficiency_threshold=0.3,
    removal_efficiency_multiplier=0.5,      # remove if below threshold * 0.5
    initial_efficiency=0.5,
    efficiency_decay=0.9,
    efficiency_update_scale=0.1,
    efficiency_gradient_multiplier=10.0,
    max_removal_fraction_per_epoch=0.25     # remove at most 25% of a layer per epoch
)
```

---

### HealthScoreConfig

Controls the thresholds for cancer and alzheimer detection.

```python
from pydnn import HealthScoreConfig

config = HealthScoreConfig(
    cancer_denominator=5.0,         # normalisation for cancer score
    alzheimer_denominator=10.0,     # normalisation for alzheimer score
    layer_weight=10,
    healthy_threshold=0.3,          # below this is clearly healthy
    at_risk_threshold=0.7,          # above this triggers at-risk state
    alzheimer_ratio_threshold=0.8,  # 80% node removal triggers concern
    cancer_ratio_threshold=2.0      # 200% growth triggers concern
)
```

---

### RewardPenaltyConfig

Controls the reward/penalty system and emotional state logic.

```python
from pydnn import RewardPenaltyConfig

config = RewardPenaltyConfig(
    cost_improvement_threshold=0.005,         # 0.5% improvement needed for reward
    efficiency_improvement_threshold=0.02,    # 2% efficiency needed
    min_learning_rate=1e-5,
    max_learning_rate=0.5,
    baseline_learning_rate=0.1,               # LR resets here on extreme states
    max_adjustment_factor=1.5,                # penalty: LR * 1.5
    min_adjustment_factor=0.7,                # reward: LR * 0.7
    extreme_threshold=0.7,                    # full reset above this
    moderate_threshold=0.6,                   # partial correction above this
    window_size=15                            # how many epochs to look back
)
```

---

### GradientConfig

Controls gradient clipping and momentum.

```python
from pydnn import GradientConfig

config = GradientConfig(
    gradient_clip_value=1.0,  # clip gradients at this magnitude
    momentum=0.9
)
```

---

### PerturbationConfig

Controls the random noise applied during the early part of Phase 3.

```python
from pydnn import PerturbationConfig

config = PerturbationConfig(
    perturbation_fraction=0.005,  # 0.5% of nodes get perturbed per epoch
    perturbation_scale=0.01       # magnitude of the noise
)
```

---

### EarlyStoppingConfig

Controls when Phase 3 decides training has converged.

```python
from pydnn import EarlyStoppingConfig

config = EarlyStoppingConfig(
    window_size=30,                   # look back this many epochs
    improvement_threshold=1e-7,       # minimum improvement to count as progress
    max_consecutive_increases=5       # stop after this many consecutive cost increases
)
```

---

### NormalizationConfig

Controls automatic input and output normalisation.

```python
from pydnn import NormalizationConfig

config = NormalizationConfig(
    normalize_input=True,    # Z-score normalise inputs before training
    normalize_output=True,   # Z-score normalise outputs (for regression)
    method='zscore',         # 'zscore' or 'minmax'
    epsilon=1e-8             # prevents division by zero
)
```

When `normalize_output=True` (default), `predict()` automatically reverses the
normalisation so outputs are in the original scale of your data. Pass
`denormalize=False` to `predict()` if you want the raw network output.

---

### SigmoidThresholdConfig

Controls the adaptive sigmoid function that smoothly raises the saturation
threshold as Phase 3 progresses.

```python
from pydnn import SigmoidThresholdConfig

config = SigmoidThresholdConfig(
    k=5.0,        # steepness of the sigmoid
    base=0.3,     # minimum threshold value
    range_val=0.5,  # how much the threshold can rise from base
    center=0.5    # efficiency value that maps to the midpoint
)
```

At low efficiency the threshold is near `base` (0.3), allowing changes freely.
At high efficiency it rises toward `base + range_val` (0.8), making changes
more conservative.

---

*End of Developer Guide.*
