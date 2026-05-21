# elasticneuralnetwork

Elastic neural networks with reversible topology, adaptive learning-rate
control, and stability monitoring — built on top of libtorch (PyTorch C++
API).

## Quickstart

```bash
pip install torch
pip install -e .

python -m elasticneuralnetwork.benchmarks.mnist --seed 42 --device cpu
```

```python
import torch
from elasticneuralnetwork import ElasticNetwork

X = torch.randn(64, 16)
y = torch.eye(4)[torch.randint(0, 4, (64,))]

model = ElasticNetwork(input_shape=(16,), output_size=4, seed=42)
result = model.fit(X, y)
preds = model.predict(X)
```

## Layout

- `csrc/` — C++17 sources (torch extension)
- `elasticneuralnetwork/` — Python package
- `tests/cpp/` — gtest C++ tests
- `tests/python/` — pytest Python tests
- `microbench/` — Google Benchmark microbenchmarks
- `pydnn/` (legacy, retired in Phase 8) — original project

## Goals

stability — reliability — memory efficiency. See `CLAUDE.md` files in each
directory for architectural details.
