# CLAUDE.md — `elasticneuralnetwork/benchmarks/`

## Purpose

Five-dataset benchmark suite that runs ENN against PyTorch MLP/CNN
and LightGBM baselines. Each per-dataset script writes a CSV row per
method + a PNG of the cost/utilization trajectories. `report.py`
aggregates everything and evaluates Phase 11 criterion 4 (ENN beats
baseline on ≥ 2 of 5 datasets with ≥ 20% parameter reduction).

The PyTorch baselines (`PyTorchMLPBaseline`, `PyTorchCNNBaseline` in
`baselines.py`) use stock `torch.nn.Sequential` plus
`torch.optim.Adam` plus a mini-batch SGD loop. That is exactly the
training loop ENN's `torch.nn.Module` subclasses are designed to slot
into — the head-to-head comparison shows ENN's reversible-topology
modules in the same harness as a plain PyTorch MLP, with the only
difference being the module itself.

## Files

| File | Role |
|---|---|
| `_common.py` | CLI argparse, RNG seeding, CSV / PNG writers, ENN runner. |
| `_tabular_common.py` | Shared CSV download + train/test split + standardisation. Includes `load_synthetic_classification` fallback for offline use. |
| `baselines.py` | `PyTorchMLPBaseline`, `PyTorchCNNBaseline`, `LightGBMBaseline`. MLP/CNN auto-size their hidden width to match ENN's final parameter count. |
| `mnist.py` | `python -m elasticneuralnetwork.benchmarks.mnist --seed S --device D` |
| `cifar10.py` | Same CLI. |
| `tabular_adult.py` | UCI Adult; falls back to synthetic Gaussian-cluster classification when the UCI mirror returns 4xx. |
| `tabular_covertype.py` | UCI Covertype (sub-sampled to 50k rows). |
| `tabular_higgs.py` | UCI Higgs (first 100k rows). |
| `report.py` | Aggregates `results/*.csv` and emits `summary.md` + `scatter.png` + the Phase 11 verdict. |

## Outputs

Every run writes under `results/`:

- `results/<dataset>_seed<N>.csv` — one row per method
  (elasticneuralnetwork, pytorch_mlp / cnn, lightgbm)
- `results/<dataset>_seed<N>_curves.png` — cost + utilization
  trajectories for ENN

## Reproducibility

- `torch.manual_seed`, `numpy.random.seed`, and per-`DataLoader`
  generators all derive from `--seed`.
- LightGBM's tree ensemble is deterministic for a fixed `random_state`
  (we accept the default, which is `0`).
- Residual non-determinism: OpenMP reduction order on the CPU
  backend (libtorch may reorder summation across threads). This
  produces sub-percent variation in final accuracy.

## Cross-refs

- Calling code: [`../modules.py`](../modules.py) (`ElasticNetwork`).
- Verification gate: [`report.py`](report.py)
  `verification_criterion_4`.
