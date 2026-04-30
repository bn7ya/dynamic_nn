"""Standalone CI benchmark for pydnn vs static baseline.

Runs regression and classification experiments with a fast epoch budget so the
script finishes in ~60 seconds. Exits non-zero if quality thresholds are missed.

Usage:
    python notebooks/beanchmark/run_benchmark.py

Pytest integration (slow marker):
    pytest notebooks/beanchmark/run_benchmark.py -v
"""
from __future__ import annotations

import sys
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# Allow running from the repo root without installing pydnn
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from pydnn import DynamicNetwork, TrainingPhaseConfig, ArchitectureConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
CI_BUDGET = 20          # epochs for fast CI runs
REGRESSION_RMSE_LIMIT = 0.70   # pydnn must beat this RMSE (generous; task is noisy)
CLASSIFICATION_ACC_MIN = 0.90  # pydnn must clear this accuracy


# ---------------------------------------------------------------------------
# Data generators (same as notebook §5)
# ---------------------------------------------------------------------------

def make_regression_data(n=2000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (0.5 * X[:, 0] ** 2 - 0.3 * X[:, 1] + 0.2 * X[:, 2] * X[:, 3]
         + 0.1 * np.sin(X[:, 4]) + rng.standard_normal(n) * 0.1)
    return X, y[:, None].astype(np.float32)


def make_classification_data(n=3000, d=20, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3.0
    labels = rng.integers(0, n_classes, size=n)
    X = centers[labels] + rng.standard_normal((n, d)).astype(np.float32)
    y_onehot = np.eye(n_classes, dtype=np.float32)[labels]
    return X.astype(np.float32), y_onehot, labels


# ---------------------------------------------------------------------------
# pydnn helpers
# ---------------------------------------------------------------------------

def fit_pydnn(X, y, seed, cost, hidden=32, budget=CI_BUDGET):
    tp = TrainingPhaseConfig(
        exploration_epochs=3, estimation_epochs=3,
        min_estimated_epochs=budget, max_estimated_epochs=budget,
        phase4_enabled=False,
    )
    arch = ArchitectureConfig(
        max_layers=4, min_initial_hidden_size=hidden,
        max_nodes_per_layer=max(128, hidden * 4),
    )
    net = DynamicNetwork(
        input_shape=(X.shape[1],), output_size=y.shape[1],
        seed=seed, cost_function=cost, device='cpu',
        training_phase=tp, architecture=arch,
    )
    net._use_cpp = False  # keep Python path for mutation/health diagnostics
    t0 = time.perf_counter()
    res = net.fit(X, y, verbose=False)
    train_time = time.perf_counter() - t0
    return net, res, train_time


def count_params(net):
    if getattr(net, '_layers', None):
        total = 0
        prev = net.input_shape[0]
        for layer in net._layers:
            out = layer["W"].shape[0]
            total += prev * out + out
            prev = out
        return total
    return 0


def aggregate(results):
    if not results:
        return {}
    out = {}
    for k in results[0]:
        vals = [r[k] for r in results if r.get(k) is not None and not isinstance(r[k], list)]
        if vals and isinstance(vals[0], (int, float)):
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_regression():
    print("\n=== 6.1  Regression ===")
    results = []
    for seed in SEEDS:
        X, y = make_regression_data(seed=seed)
        n_tr = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
        net, res, t_train = fit_pydnn(Xtr, ytr, seed, cost='MSE', hidden=32, budget=CI_BUDGET)
        pred = net.predict(Xte).reshape(-1)
        rmse = float(np.sqrt(((pred - yte.reshape(-1)) ** 2).mean()))
        p = count_params(net)
        final_cancer    = res.cancer_score_history[-1]    if res.cancer_score_history    else 0.0
        final_alzheimer = res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0
        results.append(dict(rmse=rmse, params=p, train_s=t_train,
                            nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
                            final_cancer=final_cancer, final_alzheimer=final_alzheimer))
        print(f"  seed {seed}:  RMSE={rmse:.4f}  params={p}  train={t_train:.2f}s  "
              f"grow+prune={res.nodes_added}+{res.nodes_removed}  "
              f"cancer={final_cancer:.2%}  alzheimer={final_alzheimer:.2%}")
    agg = aggregate(results)
    mean_rmse = agg['rmse'][0]
    print(f"  mean RMSE={mean_rmse:.4f} ± {agg['rmse'][1]:.4f}  "
          f"mean params={agg['params'][0]:.0f}")
    ok = mean_rmse < REGRESSION_RMSE_LIMIT
    print(f"  {'PASS' if ok else 'FAIL'}: RMSE {mean_rmse:.4f} < limit {REGRESSION_RMSE_LIMIT}")
    return ok


def run_classification():
    print("\n=== 6.2  Classification ===")
    results = []
    for seed in SEEDS:
        X, y_oh, labels = make_classification_data(seed=seed)
        n_tr = int(0.8 * len(X))
        Xtr, ytr, Xte, yte_labels = X[:n_tr], y_oh[:n_tr], X[n_tr:], labels[n_tr:]
        net, res, t_train = fit_pydnn(Xtr, ytr, seed, cost='CrossEntropy', hidden=32, budget=CI_BUDGET)
        pred = net.predict(Xte)
        acc = float((pred.argmax(axis=1) == yte_labels).mean())
        p = count_params(net)
        final_cancer    = res.cancer_score_history[-1]    if res.cancer_score_history    else 0.0
        final_alzheimer = res.alzheimer_score_history[-1] if res.alzheimer_score_history else 0.0
        results.append(dict(acc=acc, params=p, train_s=t_train,
                            nodes_added=res.nodes_added, nodes_removed=res.nodes_removed,
                            final_cancer=final_cancer, final_alzheimer=final_alzheimer))
        print(f"  seed {seed}:  acc={acc:.4f}  params={p}  train={t_train:.2f}s  "
              f"grow+prune={res.nodes_added}+{res.nodes_removed}  "
              f"cancer={final_cancer:.2%}  alzheimer={final_alzheimer:.2%}")
    agg = aggregate(results)
    mean_acc = agg['acc'][0]
    print(f"  mean acc={mean_acc:.4f} ± {agg['acc'][1]:.4f}  "
          f"mean params={agg['params'][0]:.0f}")
    ok = mean_acc >= CLASSIFICATION_ACC_MIN
    print(f"  {'PASS' if ok else 'FAIL'}: acc {mean_acc:.4f} >= threshold {CLASSIFICATION_ACC_MIN}")
    return ok


# ---------------------------------------------------------------------------
# pytest entry points (optional)
# ---------------------------------------------------------------------------

def test_regression():
    assert run_regression(), f"pydnn regression RMSE exceeded {REGRESSION_RMSE_LIMIT}"


def test_classification():
    assert run_classification(), f"pydnn classification acc below {CLASSIFICATION_ACC_MIN}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.perf_counter()
    passed = []
    passed.append(run_regression())
    passed.append(run_classification())
    elapsed = time.perf_counter() - t_start
    n_pass = sum(passed)
    n_total = len(passed)
    print(f"\n{'='*50}")
    print(f"Results: {n_pass}/{n_total} passed  ({elapsed:.1f}s total)")
    if n_pass < n_total:
        sys.exit(1)
