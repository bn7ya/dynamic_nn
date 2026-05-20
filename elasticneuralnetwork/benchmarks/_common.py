from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch


CACHE_DIR = Path.home() / ".cache" / "elasticneuralnetwork"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkRow:
    dataset: str
    method: str
    seed: int
    device: str
    accuracy: float
    parameter_count: int
    train_seconds: float
    topology_version_final: Optional[int] = None
    epochs_completed: Optional[int] = None


def make_argparser(default_seed: int = 42) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=default_seed)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--limit-samples", type=int, default=0,
                    help="If >0, subsample the training set to this size")
    p.add_argument("--epochs-cap", type=int, default=0,
                    help="If >0, cap baseline training epochs for speed")
    return p


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)


def write_rows_csv(dataset: str, seed: int,
                    rows: list[BenchmarkRow]) -> Path:
    import csv
    path = RESULTS_DIR / f"{dataset}_seed{seed}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return path


def write_curves_png(dataset: str, seed: int,
                      cost_trajectory: list[float],
                      utilization_trajectory: list[float]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(cost_trajectory, color="C0")
    ax1.set_title("cost")
    ax1.set_xlabel("epoch")
    ax2.plot(utilization_trajectory, color="C1")
    ax2.set_title("utilization")
    ax2.set_xlabel("epoch")
    fig.suptitle(f"{dataset} seed={seed}")
    fig.tight_layout()
    path = RESULTS_DIR / f"{dataset}_seed{seed}_curves.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def run_enn(X_train, y_train, X_test, y_test, seed: int,
             input_shape, output_size, device: str):
    from elasticneuralnetwork import ElasticNetwork
    model = ElasticNetwork(input_shape=input_shape,
                            output_size=output_size, seed=seed,
                            hidden_depth=2)
    start = time.time()
    result = model.fit(X_train, y_train)
    elapsed = time.time() - start
    preds = model.predict(X_test)
    pred_labels = preds.argmax(1)
    true_labels = (y_test.argmax(1) if y_test.dim() > 1 else
                    y_test.long())
    acc = (pred_labels == true_labels).float().mean().item()
    return {
        "accuracy": acc,
        "parameter_count": model.parameter_count(),
        "train_seconds": elapsed,
        "topology_version_final": result.topology_version_final,
        "epochs_completed": result.epochs_completed,
        "cost_trajectory": list(result.cost_trajectory),
        "utilization_trajectory": list(result.utilization_trajectory),
    }
