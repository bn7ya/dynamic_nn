from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import torch

from elasticneuralnetwork.benchmarks._common import CACHE_DIR


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urlretrieve(url, dest)
    return dest


def _categorize_columns(df, target_col: str):
    import pandas as pd
    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    feature_cols = [c for c in df.columns if c != target_col]
    return df, feature_cols


def _to_tensors(df, feature_cols, target_col: str):
    import numpy as np
    import pandas as pd
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col].to_numpy()
    if y_raw.dtype == object:
        classes = sorted(set(y_raw.tolist()))
        mapping = {c: i for i, c in enumerate(classes)}
        y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
        num_classes = len(classes)
    else:
        y = y_raw.astype(np.int64)
        num_classes = int(y.max()) + 1
    return torch.from_numpy(X), torch.from_numpy(y), num_classes


def split_train_test(X: torch.Tensor, y: torch.Tensor,
                       train_frac: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=gen)
    n_train = int(train_frac * X.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize(X_train: torch.Tensor, X_test: torch.Tensor):
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp_min(1e-6)
    return (X_train - mean) / std, (X_test - mean) / std


def load_synthetic_classification(name: str, seed: int,
                                    n_samples: int = 4000,
                                    n_features: int = 16,
                                    n_classes: int = 4):
    gen = torch.Generator().manual_seed(seed + hash(name) % 100)
    centers = torch.randn(n_classes, n_features, generator=gen) * 2.0
    y = torch.randint(0, n_classes, (n_samples,), generator=gen)
    X = centers[y] + torch.randn(n_samples, n_features, generator=gen)
    Xtr, ytr, Xte, yte = split_train_test(X, y, 0.8, seed)
    Xtr, Xte = standardize(Xtr, Xte)
    return Xtr, ytr, Xte, yte, n_classes
