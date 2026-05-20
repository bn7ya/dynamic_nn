from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BaselineResult:
    name: str
    parameter_count: int
    train_seconds: float
    final_metric: float
    metric_name: str


def _train_torch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    device: str,
    metric: str,
) -> BaselineResult:
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_train)
        loss = nn.functional.cross_entropy(logits, y_train.argmax(1)
                                            if y_train.dim() > 1
                                            else y_train.long())
        loss.backward()
        opt.step()
    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        pred = logits.argmax(1)
        true = (y_test.argmax(1) if y_test.dim() > 1 else y_test.long())
        acc = (pred == true).float().mean().item()
    pc = sum(p.numel() for p in model.parameters())
    return BaselineResult(model.__class__.__name__, pc, elapsed, acc,
                            "accuracy")


class PyTorchMLPBaseline:
    @staticmethod
    def train(
        X_train, y_train, X_test, y_test,
        target_param_count: int,
        epochs: int = 60,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> BaselineResult:
        in_features = int(X_train.shape[1])
        out_features = int(y_train.shape[1]) if y_train.dim() > 1 else (
            int(y_train.max().item()) + 1)
        hidden = max(8, int((target_param_count -
                              in_features - out_features) /
                             max(1, in_features + out_features) / 2))
        model = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        return _train_torch_model(model, X_train, y_train,
                                    X_test, y_test, epochs, lr, device,
                                    "accuracy")


class PyTorchCNNBaseline:
    @staticmethod
    def train(
        X_train, y_train, X_test, y_test,
        target_param_count: int,
        epochs: int = 30,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> BaselineResult:
        in_ch = X_train.shape[1]
        h, w = X_train.shape[2], X_train.shape[3]
        out_features = int(y_train.shape[1]) if y_train.dim() > 1 else (
            int(y_train.max().item()) + 1)
        ch = max(4, int((target_param_count / (in_ch * 9 + 1)) ** 0.5))
        model = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch, ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ch * 2 * (h // 4) * (w // 4), out_features),
        )
        return _train_torch_model(model, X_train, y_train,
                                    X_test, y_test, epochs, lr, device,
                                    "accuracy")


class LightGBMBaseline:
    @staticmethod
    def train(X_train, y_train, X_test, y_test,
                n_estimators: int = 100,
                ) -> BaselineResult:
        import lightgbm as lgb
        import numpy as np
        Xtr = X_train.numpy() if torch.is_tensor(X_train) else X_train
        Xte = X_test.numpy() if torch.is_tensor(X_test) else X_test
        ytr = (y_train.argmax(1).numpy()
                if torch.is_tensor(y_train) and y_train.dim() > 1
                else (y_train.numpy() if torch.is_tensor(y_train) else y_train))
        yte = (y_test.argmax(1).numpy()
                if torch.is_tensor(y_test) and y_test.dim() > 1
                else (y_test.numpy() if torch.is_tensor(y_test) else y_test))
        start = time.time()
        n_classes = int(max(int(ytr.max()) + 1, 2))
        model = lgb.LGBMClassifier(n_estimators=n_estimators,
                                    verbose=-1, num_class=
                                    n_classes if n_classes > 2 else None,
                                    objective="multiclass"
                                    if n_classes > 2 else "binary")
        model.fit(Xtr, ytr)
        elapsed = time.time() - start
        preds = model.predict(Xte)
        acc = float(np.mean(preds == yte))
        params = int(model.booster_.num_trees() * 32)
        return BaselineResult("LightGBM", params, elapsed, acc, "accuracy")
