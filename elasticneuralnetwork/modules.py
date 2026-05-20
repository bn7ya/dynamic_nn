from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from elasticneuralnetwork._enn_core import modules as _m


ReversibleLinear = _m.ReversibleLinear
ReversibleNetwork = _m.ReversibleNetwork
AdaptiveConv2d = _m.AdaptiveConv2d
HiddenActivation = _m.HiddenActivation
ReversibleNetworkConfig = _m.ReversibleNetworkConfig


def _build_hidden_sizes(input_features: int, output_features: int,
                         depth: int) -> list[int]:
    if depth <= 0:
        return []
    geometric = []
    a = max(input_features, output_features)
    b = min(input_features, output_features)
    for i in range(1, depth + 1):
        ratio = i / (depth + 1)
        geometric.append(int(round(a * (b / a) ** ratio)))
    return geometric


class ElasticNetwork:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_size: int,
        seed: int = 0,
        hidden_depth: int = 1,
        hidden_activation: HiddenActivation = HiddenActivation.ReLU,
        cost_function: str = "CrossEntropy",
    ) -> None:
        torch.manual_seed(seed)
        self.input_shape = tuple(input_shape)
        self.output_size = output_size
        flat = 1
        for d in self.input_shape:
            flat *= d
        cfg = ReversibleNetworkConfig()
        cfg.input_features = flat
        cfg.output_features = output_size
        cfg.hidden_sizes = _build_hidden_sizes(flat, output_size,
                                                hidden_depth)
        cfg.hidden_activation = hidden_activation
        self.network = ReversibleNetwork(cfg)
        self.cost_function = cost_function
        self._last_result = None
        self._stats = None

    def _flatten(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(X.shape[0], -1) if X.dim() > 2 else X

    def fit(self, X, y):
        from elasticneuralnetwork.training import PhaseController, CostFunction
        cost = (CostFunction.CrossEntropy
                if self.cost_function == "CrossEntropy"
                else CostFunction.MSE)
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y)
        if cost == CostFunction.MSE:
            y = y.float()
        Xf = self._flatten(X)
        controller = PhaseController(self.network, cost)
        self._last_result = controller.fit(Xf, y)
        self._stats = controller.statistics()
        return self._last_result

    def _recovery_finetune(self, X, y, cost, epochs: int = 40,
                             lr: float = 1e-3, batch_size: int = 64) -> None:
        from elasticneuralnetwork.training import CostFunction
        params = list(self.network.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = X.shape[0]
        y_idx = (y.argmax(1) if y.dim() > 1 and cost == CostFunction.CrossEntropy
                  else (y.long() if cost == CostFunction.CrossEntropy else y))
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = self.network.forward(X[idx])
                if cost == CostFunction.CrossEntropy:
                    loss = torch.nn.functional.cross_entropy(
                        logits, y_idx[idx])
                else:
                    loss = torch.nn.functional.mse_loss(logits, y[idx])
                loss.backward()
                opt.step()

    def _final_prune_by_utilization(self, drop_fraction: float = 0.10) -> None:
        for i in range(self.network.num_layers() - 1):
            if not self.network.layer_active(i):
                continue
            layer = self.network.layer(i)
            weight = layer.weight.detach()
            row_norms = weight.abs().sum(dim=1)
            n = int(row_norms.numel())
            if n <= 4:
                continue
            num_drop = max(0, int(drop_fraction * n))
            num_drop = min(num_drop, n - 4)
            if num_drop == 0:
                continue
            threshold = row_norms.kthvalue(num_drop).values.item()
            to_prune = [int(j) for j in range(n)
                          if row_norms[j].item() <= threshold]
            to_prune = to_prune[:num_drop]
            layer.prune_nodes(to_prune)
        self.network.compact()

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        Xf = self._flatten(X)
        with torch.no_grad():
            return self.network.forward(Xf)

    def compact(self):
        return self.network.compact()

    def statistics(self):
        return self._stats

    def topology_version(self):
        return self.network.topology_version()

    def parameter_count(self):
        return sum(p.numel() for p in self.network.parameters())
