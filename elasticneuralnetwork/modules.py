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
        hidden_depth: int = 2,
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
