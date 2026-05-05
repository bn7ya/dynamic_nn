"""
Building-block layers: Linear, LayerNorm, RMSNorm, FeedForward, SwiGLU,
GELU/SiLU activations, and Dropout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import autograd as A
from .autograd import Parameter, Tensor
from .module import Module


class Linear(Module):
    """Affine: ``y = x @ W^T + b``. ``W`` shape is (out, in)."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rng = rng if rng is not None else np.random.default_rng()
        # Kaiming-uniform-ish init (matches PyTorch default for Linear).
        bound = 1.0 / np.sqrt(in_features)
        w = rng.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(w)
        if bias:
            b = rng.uniform(-bound, bound, (out_features,)).astype(np.float32)
            self.bias: Optional[Parameter] = Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x @ W^T
        out = x @ self.weight.transpose(1, 0)
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """Standard layer normalisation over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = float(eps)
        self.weight = Parameter(np.ones((dim,), dtype=np.float32))
        self.bias = Parameter(np.zeros((dim,), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        normed = (x.data - mean) * inv
        out_data = normed * self.weight.data + self.bias.data
        out = Tensor(out_data, _children=(x, self.weight, self.bias), _op="layernorm")

        d = self.dim

        def _bw():
            g = out.grad
            # grad through gain/bias is straightforward
            gw = (g * normed).reshape(-1, d).sum(axis=0)
            gb = g.reshape(-1, d).sum(axis=0)
            self.weight.grad = gw if self.weight.grad is None else self.weight.grad + gw
            self.bias.grad = gb if self.bias.grad is None else self.bias.grad + gb
            # grad wrt x — standard layernorm derivation
            g_norm = g * self.weight.data
            mean_g = g_norm.mean(axis=-1, keepdims=True)
            mean_gn = (g_norm * normed).mean(axis=-1, keepdims=True)
            gx = (g_norm - mean_g - normed * mean_gn) * inv
            x.grad = gx if x.grad is None else x.grad + gx

        out._backward = _bw
        return out


class RMSNorm(Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Used in modern LLMs (LLaMA, T5). Faster than LayerNorm and works well.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = float(eps)
        self.weight = Parameter(np.ones((dim,), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        rms = np.sqrt((x.data ** 2).mean(axis=-1, keepdims=True) + self.eps)
        normed = x.data / rms
        out_data = normed * self.weight.data
        out = Tensor(out_data, _children=(x, self.weight), _op="rmsnorm")

        d = self.dim

        def _bw():
            g = out.grad
            gw = (g * normed).reshape(-1, d).sum(axis=0)
            self.weight.grad = gw if self.weight.grad is None else self.weight.grad + gw
            g_norm = g * self.weight.data
            # d/dx_i [x_i / rms] = (1/rms) * (delta_ij - x_i x_j / (d * rms^2))
            mean_gn_x = (g_norm * x.data).mean(axis=-1, keepdims=True)
            gx = (g_norm - x.data * mean_gn_x / (rms ** 2)) / rms
            x.grad = gx if x.grad is None else x.grad + gx

        out._backward = _bw
        return out


class Dropout(Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        return A.dropout(x, p=self.p, training=self.training)


class FeedForward(Module):
    """Standard transformer FFN: Linear -> activation -> Linear (+ dropout).

    Activation is one of {"gelu", "relu", "silu"}.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 activation: str = "gelu", dropout: float = 0.0,
                 bias: bool = True,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = Linear(dim, hidden_dim, bias=bias, rng=rng)
        self.fc2 = Linear(hidden_dim, dim, bias=bias, rng=rng)
        self.dropout = Dropout(dropout)
        self.activation = activation
        if activation == "gelu":
            self._act = A.gelu
        elif activation == "relu":
            self._act = A.relu
        elif activation == "silu":
            self._act = A.silu
        else:
            raise ValueError(f"unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self._act(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class SwiGLU(Module):
    """SwiGLU FFN as in LLaMA: ``(silu(x W1) * (x W3)) W2``.

    Hidden dim defaults to ``2/3 * 4 * dim`` rounded up to a multiple of 64,
    which keeps the parameter count comparable to a vanilla FFN with
    ``hidden_dim = 4 * dim``.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 dropout: float = 0.0, bias: bool = False,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if hidden_dim is None:
            raw = int((2.0 / 3.0) * 4 * dim)
            hidden_dim = ((raw + 63) // 64) * 64
            hidden_dim = max(hidden_dim, 64)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = Linear(dim, hidden_dim, bias=bias, rng=rng)
        self.w3 = Linear(dim, hidden_dim, bias=bias, rng=rng)
        self.w2 = Linear(hidden_dim, dim, bias=bias, rng=rng)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = A.silu(self.w1(x))
        up = self.w3(x)
        h = gate * up
        h = self.dropout(h)
        return self.w2(h)
