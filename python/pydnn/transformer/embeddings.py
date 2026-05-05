"""
Token embeddings and positional encodings.

Includes:
- ``TokenEmbedding`` (learned table)
- ``SinusoidalPositionalEncoding`` (no parameters; original Transformer)
- ``LearnedPositionalEncoding`` (parametric, BERT-style)
- ``RotaryEmbedding`` (RoPE; applied inside attention)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import autograd as A
from .autograd import Parameter, Tensor
from .module import Module


class TokenEmbedding(Module):
    """Learnable token embedding table."""

    def __init__(self, vocab_size: int, dim: int,
                 padding_idx: Optional[int] = None,
                 scale: bool = True,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.scale = scale
        rng = rng if rng is not None else np.random.default_rng()
        w = rng.standard_normal((vocab_size, dim)).astype(np.float32) * (1.0 / np.sqrt(dim))
        if padding_idx is not None:
            w[padding_idx] = 0.0
        self.weight = Parameter(w)

    def forward(self, ids: np.ndarray) -> Tensor:
        out = A.embedding(np.asarray(ids), self.weight)
        if self.scale:
            out = out * Tensor(np.float32(np.sqrt(self.dim)))
        return out


class SinusoidalPositionalEncoding(Module):
    """Fixed sin/cos positional encoding from "Attention Is All You Need"."""

    def __init__(self, dim: int, max_len: int = 8192) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        pe = np.zeros((max_len, dim), dtype=np.float32)
        pos = np.arange(0, max_len, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
        # Stored as a non-parameter constant.
        self._pe = pe

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        return x + Tensor(self._pe[None, :T, :])


class LearnedPositionalEncoding(Module):
    """BERT/GPT-style learned position embeddings."""

    def __init__(self, max_len: int, dim: int,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        self.max_len = max_len
        self.dim = dim
        w = rng.standard_normal((max_len, dim)).astype(np.float32) * 0.02
        self.weight = Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        ids = np.arange(T, dtype=np.int64)
        pos = A.embedding(ids, self.weight)
        return x + pos


class RotaryEmbedding:
    """Rotary positional embedding (RoPE).

    Not a ``Module`` — it has no learnable parameters. Call
    ``apply(q, k, offset=0)`` to rotate Q and K **inside** the
    attention block. ``q`` / ``k`` are expected with shape
    ``(B, H, T, head_dim)``; ``head_dim`` must be even.
    """

    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)  # (max_len, head_dim/2)
        self._cos = np.cos(freqs).astype(np.float32)
        self._sin = np.sin(freqs).astype(np.float32)

    def _rotate(self, x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
        # Split last dim into even / odd halves and rotate.
        x_data = x.data
        x_even = x_data[..., 0::2]
        x_odd = x_data[..., 1::2]
        rotated = np.empty_like(x_data)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos

        out = Tensor(rotated, _children=(x,), _op="rope")

        def _bw():
            g = out.grad
            g_even = g[..., 0::2]
            g_odd = g[..., 1::2]
            gx = np.empty_like(g)
            # Inverse rotation (transpose of rotation matrix)
            gx[..., 0::2] = g_even * cos + g_odd * sin
            gx[..., 1::2] = -g_even * sin + g_odd * cos
            x.grad = gx if x.grad is None else x.grad + gx

        out._backward = _bw
        return out

    def apply(self, q: Tensor, k: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        T = q.shape[-2]
        if offset + T > self.max_len:
            raise ValueError(
                f"RoPE max_len {self.max_len} exceeded by offset+T={offset + T}"
            )
        cos = self._cos[offset:offset + T][None, None, :, :]  # (1,1,T,d/2)
        sin = self._sin[offset:offset + T][None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
