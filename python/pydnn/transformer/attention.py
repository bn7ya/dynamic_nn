"""
Attention modules.

- ``scaled_dot_product_attention`` — primitive (B, H, T, d) op
- ``MultiHeadAttention`` — standard MHA with optional causal mask + RoPE
- ``GroupedQueryAttention`` — modern GQA / MQA (LLaMA-2 style)
- ``KVCache`` — incremental decoding cache
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import autograd as A
from .autograd import Tensor
from .embeddings import RotaryEmbedding
from .layers import Linear
from .module import Module


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Compute attention(Q, K, V).

    Inputs are shape ``(B, H, T_q, d)`` for Q and ``(B, H, T_k, d)`` for K/V.
    ``mask`` is a boolean array broadcastable to ``(B, H, T_q, T_k)`` where
    ``True`` means *masked out* (i.e. set to -inf before softmax).
    """
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    # (B, H, T_q, d) @ (B, H, d, T_k) -> (B, H, T_q, T_k)
    scores = (q @ k.transpose(0, 1, 3, 2)) * Tensor(np.float32(scale))
    if mask is not None:
        scores = A.masked_fill(scores, mask, -1e9)
    weights = A.softmax(scores, axis=-1)
    weights = A.dropout(weights, p=dropout_p, training=training)
    return weights @ v


@dataclass
class KVCache:
    """Incremental K/V buffers for autoregressive decoding."""
    k: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None

    def length(self) -> int:
        return 0 if self.k is None else int(self.k.shape[2])

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = np.concatenate([self.k, k_new], axis=2)
            self.v = np.concatenate([self.v, v_new], axis=2)
        return self.k, self.v

    def reset(self) -> None:
        self.k = None
        self.v = None


def _causal_mask(T_q: int, T_k: int, offset: int = 0) -> np.ndarray:
    """Lower-triangular mask. ``True`` where position should be hidden."""
    # For incremental decoding, query position i corresponds to absolute
    # index ``offset + i`` and may attend to keys 0..offset+i.
    q_idx = np.arange(T_q)[:, None] + offset
    k_idx = np.arange(T_k)[None, :]
    return k_idx > q_idx  # (T_q, T_k)


class MultiHeadAttention(Module):
    """Standard multi-head attention with Q/K/V/Out projections.

    If ``rotary`` is provided, RoPE is applied to Q and K. If ``causal`` is
    True, a causal mask is built automatically.
    """

    def __init__(self, dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True,
                 causal: bool = False,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.dropout_p = float(dropout)
        self.rotary = rotary
        self.q_proj = Linear(dim, dim, bias=bias, rng=rng)
        self.k_proj = Linear(dim, dim, bias=bias, rng=rng)
        self.v_proj = Linear(dim, dim, bias=bias, rng=rng)
        self.out_proj = Linear(dim, dim, bias=bias, rng=rng)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def forward(self,
                x: Tensor,
                context: Optional[Tensor] = None,
                attn_mask: Optional[np.ndarray] = None,
                key_padding_mask: Optional[np.ndarray] = None,
                kv_cache: Optional[KVCache] = None) -> Tensor:
        """Run attention.

        - ``x`` is the query source. If ``context`` is provided, K/V come
          from it (cross-attention); otherwise from ``x`` (self-attention).
        - ``attn_mask`` shape: ``(T_q, T_k)`` or broadcastable. Bool, True = mask.
        - ``key_padding_mask`` shape: ``(B, T_k)``. Bool, True = pad / mask.
        - ``kv_cache`` enables incremental decoding (self-attention only).
        """
        kv_source = context if context is not None else x

        q = self._split_heads(self.q_proj(x))         # (B, H, T_q, d)
        k = self._split_heads(self.k_proj(kv_source))  # (B, H, T_k, d)
        v = self._split_heads(self.v_proj(kv_source))

        offset = 0
        if kv_cache is not None and context is None:
            offset = kv_cache.length()
            cached_k, cached_v = kv_cache.append(k.data, v.data)
            # Replace k/v with the appended buffer; we don't need autograd
            # through cached tensors at inference time.
            k = Tensor(cached_k)
            v = Tensor(cached_v)

        if self.rotary is not None:
            q, k = self.rotary.apply(q, k, offset=offset)

        # Build mask
        T_q = q.shape[2]
        T_k = k.shape[2]
        mask = None
        if self.causal:
            mask = _causal_mask(T_q, T_k, offset=offset)
            mask = mask[None, None, :, :]  # (1,1,T_q,T_k)
        if attn_mask is not None:
            am = np.asarray(attn_mask, dtype=bool)
            while am.ndim < 4:
                am = am[None, ...]
            mask = am if mask is None else (mask | am)
        if key_padding_mask is not None:
            kpm = np.asarray(key_padding_mask, dtype=bool)[:, None, None, :]
            mask = kpm if mask is None else (mask | kpm)

        attn = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout_p, training=self.training,
        )
        out = self._merge_heads(attn)
        return self.out_proj(out)


class GroupedQueryAttention(Module):
    """Grouped-Query / Multi-Query Attention (LLaMA-2 / Mistral style).

    K and V are projected with ``num_kv_heads`` heads (typically << ``num_heads``)
    and then repeated to match the query heads. ``num_kv_heads = 1`` reproduces
    classic Multi-Query Attention.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 dropout: float = 0.0, bias: bool = False,
                 causal: bool = False,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.repeat = num_heads // num_kv_heads
        self.causal = causal
        self.dropout_p = float(dropout)
        self.rotary = rotary
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = Linear(dim, dim, bias=bias, rng=rng)
        self.k_proj = Linear(dim, kv_dim, bias=bias, rng=rng)
        self.v_proj = Linear(dim, kv_dim, bias=bias, rng=rng)
        self.out_proj = Linear(dim, dim, bias=bias, rng=rng)

    def _split(self, x: Tensor, n_heads: int) -> Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        # (B, H_kv, T, d) -> (B, H, T, d) by repeating along head axis.
        if self.repeat == 1:
            return x
        B, Hkv, T, D = x.shape
        # Use plain numpy repeat then re-wrap; KV repeat is non-learnable.
        repeated = np.repeat(x.data, self.repeat, axis=1)
        out = Tensor(repeated, _children=(x,), _op="repeat_kv")
        repeat = self.repeat

        def _bw():
            g = out.grad.reshape(B, Hkv, repeat, T, D).sum(axis=2)
            x.grad = g if x.grad is None else x.grad + g

        out._backward = _bw
        return out

    def forward(self,
                x: Tensor,
                context: Optional[Tensor] = None,
                attn_mask: Optional[np.ndarray] = None,
                key_padding_mask: Optional[np.ndarray] = None,
                kv_cache: Optional[KVCache] = None) -> Tensor:
        kv_source = context if context is not None else x

        q = self._split(self.q_proj(x), self.num_heads)
        k = self._split(self.k_proj(kv_source), self.num_kv_heads)
        v = self._split(self.v_proj(kv_source), self.num_kv_heads)

        offset = 0
        if kv_cache is not None and context is None:
            offset = kv_cache.length()
            cached_k, cached_v = kv_cache.append(k.data, v.data)
            k = Tensor(cached_k)
            v = Tensor(cached_v)

        if self.rotary is not None:
            q, k = self.rotary.apply(q, k, offset=offset)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        T_q, T_k = q.shape[2], k.shape[2]
        mask = None
        if self.causal:
            mask = _causal_mask(T_q, T_k, offset=offset)[None, None, :, :]
        if attn_mask is not None:
            am = np.asarray(attn_mask, dtype=bool)
            while am.ndim < 4:
                am = am[None, ...]
            mask = am if mask is None else (mask | am)
        if key_padding_mask is not None:
            kpm = np.asarray(key_padding_mask, dtype=bool)[:, None, None, :]
            mask = kpm if mask is None else (mask | kpm)

        attn = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout_p, training=self.training,
        )
        B, H, T, D = attn.shape
        out = attn.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.out_proj(out)
