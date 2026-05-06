"""
Minimal NumPy-backed autograd engine for the transformer subpackage.

This is a small reverse-mode autodiff Tensor type built on top of
``numpy.ndarray``. It is *not* meant to compete with PyTorch / JAX —
it exists so that the transformer / MoE / encoder / decoder code in
``pydnn.transformer`` can run end-to-end (forward + backward) without
adding a heavy ML framework dependency, while still composing nicely
with the rest of ``pydnn`` (NumPy in, NumPy out).

Design:

- ``Tensor`` wraps a ``np.ndarray`` and records the op graph as a
  set of parent Tensors plus a closure ``_backward``.
- Calling ``.backward()`` on a scalar (or any tensor with an explicit
  upstream grad) walks the graph in topological order and accumulates
  grads into ``parent.grad``.
- ``Parameter`` is just a Tensor with ``requires_grad=True`` that
  modules collect via ``Module.parameters()``.

Only the ops the transformer stack actually needs are implemented:
matmul, add, sub, mul, neg, sum, mean, exp, log, softmax, transpose,
reshape, gather (embedding lookup), masking, layernorm/rmsnorm,
gelu, silu, dropout. Everything else can be expressed in terms of
those.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _backend as _bk


ArrayLike = Union[np.ndarray, "Tensor", float, int, list]


def _as_array(x: ArrayLike, dtype=np.float32) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x, dtype=dtype)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` so its shape matches ``shape`` (inverse of broadcast)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A NumPy-backed tensor with reverse-mode autograd."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        self.data = _as_array(data)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev: Tuple["Tensor", ...] = tuple(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad})"
        )

    # ---------------------------------------------------------------
    # Graph machinery
    # ---------------------------------------------------------------

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        topo: List[Tensor] = []
        visited: set = set()

        def build(v: Tensor) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() without an explicit grad is only valid on a "
                    f"scalar tensor (got shape {self.shape})"
                )
            grad = np.ones_like(self.data)
        else:
            grad = _as_array(grad)

        self.grad = grad if self.grad is None else self.grad + grad

        for v in reversed(topo):
            v._backward()

    # ---------------------------------------------------------------
    # Math ops
    # ---------------------------------------------------------------

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="add")

        def _bw():
            if self.requires_grad or self.grad is not None or _grad_required(self):
                g = _unbroadcast(out.grad, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad or other.grad is not None or _grad_required(other):
                g = _unbroadcast(out.grad, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _bw
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, _children=(self,), _op="neg")

        def _bw():
            g = -out.grad
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _bw
        return out

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else Tensor(-_as_array(other)))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return Tensor(_as_array(other)) - self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="mul")

        def _bw():
            g_self = _unbroadcast(out.grad * other.data, self.shape)
            g_other = _unbroadcast(out.grad * self.data, other.shape)
            self.grad = g_self if self.grad is None else self.grad + g_self
            other.grad = g_other if other.grad is None else other.grad + g_other

        out._backward = _bw
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op="div")

        def _bw():
            g_self = _unbroadcast(out.grad / other.data, self.shape)
            g_other = _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
            self.grad = g_self if self.grad is None else self.grad + g_self
            other.grad = g_other if other.grad is None else other.grad + g_other

        out._backward = _bw
        return out

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return Tensor(_as_array(other)) / self

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, _children=(self,), _op=f"pow{power}")

        def _bw():
            g = out.grad * power * (self.data ** (power - 1))
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _bw
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    # ---------------------------------------------------------------
    # Shape ops
    # ---------------------------------------------------------------

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), _children=(self,), _op="reshape")

        def _bw():
            g = out.grad.reshape(self.shape)
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _bw
        return out

    def transpose(self, *axes: int) -> "Tensor":
        if not axes:
            axes_t = tuple(reversed(range(self.ndim)))
        else:
            axes_t = tuple(axes)
        out = Tensor(self.data.transpose(axes_t), _children=(self,), _op="transpose")

        inv = np.argsort(axes_t)

        def _bw():
            g = out.grad.transpose(tuple(inv))
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _bw
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op="sum")

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)
            g = np.broadcast_to(g, self.shape).copy()
            self.grad = g if self.grad is None else self.grad + g

        out._backward = _bw
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            ax = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.shape[a] for a in ax]))
        return self.sum(axis=axis, keepdims=keepdims) / Tensor(np.float32(n))


def _grad_required(t: Tensor) -> bool:
    """A node needs its grad accumulated if it has parents (an op produced it)."""
    return bool(t._prev) or t.requires_grad


# ---------------------------------------------------------------
# Free functions / ops
# ---------------------------------------------------------------


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Batched matmul. Supports broadcasting on leading dims.

    Forward and the two backward gemms route through
    :mod:`pydnn.transformer._backend`, so the cpu / cuda kernels in
    ``_dnn_core.transformer_ops`` are used when available. The pure
    NumPy semantics are preserved bit-equivalently to within fp32 noise.
    """
    out = Tensor(_bk.matmul(a.data, b.data), _children=(a, b), _op="matmul")

    def _bw():
        g = out.grad
        # ga = g @ b^T  (along the matrix dims)
        ga = _bk.matmul(g, b.data, transpose_b=True)
        # gb = a^T @ g
        gb = _bk.matmul(a.data, g, transpose_a=True)
        ga = _unbroadcast(ga, a.shape)
        gb = _unbroadcast(gb, b.shape)
        a.grad = ga if a.grad is None else a.grad + ga
        b.grad = gb if b.grad is None else b.grad + gb

    out._backward = _bw
    return out


def exp(x: Tensor) -> Tensor:
    out_data = np.exp(x.data)
    out = Tensor(out_data, _children=(x,), _op="exp")

    def _bw():
        g = out.grad * out_data
        x.grad = g if x.grad is None else x.grad + g

    out._backward = _bw
    return out


def log(x: Tensor, eps: float = 1e-12) -> Tensor:
    out = Tensor(np.log(x.data + eps), _children=(x,), _op="log")

    def _bw():
        g = out.grad / (x.data + eps)
        x.grad = g if x.grad is None else x.grad + g

    out._backward = _bw
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax with autograd."""
    out_data = _bk.softmax(x.data, axis=axis)
    out = Tensor(out_data, _children=(x,), _op="softmax")

    def _bw():
        gx = _bk.softmax_backward(out_data, out.grad, axis=axis)
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    out_data = shifted - log_z
    out = Tensor(out_data, _children=(x,), _op="log_softmax")

    def _bw():
        g = out.grad
        # softmax = exp(out)
        sm = np.exp(out_data)
        gx = g - sm * g.sum(axis=axis, keepdims=True)
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def gelu(x: Tensor) -> Tensor:
    """Approximate GELU (tanh formulation)."""
    out = Tensor(_bk.gelu(x.data), _children=(x,), _op="gelu")

    def _bw():
        gx = _bk.gelu_backward(x.data, out.grad)
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish: x * sigmoid(x). Used in SwiGLU FFN."""
    out = Tensor(_bk.silu(x.data), _children=(x,), _op="silu")

    def _bw():
        gx = _bk.silu_backward(x.data, out.grad)
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def relu(x: Tensor) -> Tensor:
    mask = (x.data > 0).astype(x.data.dtype)
    out = Tensor(x.data * mask, _children=(x,), _op="relu")

    def _bw():
        gx = out.grad * mask
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def dropout(x: Tensor, p: float = 0.0, training: bool = True) -> Tensor:
    if not training or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = (np.random.random(x.shape) < keep).astype(x.data.dtype) / keep
    out = Tensor(x.data * mask, _children=(x,), _op="dropout")

    def _bw():
        gx = out.grad * mask
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def embedding(ids: np.ndarray, weight: Tensor) -> Tensor:
    """Look up rows of ``weight`` by integer ``ids``.

    ``ids`` is plain numpy (no autograd through indices). Output shape is
    ``ids.shape + (weight.shape[-1],)``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    out_data = _bk.embedding_forward(weight.data, ids)
    out = Tensor(out_data, _children=(weight,), _op="embedding")

    def _bw():
        gw = _bk.embedding_backward(out.grad, ids,
                                    weight.shape[0], weight.shape[1])
        weight.grad = gw if weight.grad is None else weight.grad + gw

    out._backward = _bw
    return out


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    arrays = [t.data for t in tensors]
    out_data = np.concatenate(arrays, axis=axis)
    out = Tensor(out_data, _children=tuple(tensors), _op="concat")

    sizes = [t.shape[axis] for t in tensors]
    splits = np.cumsum(sizes)[:-1]

    def _bw():
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            t.grad = g if t.grad is None else t.grad + g

    out._backward = _bw
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    out_data = np.stack([t.data for t in tensors], axis=axis)
    out = Tensor(out_data, _children=tuple(tensors), _op="stack")

    def _bw():
        grads = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, grads):
            g_squeezed = np.squeeze(g, axis=axis)
            t.grad = g_squeezed if t.grad is None else t.grad + g_squeezed

    out._backward = _bw
    return out


def masked_fill(x: Tensor, mask: np.ndarray, value: float) -> Tensor:
    """Set positions where ``mask`` is True to ``value`` (no grad through mask)."""
    mask = np.asarray(mask, dtype=bool)
    out_data = np.where(mask, np.float32(value), x.data)
    out = Tensor(out_data, _children=(x,), _op="masked_fill")
    keep = (~mask).astype(x.data.dtype)

    def _bw():
        gx = out.grad * keep
        x.grad = gx if x.grad is None else x.grad + gx

    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray,
                  ignore_index: int = -100) -> Tensor:
    """Mean cross-entropy. ``logits`` is (..., V), ``targets`` is (...) ints.

    Positions where ``targets == ignore_index`` are excluded from the mean.
    """
    targets = np.asarray(targets, dtype=np.int64)
    loss_val, log_probs, valid_count = _bk.xent_forward(
        logits.data, targets, int(ignore_index))

    out = Tensor(np.float32(loss_val), _children=(logits,), _op="cross_entropy")

    def _bw():
        g = _bk.xent_backward(log_probs, targets,
                              float(out.grad), int(valid_count),
                              int(ignore_index))
        logits.grad = g if logits.grad is None else logits.grad + g

    out._backward = _bw
    return out


# ---------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------


def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(shape: Tuple[int, ...], scale: float = 1.0,
          requires_grad: bool = False, rng: Optional[np.random.Generator] = None) -> Tensor:
    rng = rng if rng is not None else np.random.default_rng()
    return Tensor(rng.standard_normal(shape).astype(np.float32) * scale,
                  requires_grad=requires_grad)


class Parameter(Tensor):
    """A learnable Tensor; just a marker subclass with ``requires_grad=True``."""

    def __init__(self, data: ArrayLike):
        super().__init__(data, requires_grad=True)
