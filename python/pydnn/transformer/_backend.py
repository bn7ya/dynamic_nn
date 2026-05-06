"""Backend dispatcher for the transformer autograd ops.

The transformer subpackage's autograd engine in :mod:`autograd` (and the
hand-rolled forward/backward in :mod:`layers`) call into this module
instead of using NumPy directly. We have three possible backends:

- ``numpy`` — the original pure-NumPy reference. Always available;
  this is the regression baseline.
- ``cpu``   — the C++ kernels in ``_dnn_core.transformer_ops`` (OpenMP
  parallel). Available when the C++ extension was built.
- ``cuda``  — the CUDA kernels in the same submodule. Available when
  the extension was built with ``DNN_ENABLE_CUDA=ON`` *and* a device
  is present.

The grow/prune logic in :mod:`pydnn.transformer.dynamic` is unchanged:
it composes the same Module / autograd primitives, which now happen
to be faster. Mid-training topology mutations still go through the
ModuleList; the optimizer rebuilds its parameter view each step.

Selection
---------

Resolved once at import time, but overridable via
``PYDNN_TRANSFORMER_BACKEND={numpy|cpu|cuda}`` for testing parity.
Inside hot ops we re-check ``current_backend()`` so users who flip the
env var (or call :func:`set_backend`) between epochs see the change.

Numerical contract
------------------

The C++ kernels target *bit-equivalent* outputs to the NumPy reference
within ~1e-5 relative tolerance for fp32. Tests in
``python/pydnn/transformer/tests/test_backend_parity.py`` enforce that.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# Import the native ops submodule lazily; we want to keep the package
# import-cheap (per the transformer/CLAUDE.md import-cheap invariant).
try:
    from .. import _dnn_core  # type: ignore
    _ops = getattr(_dnn_core, "transformer_ops", None)
except Exception:  # pragma: no cover - extension simply not built
    _ops = None


_VALID_BACKENDS = ("numpy", "cpu", "cuda")


def _resolve_default() -> str:
    env = os.environ.get("PYDNN_TRANSFORMER_BACKEND", "").strip().lower()
    if env in _VALID_BACKENDS:
        if env == "cuda" and not _cuda_ready():
            return "cpu" if _ops is not None else "numpy"
        if env == "cpu" and _ops is None:
            return "numpy"
        return env
    if _ops is None:
        return "numpy"
    if _cuda_ready():
        return "cuda"
    return "cpu"


def _cuda_ready() -> bool:
    return bool(_ops is not None and getattr(_ops, "CUDA_AVAILABLE", False))


_backend: str = _resolve_default()


def current_backend() -> str:
    return _backend


def set_backend(name: str) -> None:
    global _backend
    name = name.lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {_VALID_BACKENDS}")
    if name == "cpu" and _ops is None:
        raise RuntimeError("native cpu backend not available; rebuild with DNN_BUILD_PYTHON=ON")
    if name == "cuda" and not _cuda_ready():
        raise RuntimeError("cuda backend not available; rebuild with DNN_ENABLE_CUDA=ON")
    _backend = name


def available_backends() -> Tuple[str, ...]:
    out = ["numpy"]
    if _ops is not None:
        out.append("cpu")
        if _cuda_ready():
            out.append("cuda")
    return tuple(out)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _f32(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float32 and a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a, dtype=np.float32)


def _i64(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.int64 and a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a, dtype=np.int64)


def _native_for(name: str):
    """Return the bound function for `name` on the active backend, or None.

    We try ``<name>_<backend>`` first, falling back to the CPU variant
    if the CUDA twin isn't bound (e.g. layernorm backward — currently
    CPU-only on the host-numpy side; the CUDA matmul/softmax/activations
    are wired below).
    """
    if _ops is None or _backend == "numpy":
        return None
    fn = getattr(_ops, f"{name}_{_backend}", None)
    if fn is not None:
        return fn
    # CUDA selected but specific op is CPU-only: degrade silently to CPU.
    return getattr(_ops, f"{name}_cpu", None)


# ----------------------------------------------------------------
# Op wrappers. Each one takes/returns NumPy and matches the original
# pure-NumPy semantics in `autograd.py` / `layers.py`.
# ----------------------------------------------------------------

def matmul(a: np.ndarray, b: np.ndarray,
           transpose_a: bool = False, transpose_b: bool = False) -> np.ndarray:
    fn = _native_for("matmul")
    if fn is None:
        a2 = a.swapaxes(-1, -2) if transpose_a else a
        b2 = b.swapaxes(-1, -2) if transpose_b else b
        return a2 @ b2
    return fn(_f32(a), _f32(b), transpose_a, transpose_b)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # The native kernel only handles last-axis softmax; transpose for
    # other axes (rare in this codebase — attention, MoE gating, and
    # cross_entropy are all last-axis).
    if axis not in (-1, x.ndim - 1):
        shifted = x - x.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=axis, keepdims=True)
    fn = _native_for("softmax")
    if fn is None:
        shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)
    return fn(_f32(x))


def softmax_backward(y: np.ndarray, dy: np.ndarray, axis: int = -1) -> np.ndarray:
    if axis not in (-1, y.ndim - 1):
        dot = (dy * y).sum(axis=axis, keepdims=True)
        return y * (dy - dot)
    fn = _native_for("softmax_backward")
    if fn is None:
        dot = (dy * y).sum(axis=-1, keepdims=True)
        return y * (dy - dot)
    return fn(_f32(y), _f32(dy))


def gelu(x: np.ndarray) -> np.ndarray:
    fn = _native_for("gelu")
    if fn is None:
        c = np.float32(np.sqrt(2.0 / np.pi))
        a = np.float32(0.044715)
        t = np.tanh(c * (x + a * x ** 3))
        return 0.5 * x * (1.0 + t)
    return fn(_f32(x))


def gelu_backward(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    fn = _native_for("gelu_backward")
    if fn is None:
        c = np.float32(np.sqrt(2.0 / np.pi))
        a = np.float32(0.044715)
        inner = c * (x + a * x ** 3)
        t = np.tanh(inner)
        sech2 = 1.0 - t ** 2
        d_inner = c * (1.0 + 3.0 * a * x ** 2)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner)
    return fn(_f32(x), _f32(dy))


def silu(x: np.ndarray) -> np.ndarray:
    fn = _native_for("silu")
    if fn is None:
        s = 1.0 / (1.0 + np.exp(-x))
        return x * s
    return fn(_f32(x))


def silu_backward(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    fn = _native_for("silu_backward")
    if fn is None:
        s = 1.0 / (1.0 + np.exp(-x))
        return dy * (s + x * s * (1.0 - s))
    return fn(_f32(x), _f32(dy))


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fn = _native_for("layernorm_forward")
    if fn is None:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        y = (x - mean) * inv * gamma + beta
        # Flatten mean/inv_std so callers don't need to special-case shape.
        rows = int(np.prod(x.shape[:-1]))
        return y.astype(x.dtype, copy=False), mean.reshape(rows), inv.reshape(rows)
    return fn(_f32(x), _f32(gamma), _f32(beta), float(eps))


def layernorm_backward(x: np.ndarray, gamma: np.ndarray, mean: np.ndarray,
                       inv_std: np.ndarray, dy: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fn = _native_for("layernorm_backward")
    if fn is None:
        cols = x.shape[-1]
        m = mean.reshape(*x.shape[:-1], 1)
        inv = inv_std.reshape(*x.shape[:-1], 1)
        normed = (x - m) * inv
        gw = (dy * normed).reshape(-1, cols).sum(axis=0)
        gb = dy.reshape(-1, cols).sum(axis=0)
        g_norm = dy * gamma
        mean_g = g_norm.mean(axis=-1, keepdims=True)
        mean_gn = (g_norm * normed).mean(axis=-1, keepdims=True)
        dx = (g_norm - mean_g - normed * mean_gn) * inv
        return dx, gw, gb
    return fn(_f32(x), _f32(gamma), _f32(mean), _f32(inv_std), _f32(dy))


def rmsnorm_forward(x: np.ndarray, gamma: np.ndarray, eps: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    fn = _native_for("rmsnorm_forward")
    if fn is None:
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
        y = (x / rms) * gamma
        rows = int(np.prod(x.shape[:-1]))
        return y.astype(x.dtype, copy=False), (1.0 / rms).reshape(rows)
    return fn(_f32(x), _f32(gamma), float(eps))


def rmsnorm_backward(x: np.ndarray, gamma: np.ndarray, inv_rms: np.ndarray,
                     dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fn = _native_for("rmsnorm_backward")
    if fn is None:
        cols = x.shape[-1]
        inv = inv_rms.reshape(*x.shape[:-1], 1)
        normed = x * inv
        gw = (dy * normed).reshape(-1, cols).sum(axis=0)
        g_norm = dy * gamma
        mean_gn_x = (g_norm * x).mean(axis=-1, keepdims=True)
        dx = (g_norm - x * mean_gn_x * (inv ** 2)) * inv
        return dx, gw
    return fn(_f32(x), _f32(gamma), _f32(inv_rms), _f32(dy))


def embedding_forward(weight: np.ndarray, ids: np.ndarray) -> np.ndarray:
    fn = _native_for("embedding_forward")
    if fn is None:
        return weight[ids]
    out = fn(_f32(weight), _i64(ids))
    return out.reshape(*ids.shape, weight.shape[-1])


def embedding_backward(dy: np.ndarray, ids: np.ndarray, vocab: int, dim: int) -> np.ndarray:
    fn = _native_for("embedding_backward")
    if fn is None:
        gw = np.zeros((vocab, dim), dtype=np.float32)
        np.add.at(gw, ids, dy)
        return gw
    flat_dy = _f32(dy.reshape(-1, dim))
    flat_ids = _i64(ids.reshape(-1))
    return fn(flat_dy, flat_ids, int(vocab), int(dim))


def xent_forward(logits: np.ndarray, targets: np.ndarray, ignore_index: int
                 ) -> Tuple[float, np.ndarray, int]:
    fn = _native_for("xent_forward")
    if fn is None:
        flat = logits.reshape(-1, logits.shape[-1])
        t = targets.reshape(-1).astype(np.int64)
        shifted = flat - flat.max(axis=-1, keepdims=True)
        log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = (shifted - log_z).astype(np.float32)
        valid = t != ignore_index
        n = int(valid.sum())
        denom = max(n, 1)
        safe = np.where(valid, t, 0)
        nll = -log_probs[np.arange(flat.shape[0]), safe]
        nll = np.where(valid, nll, 0.0)
        loss = float(nll.sum() / denom)
        return loss, log_probs.reshape(logits.shape), n
    flat = _f32(logits.reshape(-1, logits.shape[-1]))
    t = _i64(targets.reshape(-1))
    loss, lp, valid = fn(flat, t, int(ignore_index))
    return float(loss), lp.reshape(logits.shape), int(valid)


def xent_backward(log_probs: np.ndarray, targets: np.ndarray, grad_loss: float,
                  valid: int, ignore_index: int) -> np.ndarray:
    fn = _native_for("xent_backward")
    if fn is None:
        flat_lp = log_probs.reshape(-1, log_probs.shape[-1])
        t = targets.reshape(-1).astype(np.int64)
        valid_mask = t != ignore_index
        n = max(valid, 1)
        safe = np.where(valid_mask, t, 0)
        sm = np.exp(flat_lp).copy()
        sm[np.arange(flat_lp.shape[0]), safe] -= 1.0
        sm = sm * valid_mask[:, None]
        return ((grad_loss / n) * sm).reshape(log_probs.shape).astype(np.float32)
    flat_lp = _f32(log_probs.reshape(-1, log_probs.shape[-1]))
    t = _i64(targets.reshape(-1))
    out = fn(flat_lp, t, float(grad_loss), int(valid), int(ignore_index))
    return out.reshape(log_probs.shape)
