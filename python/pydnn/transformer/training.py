"""
Training utilities: AdamW optimizer, LR schedules, gradient clipping,
and a small ``Trainer`` helper that runs a causal-LM training loop on
NumPy token arrays.

These are intentionally minimal — enough to actually train the
transformers in this subpackage without dragging in another framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import math
import time

import numpy as np

from . import autograd as A
from .autograd import Parameter, Tensor
from .module import Module


# ============================================================
# Optimizer
# ============================================================


class AdamW:
    """AdamW (Adam with decoupled weight decay) — the modern default."""

    def __init__(self, params: Iterable[Parameter],
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        self.params: List[Parameter] = [p for p in params]
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            update = self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            if self.weight_decay > 0:
                update = update + self.lr * self.weight_decay * p.data
            p.data -= update

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)


class SGD:
    """Plain SGD with optional momentum, mostly for comparison."""

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-2,
                 momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        self.params = [p for p in params]
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._buf = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p, buf in zip(self.params, self._buf):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            if self.momentum > 0:
                buf[:] = self.momentum * buf + g
                p.data -= self.lr * buf
            else:
                p.data -= self.lr * g

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)


# ============================================================
# Learning rate schedules
# ============================================================


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int,
                       max_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def linear_with_warmup(step: int, warmup_steps: int, total_steps: int,
                       max_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return max_lr + (min_lr - max_lr) * progress


# ============================================================
# Gradient clipping
# ============================================================


def clip_grad_norm(params: Iterable[Parameter], max_norm: float) -> float:
    """Global L2 grad-norm clipping. Returns the (pre-clip) global norm."""
    total = 0.0
    plist = []
    for p in params:
        if p.grad is None:
            continue
        plist.append(p)
        total += float((p.grad ** 2).sum())
    norm = math.sqrt(total)
    if norm > max_norm and norm > 0:
        scale = max_norm / (norm + 1e-6)
        for p in plist:
            p.grad *= scale
    return norm


# ============================================================
# Causal-LM trainer
# ============================================================


@dataclass
class TrainResult:
    losses: List[float] = field(default_factory=list)
    aux_losses: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    steps: int = 0
    seconds: float = 0.0


class CausalLMTrainer:
    """Train a ``DecoderOnlyModel`` on next-token prediction.

    Pass either:
    - A 2-D array ``token_ids`` of shape ``(N, T)`` and the trainer will
      use ``ids[:, :-1]`` as input and ``ids[:, 1:]`` as targets, OR
    - A long 1-D array ``token_ids`` of shape ``(L,)``; batches are then
      sampled by picking random windows of length ``seq_len + 1``.
    """

    def __init__(self, model: Module,
                 lr: float = 3e-4,
                 weight_decay: float = 0.01,
                 betas=(0.9, 0.95),
                 grad_clip: Optional[float] = 1.0,
                 warmup_steps: int = 100,
                 schedule: str = "cosine",      # "cosine" | "linear" | "constant"
                 pad_token_id: int = 0,
                 ignore_pad_in_loss: bool = True) -> None:
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr,
                               betas=betas, weight_decay=weight_decay)
        self.max_lr = float(lr)
        self.grad_clip = grad_clip
        self.warmup_steps = int(warmup_steps)
        self.schedule = schedule
        self.pad_token_id = int(pad_token_id)
        self.ignore_pad = bool(ignore_pad_in_loss)
        self._step = 0

    def _compute_lr(self, total_steps: int) -> float:
        if self.schedule == "cosine":
            return cosine_with_warmup(self._step, self.warmup_steps, total_steps, self.max_lr)
        if self.schedule == "linear":
            return linear_with_warmup(self._step, self.warmup_steps, total_steps, self.max_lr)
        return self.max_lr

    def _make_batch(self, token_ids: np.ndarray, batch_size: int,
                    seq_len: int, rng: np.random.Generator) -> np.ndarray:
        if token_ids.ndim == 2:
            idx = rng.integers(0, token_ids.shape[0], size=batch_size)
            chunk = token_ids[idx]
            # Truncate / pad to seq_len + 1 so input and target both fit.
            if chunk.shape[1] > seq_len + 1:
                start = rng.integers(0, chunk.shape[1] - seq_len)
                chunk = chunk[:, start:start + seq_len + 1]
            return chunk
        # 1-D long stream
        starts = rng.integers(0, token_ids.shape[0] - seq_len - 1, size=batch_size)
        return np.stack([token_ids[s:s + seq_len + 1] for s in starts], axis=0)

    def fit(self, token_ids: np.ndarray,
            steps: int = 100,
            batch_size: int = 16,
            seq_len: int = 64,
            log_every: int = 10,
            verbose: bool = True,
            rng: Optional[np.random.Generator] = None,
            adapt_every: Optional[int] = None,
            adapt_window: int = 50) -> TrainResult:
        """Train for ``steps`` steps.

        ``adapt_every`` (optional): when set, every ``adapt_every`` steps the
        trainer calls ``self.model.adapt(result.losses[-adapt_window:])`` if
        the model exposes an ``adapt`` method (e.g. ``DynamicTransformer``).
        Silently no-op for models without ``adapt``.
        """
        rng = rng if rng is not None else np.random.default_rng()
        token_ids = np.asarray(token_ids, dtype=np.int64)
        result = TrainResult()
        t0 = time.time()
        self.model.train()
        can_adapt = (adapt_every is not None
                     and adapt_every > 0
                     and hasattr(self.model, "adapt"))
        for _ in range(steps):
            batch = self._make_batch(token_ids, batch_size, seq_len, rng)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            ignore = -100
            if self.ignore_pad:
                targets = np.where(targets == self.pad_token_id, ignore, targets)

            lr_now = self._compute_lr(steps)
            self.optimizer.set_lr(lr_now)
            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = A.cross_entropy(logits, targets, ignore_index=ignore)
            aux = self.model.aux_loss() if hasattr(self.model, "aux_loss") else None
            total = loss if aux is None else (loss + aux)
            total.backward()

            if self.grad_clip is not None:
                clip_grad_norm(self.optimizer.params, self.grad_clip)
            self.optimizer.step()

            self._step += 1
            result.losses.append(float(loss.data))
            if aux is not None:
                result.aux_losses.append(float(aux.data))
            if verbose and (self._step % log_every == 0 or self._step == 1):
                msg = f"step {self._step:5d}  loss={float(loss.data):.4f}  lr={lr_now:.2e}"
                if aux is not None:
                    msg += f"  aux={float(aux.data):.4f}"
                print(msg)

            if can_adapt and self._step % adapt_every == 0:
                info = self.model.adapt(result.losses[-adapt_window:])
                if verbose and any(info.get(k)
                                   for k in ("pruned", "grew", "reinit_experts")):
                    print(f"  step {self._step:5d}  model.adapt -> {info}")
        result.steps = self._step
        result.seconds = time.time() - t0
        if result.losses:
            result.final_loss = result.losses[-1]
        return result
