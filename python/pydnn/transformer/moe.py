"""
Mixture of Experts (MoE).

Implements a noisy top-k gated MoE layer with optional load-balancing
auxiliary loss (Switch Transformer / GShard style). Each expert is a
SwiGLU (or vanilla FFN) block; the gate is a single Linear that
selects ``top_k`` experts per token.

The ``aux_loss`` can be added to the language-modelling loss so that
the gate spreads tokens across experts evenly. Pull it after a forward
pass via ``layer.last_aux_loss``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import autograd as A
from .autograd import Parameter, Tensor
from .layers import FeedForward, Linear, SwiGLU
from .module import Module, ModuleList


class Expert(Module):
    """Default expert: SwiGLU FFN. Override by passing ``expert_factory``."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim=hidden_dim, dropout=dropout, rng=rng)

    def forward(self, x):
        return self.ffn(x)


class MixtureOfExperts(Module):
    """Token-level top-k routed MoE layer.

    Parameters
    ----------
    dim:
        Model hidden size.
    num_experts:
        Total experts in the layer.
    top_k:
        Number of experts each token is routed to (1 or 2 are typical).
    capacity_factor:
        Soft cap on tokens per expert (informational; we don't drop tokens
        in this implementation, but the value is used to compute the
        load-balance loss reference).
    hidden_dim:
        Per-expert FFN hidden dim.
    aux_loss_weight:
        Coefficient stored alongside the loss so callers can add
        ``aux_loss_weight * layer.last_aux_loss`` to their training loss.
    expert_factory:
        Callable ``(dim, rng) -> Module`` to override the default expert.
    """

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 capacity_factor: float = 1.25,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 aux_loss_weight: float = 0.01,
                 expert_factory=None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k {top_k} > num_experts {num_experts}")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = float(capacity_factor)
        self.aux_loss_weight = float(aux_loss_weight)
        self.gate = Linear(dim, num_experts, bias=False, rng=rng)

        if expert_factory is None:
            experts = [
                Expert(dim, hidden_dim=hidden_dim, dropout=dropout, rng=rng)
                for _ in range(num_experts)
            ]
        else:
            experts = [expert_factory(dim, rng) for _ in range(num_experts)]
        self.experts = ModuleList(experts)
        # Bookkeeping populated by forward().
        self.last_aux_loss: Optional[Tensor] = None
        self.last_expert_load: Optional[np.ndarray] = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)              # (N, D)

        # Gating logits and softmax over experts.
        logits = self.gate(flat)                # (N, E)
        probs = A.softmax(logits, axis=-1)      # (N, E)

        # Pick top-k experts per token (np-side; routing indices are
        # discrete so no grad flows through them).
        probs_np = probs.data
        topk_idx = np.argpartition(-probs_np, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        # Sort within the top-k slice so weights are descending.
        topk_vals = np.take_along_axis(probs_np, topk_idx, axis=-1)
        order = np.argsort(-topk_vals, axis=-1)
        topk_idx = np.take_along_axis(topk_idx, order, axis=-1)

        # Renormalise top-k routing weights so each token's expert weights
        # sum to 1 — this lets the gate gradients still flow via ``probs``.
        gather = self._gather_probs(probs, topk_idx)             # (N, k)
        weights = gather / (gather.sum(axis=-1, keepdims=True) + Tensor(np.float32(1e-9)))

        # Run each expert on the subset of tokens routed to it, then
        # scatter the outputs back into a single (N, D) tensor weighted by
        # the gate values.
        N = B * T
        out_pieces = []  # list of (Tensor of shape (N, D), )
        # Build per-expert masks once (numpy).
        # token_to_slot[n, j] == expert chosen by token n at rank j.
        for e_idx in range(self.num_experts):
            # which (token, rank) pairs picked expert e_idx?
            mask = (topk_idx == e_idx)
            if not mask.any():
                continue
            token_ids, ranks = np.nonzero(mask)
            tokens_for_expert = flat.reshape(N, D)[token_ids] if False else None
            # We can't easily slice a Tensor with arbitrary indices through
            # the autograd engine, so do it with embedding-style gather via
            # a scatter trick: build a (M, D) view by re-running embedding
            # on flat treated as a vocab.
            tokens_view = A.embedding(token_ids, flat)            # (M, D)
            expert_out = self.experts[e_idx](tokens_view)         # (M, D)
            # Multiply by routing weight w[token_ids, ranks].
            w_tensor = self._gather_weights(weights, token_ids, ranks)  # (M, 1)
            contribution = expert_out * w_tensor
            # Scatter-add back into a (N, D) buffer.
            out_pieces.append(self._scatter_add(contribution, token_ids, N, D))

        if not out_pieces:
            # No expert was hit — degenerate case. Return zeros + grad.
            zeros = Tensor(np.zeros((N, D), dtype=np.float32),
                           _children=(flat,), _op="moe_empty")
            zeros._backward = lambda: None
            combined = zeros
        else:
            combined = out_pieces[0]
            for piece in out_pieces[1:]:
                combined = combined + piece

        # Load-balancing aux loss (Switch Transformer eq. 4):
        #     L_aux = E * sum_e (f_e * P_e)
        # where f_e = fraction of tokens routed to e (one-hot, top-1)
        # and P_e = mean gate prob for e.
        f = np.zeros((self.num_experts,), dtype=np.float32)
        # Use just the top-1 choice for f.
        top1 = topk_idx[:, 0]
        for e in range(self.num_experts):
            f[e] = float((top1 == e).sum()) / max(N, 1)
        self.last_expert_load = f
        # P_e is differentiable (mean of softmax probs over tokens).
        P = probs.mean(axis=0)   # (E,)
        f_tensor = Tensor(f)
        aux = (f_tensor * P).sum() * Tensor(np.float32(self.num_experts))
        self.last_aux_loss = aux

        return combined.reshape(B, T, D)

    # -------------------- helpers --------------------

    def _gather_probs(self, probs: Tensor, idx: np.ndarray) -> Tensor:
        """Gather along the expert axis. ``idx`` shape (N, k)."""
        N, k = idx.shape
        rows = np.arange(N)[:, None]
        gathered = probs.data[rows, idx]
        out = Tensor(gathered, _children=(probs,), _op="gather_probs")

        def _bw():
            g = out.grad
            gp = np.zeros_like(probs.data)
            np.add.at(gp, (rows, idx), g)
            probs.grad = gp if probs.grad is None else probs.grad + gp

        out._backward = _bw
        return out

    def _gather_weights(self, weights: Tensor,
                        token_ids: np.ndarray, ranks: np.ndarray) -> Tensor:
        """Gather ``weights[token_ids, ranks]`` -> shape (M, 1)."""
        gathered = weights.data[token_ids, ranks][:, None]
        out = Tensor(gathered, _children=(weights,), _op="gather_w")

        def _bw():
            g = out.grad[:, 0]
            gw = np.zeros_like(weights.data)
            np.add.at(gw, (token_ids, ranks), g)
            weights.grad = gw if weights.grad is None else weights.grad + gw

        out._backward = _bw
        return out

    def _scatter_add(self, contrib: Tensor, token_ids: np.ndarray,
                     N: int, D: int) -> Tensor:
        """Scatter (M, D) ``contrib`` rows into a fresh (N, D) buffer."""
        buf = np.zeros((N, D), dtype=np.float32)
        np.add.at(buf, token_ids, contrib.data)
        out = Tensor(buf, _children=(contrib,), _op="scatter_add")

        def _bw():
            g = out.grad[token_ids]
            contrib.grad = g if contrib.grad is None else contrib.grad + g

        out._backward = _bw
        return out
