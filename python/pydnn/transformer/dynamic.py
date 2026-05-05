"""
``DynamicTransformer`` — a transformer that grows and prunes itself
during training, mirroring the spirit of ``pydnn.DynamicNetwork``.

Unlike the dense Dynamic NN core (which mutates per-node masks), here
we adapt structural hyper-parameters that actually shape the
transformer:

- **Layer pruning**: the contribution of a residual block (attention
  *or* FFN) is scaled by a per-block ``alpha`` weight. When a block's
  utilisation (mean abs activation magnitude / cost trend) falls below
  ``prune_threshold`` for ``patience`` consecutive checkpoints, its
  alpha is driven toward 0 — a soft removal that doesn't erase weights.
- **Layer growth**: when validation loss plateaus and the model is
  under-parameterised, an additional decoder block can be appended
  (initialised so it acts close to the identity, mirroring stochastic-
  depth zero-init).
- **Expert health monitoring** (when ``ffn_type="moe"``): under-used
  experts get re-initialised, hot experts trigger a "split" suggestion
  in the health report.
- ``compact()`` collapses zeroed-out blocks at end of training, just
  like ``Network::compact()`` in the C++ core.

This module deliberately reuses the components in ``model.py`` rather
than duplicating them, so the dynamic version is a drop-in replacement
for ``DecoderOnlyModel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import autograd as A
from .autograd import Parameter, Tensor
from .layers import Dropout, FeedForward, LayerNorm, Linear, RMSNorm, SwiGLU
from .model import (
    DecoderOnlyModel,
    TransformerConfig,
    TransformerDecoderLayer,
    _gather_aux_losses,
    _make_attention,
    _make_ffn,
    _make_norm,
    _make_pos_encoding,
)
from .module import Module, ModuleList
from .moe import MixtureOfExperts


# ============================================================
# Health report
# ============================================================


@dataclass
class TransformerHealth:
    """Snapshot of the dynamic transformer's structural state."""
    active_layers: int
    total_layers: int
    layer_alphas: List[float]
    layer_utilisation: List[float]
    pruned_layers: List[int] = field(default_factory=list)
    expert_loads: List[List[float]] = field(default_factory=list)
    diagnosis: str = "healthy"


# ============================================================
# Adaptive decoder block
# ============================================================


class AdaptiveDecoderBlock(TransformerDecoderLayer):
    """Decoder block whose attention/FFN contributions are scaled by alphas.

    ``alpha_attn`` and ``alpha_ffn`` are learnable scalars (initialised to 1)
    that act as soft on/off switches. Setting them to 0 is equivalent to
    soft-removing that sub-block. ``self.utilisation`` tracks the mean
    absolute activation contribution and is updated lazily on each forward.
    """

    def __init__(self, cfg: TransformerConfig,
                 has_cross_attention: bool = False,
                 rotary=None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(cfg, has_cross_attention=has_cross_attention,
                         rotary=rotary, rng=rng)
        self.alpha_attn = Parameter(np.array([1.0], dtype=np.float32))
        self.alpha_ffn = Parameter(np.array([1.0], dtype=np.float32))
        self._utilisation_attn: float = 1.0
        self._utilisation_ffn: float = 1.0
        self._ema = 0.95
        self.active: bool = True

    def freeze_alphas(self, value: float = 0.0) -> None:
        """Hard-set alphas (e.g. 0 to soft-remove). Stops gradient flow."""
        self.alpha_attn.data[...] = value
        self.alpha_ffn.data[...] = value
        self.alpha_attn.requires_grad = False
        self.alpha_ffn.requires_grad = False
        self.active = bool(value > 0)

    def forward(self, x, memory=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                self_kv_cache=None):
        if not self.active:
            return x

        sa = self.self_attn(self.norm1(x), key_padding_mask=tgt_key_padding_mask,
                            kv_cache=self_kv_cache)
        sa_scaled = sa * self.alpha_attn
        x = x + self.dropout1(sa_scaled)

        if self.has_cross_attention and memory is not None:
            ca = self.cross_attn(self.norm2(x), context=memory,
                                 key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout2(ca)

        ff = self.ffn(self.norm3(x))
        ff_scaled = ff * self.alpha_ffn
        x = x + self.dropout3(ff_scaled)

        # Track running utilisation (for health monitoring).
        self._utilisation_attn = (self._ema * self._utilisation_attn
                                  + (1 - self._ema) * float(np.abs(sa.data).mean()))
        self._utilisation_ffn = (self._ema * self._utilisation_ffn
                                 + (1 - self._ema) * float(np.abs(ff.data).mean()))
        return x

    def utilisation(self) -> float:
        return 0.5 * (self._utilisation_attn + self._utilisation_ffn)


# ============================================================
# DynamicTransformer
# ============================================================


class DynamicTransformer(Module):
    """A causal LM that adapts depth + expert health during training.

    Same surface as ``DecoderOnlyModel`` (``forward``, ``generate``,
    ``aux_loss``) plus:

    - ``health_report()`` — current structural state.
    - ``adapt(loss_history, ...)`` — call periodically (e.g. every N
      steps) to soft-prune dead layers and grow if loss has plateaued.
    - ``compact()`` — drop soft-removed layers from the stack.
    """

    def __init__(self, cfg: TransformerConfig,
                 prune_threshold: float = 1e-3,
                 grow_patience: int = 5,
                 prune_patience: int = 3,
                 max_layers: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.prune_threshold = float(prune_threshold)
        self.grow_patience = int(grow_patience)
        self.prune_patience = int(prune_patience)
        self.max_layers = max_layers if max_layers is not None else cfg.num_decoder_layers * 4
        self._rng = np.random.default_rng(cfg.seed)
        self._plateau_count = 0
        self._low_util_streak: Dict[int, int] = {}

        # Embeddings
        from .embeddings import TokenEmbedding
        self.token_emb = TokenEmbedding(cfg.vocab_size, cfg.dim,
                                        padding_idx=cfg.pad_token_id, rng=self._rng)
        self.pos, self.rotary = _make_pos_encoding(cfg, self._rng)
        self.dropout = Dropout(cfg.dropout)

        # Adaptive blocks (no cross-attention, decoder-only).
        blocks = [
            AdaptiveDecoderBlock(cfg, has_cross_attention=False,
                                 rotary=self.rotary, rng=self._rng)
            for _ in range(cfg.num_decoder_layers)
        ]
        self.blocks = ModuleList(blocks)
        self.final_norm = _make_norm(cfg)
        self.tie_embeddings = cfg.tie_embeddings
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.dim, cfg.vocab_size, bias=False, rng=self._rng)

    # -------------------- forward --------------------

    def forward(self, input_ids: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                kv_caches: Optional[List] = None) -> Tensor:
        kpm = None if attention_mask is None else (attention_mask == 0)
        x = self.token_emb(input_ids)
        if self.pos is not None:
            x = self.pos(x)
        x = self.dropout(x)
        for i, blk in enumerate(self.blocks):
            kvc = kv_caches[i] if kv_caches is not None else None
            x = blk(x, memory=None, tgt_key_padding_mask=kpm, self_kv_cache=kvc)
        x = self.final_norm(x)
        if self.tie_embeddings:
            return x @ self.token_emb.weight.transpose(1, 0)
        return self.lm_head(x)

    def aux_loss(self) -> Optional[Tensor]:
        return _gather_aux_losses(self)

    # Reuse generation from DecoderOnlyModel.
    generate = DecoderOnlyModel.generate
    make_kv_caches = DecoderOnlyModel.make_kv_caches

    # -------------------- adaptation --------------------

    def health_report(self) -> TransformerHealth:
        alphas = []
        utils = []
        pruned = []
        for i, blk in enumerate(self.blocks):
            a = 0.5 * (float(blk.alpha_attn.data.item()) + float(blk.alpha_ffn.data.item()))
            alphas.append(a)
            utils.append(blk.utilisation())
            if not blk.active:
                pruned.append(i)
        expert_loads = []
        for sub in self.modules():
            if isinstance(sub, MixtureOfExperts) and sub.last_expert_load is not None:
                expert_loads.append(list(sub.last_expert_load.tolist()))
        active = sum(1 for b in self.blocks if b.active)
        diag = "healthy"
        if active < max(1, len(self.blocks) // 2):
            diag = "over-pruned"
        elif active == len(self.blocks) and self._plateau_count >= self.grow_patience:
            diag = "under-capacity"
        return TransformerHealth(
            active_layers=active,
            total_layers=len(self.blocks),
            layer_alphas=alphas,
            layer_utilisation=utils,
            pruned_layers=pruned,
            expert_loads=expert_loads,
            diagnosis=diag,
        )

    def adapt(self, loss_history: List[float], plateau_window: int = 10,
              plateau_eps: float = 1e-3, allow_grow: bool = True,
              allow_prune: bool = True) -> Dict[str, Any]:
        """Inspect health + loss trend and apply a structural mutation.

        Returns a dict describing what changed (handy to log).
        """
        actions: Dict[str, Any] = {"pruned": [], "grew": False, "reinit_experts": []}

        # 1) Soft-prune dead blocks.
        if allow_prune:
            for i, blk in enumerate(self.blocks):
                if not blk.active:
                    continue
                if blk.utilisation() < self.prune_threshold:
                    self._low_util_streak[i] = self._low_util_streak.get(i, 0) + 1
                    if self._low_util_streak[i] >= self.prune_patience:
                        blk.freeze_alphas(0.0)
                        actions["pruned"].append(i)
                        self._low_util_streak[i] = 0
                else:
                    self._low_util_streak[i] = 0

        # 2) Plateau detection -> consider growth.
        if len(loss_history) >= plateau_window:
            recent = loss_history[-plateau_window:]
            improvement = recent[0] - recent[-1]
            if improvement < plateau_eps:
                self._plateau_count += 1
            else:
                self._plateau_count = 0

        if (allow_grow and self._plateau_count >= self.grow_patience
                and len(self.blocks) < self.max_layers):
            self._add_block()
            actions["grew"] = True
            self._plateau_count = 0

        # 3) Re-init dead experts in any MoE layer.
        for sub in self.modules():
            if isinstance(sub, MixtureOfExperts) and sub.last_expert_load is not None:
                load = sub.last_expert_load
                threshold = 0.1 / sub.num_experts  # ~10x below uniform
                for e_idx, frac in enumerate(load):
                    if frac < threshold:
                        self._reinit_expert(sub, e_idx)
                        actions["reinit_experts"].append((id(sub), e_idx))
        return actions

    def _add_block(self) -> None:
        """Append a near-identity block (alphas initialised tiny)."""
        blk = AdaptiveDecoderBlock(self.cfg, has_cross_attention=False,
                                   rotary=self.rotary, rng=self._rng)
        # Start near-zero so the new block can't disrupt the current model.
        blk.alpha_attn.data[...] = 1e-3
        blk.alpha_ffn.data[...] = 1e-3
        self.blocks.append(blk)

    @staticmethod
    def _reinit_expert(moe: MixtureOfExperts, idx: int) -> None:
        rng = np.random.default_rng()
        for p in moe.experts[idx].parameters():
            shape = p.data.shape
            bound = 1.0 / np.sqrt(max(shape[-1], 1))
            p.data[...] = rng.uniform(-bound, bound, shape).astype(np.float32)

    def compact(self) -> int:
        """Permanently drop blocks with active=False. Returns # removed."""
        kept = [b for b in self.blocks if b.active]
        removed = len(self.blocks) - len(kept)
        if removed == 0:
            return 0
        self.blocks = ModuleList(kept)
        return removed

    def num_active_parameters(self) -> int:
        n = 0
        for p in self.token_emb.parameters():
            n += int(p.data.size)
        for blk in self.blocks:
            if not blk.active:
                continue
            for p in blk.parameters():
                n += int(p.data.size)
        for p in self.final_norm.parameters():
            n += int(p.data.size)
        return n
