"""
Encoder, decoder, and full Transformer / GPT / BERT-style models.

Layouts:
- ``TransformerEncoderLayer`` / ``TransformerDecoderLayer``: pre-norm
  blocks (modern default; matches LLaMA/GPT-NeoX/etc).
- ``TransformerEncoder`` / ``TransformerDecoder``: stacks of layers.
- ``Transformer``: full encoder-decoder (Vaswani 2017, modernised).
- ``EncoderOnlyModel``: BERT-style classifier / encoder.
- ``DecoderOnlyModel``: GPT-style causal language model.
- ``Seq2SeqModel``: the encoder-decoder LM with a tied output head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import autograd as A
from .attention import GroupedQueryAttention, KVCache, MultiHeadAttention
from .autograd import Parameter, Tensor
from .embeddings import (
    LearnedPositionalEncoding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from .layers import Dropout, FeedForward, LayerNorm, Linear, RMSNorm, SwiGLU
from .module import Module, ModuleList
from .moe import MixtureOfExperts


# ============================================================
# Config
# ============================================================


@dataclass
class TransformerConfig:
    """Configuration for the Transformer family.

    All fields have sensible defaults; pass only what you need to override.
    """
    vocab_size: int = 32_000
    dim: int = 256
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # None -> standard MHA
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    ffn_hidden_dim: Optional[int] = None
    activation: str = "gelu"            # {"gelu","relu","silu"}
    ffn_type: str = "ffn"               # {"ffn","swiglu","moe"}
    norm_type: str = "layernorm"        # {"layernorm","rmsnorm"}
    pre_norm: bool = True
    max_seq_len: int = 2048
    pos_encoding: str = "sinusoidal"    # {"sinusoidal","learned","rope","none"}
    dropout: float = 0.1
    attention_dropout: float = 0.0
    bias: bool = True
    tie_embeddings: bool = True
    pad_token_id: int = 0
    # MoE-specific
    num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    # Init
    seed: Optional[int] = None


def _make_norm(cfg: TransformerConfig) -> Module:
    if cfg.norm_type == "layernorm":
        return LayerNorm(cfg.dim)
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.dim)
    raise ValueError(f"unknown norm_type: {cfg.norm_type}")


def _make_ffn(cfg: TransformerConfig, rng: np.random.Generator) -> Module:
    if cfg.ffn_type == "ffn":
        return FeedForward(cfg.dim, hidden_dim=cfg.ffn_hidden_dim,
                           activation=cfg.activation, dropout=cfg.dropout,
                           bias=cfg.bias, rng=rng)
    if cfg.ffn_type == "swiglu":
        return SwiGLU(cfg.dim, hidden_dim=cfg.ffn_hidden_dim,
                      dropout=cfg.dropout, bias=False, rng=rng)
    if cfg.ffn_type == "moe":
        return MixtureOfExperts(
            cfg.dim, num_experts=cfg.num_experts, top_k=cfg.moe_top_k,
            hidden_dim=cfg.ffn_hidden_dim, dropout=cfg.dropout,
            aux_loss_weight=cfg.moe_aux_loss_weight, rng=rng,
        )
    raise ValueError(f"unknown ffn_type: {cfg.ffn_type}")


def _make_attention(cfg: TransformerConfig, causal: bool,
                    rotary: Optional[RotaryEmbedding],
                    rng: np.random.Generator) -> Module:
    if cfg.num_kv_heads is None or cfg.num_kv_heads == cfg.num_heads:
        return MultiHeadAttention(
            cfg.dim, cfg.num_heads,
            dropout=cfg.attention_dropout, bias=cfg.bias,
            causal=causal, rotary=rotary, rng=rng,
        )
    return GroupedQueryAttention(
        cfg.dim, cfg.num_heads, cfg.num_kv_heads,
        dropout=cfg.attention_dropout, bias=cfg.bias,
        causal=causal, rotary=rotary, rng=rng,
    )


# ============================================================
# Encoder
# ============================================================


class TransformerEncoderLayer(Module):
    """Pre-norm encoder block: x = x + Attn(LN(x));  x = x + FFN(LN(x))."""

    def __init__(self, cfg: TransformerConfig,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.norm1 = _make_norm(cfg)
        self.self_attn = _make_attention(cfg, causal=False, rotary=rotary, rng=rng)
        self.dropout1 = Dropout(cfg.dropout)
        self.norm2 = _make_norm(cfg)
        self.ffn = _make_ffn(cfg, rng=rng)
        self.dropout2 = Dropout(cfg.dropout)

    def forward(self, x: Tensor,
                key_padding_mask: Optional[np.ndarray] = None) -> Tensor:
        if self.cfg.pre_norm:
            attn_out = self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask)
            x = x + self.dropout1(attn_out)
            ffn_out = self.ffn(self.norm2(x))
            x = x + self.dropout2(ffn_out)
        else:
            attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
            x = self.norm1(x + self.dropout1(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TransformerEncoder(Module):
    def __init__(self, cfg: TransformerConfig,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.layers = ModuleList([
            TransformerEncoderLayer(cfg, rotary=rotary, rng=rng)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.norm = _make_norm(cfg)

    def forward(self, x: Tensor,
                key_padding_mask: Optional[np.ndarray] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


# ============================================================
# Decoder
# ============================================================


class TransformerDecoderLayer(Module):
    """Decoder block: causal self-attn + (optional) cross-attn + FFN."""

    def __init__(self, cfg: TransformerConfig,
                 has_cross_attention: bool = True,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.has_cross_attention = has_cross_attention
        self.norm1 = _make_norm(cfg)
        self.self_attn = _make_attention(cfg, causal=True, rotary=rotary, rng=rng)
        self.dropout1 = Dropout(cfg.dropout)
        if has_cross_attention:
            self.norm2 = _make_norm(cfg)
            # Cross-attention is never causal and never uses RoPE.
            self.cross_attn = _make_attention(cfg, causal=False, rotary=None, rng=rng)
            self.dropout2 = Dropout(cfg.dropout)
        self.norm3 = _make_norm(cfg)
        self.ffn = _make_ffn(cfg, rng=rng)
        self.dropout3 = Dropout(cfg.dropout)

    def forward(self,
                x: Tensor,
                memory: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[np.ndarray] = None,
                memory_key_padding_mask: Optional[np.ndarray] = None,
                self_kv_cache: Optional[KVCache] = None) -> Tensor:
        # Self-attention (causal)
        sa = self.self_attn(self.norm1(x), key_padding_mask=tgt_key_padding_mask,
                            kv_cache=self_kv_cache)
        x = x + self.dropout1(sa)

        # Cross-attention to encoder memory
        if self.has_cross_attention and memory is not None:
            ca = self.cross_attn(self.norm2(x), context=memory,
                                 key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout2(ca)

        # FFN
        ff = self.ffn(self.norm3(x))
        return x + self.dropout3(ff)


class TransformerDecoder(Module):
    def __init__(self, cfg: TransformerConfig,
                 has_cross_attention: bool = True,
                 rotary: Optional[RotaryEmbedding] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.layers = ModuleList([
            TransformerDecoderLayer(cfg, has_cross_attention=has_cross_attention,
                                    rotary=rotary, rng=rng)
            for _ in range(cfg.num_decoder_layers)
        ])
        self.norm = _make_norm(cfg)

    def forward(self,
                x: Tensor,
                memory: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[np.ndarray] = None,
                memory_key_padding_mask: Optional[np.ndarray] = None,
                kv_caches: Optional[List[KVCache]] = None) -> Tensor:
        for i, layer in enumerate(self.layers):
            kvc = kv_caches[i] if kv_caches is not None else None
            x = layer(x, memory=memory,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      self_kv_cache=kvc)
        return self.norm(x)


# ============================================================
# Embeddings + LM head wiring
# ============================================================


def _make_pos_encoding(cfg: TransformerConfig, rng: np.random.Generator):
    """Returns (pos_module_or_none, rotary_or_none)."""
    if cfg.pos_encoding == "none":
        return None, None
    if cfg.pos_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(cfg.dim, max_len=cfg.max_seq_len), None
    if cfg.pos_encoding == "learned":
        return LearnedPositionalEncoding(cfg.max_seq_len, cfg.dim, rng=rng), None
    if cfg.pos_encoding == "rope":
        head_dim = cfg.dim // cfg.num_heads
        return None, RotaryEmbedding(head_dim, max_len=cfg.max_seq_len)
    raise ValueError(f"unknown pos_encoding: {cfg.pos_encoding}")


def _gather_aux_losses(module: Module) -> Optional[Tensor]:
    """Sum any MoE auxiliary losses recorded under ``module``."""
    pieces: List[Tensor] = []
    for sub in module.modules():
        if isinstance(sub, MixtureOfExperts) and sub.last_aux_loss is not None:
            pieces.append(sub.last_aux_loss * Tensor(np.float32(sub.aux_loss_weight)))
    if not pieces:
        return None
    total = pieces[0]
    for p in pieces[1:]:
        total = total + p
    return total


# ============================================================
# Encoder-only (BERT-like)
# ============================================================


class EncoderOnlyModel(Module):
    """BERT-style encoder. Returns hidden states; add a head externally."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.token_emb = TokenEmbedding(cfg.vocab_size, cfg.dim,
                                        padding_idx=cfg.pad_token_id, rng=rng)
        self.pos, self.rotary = _make_pos_encoding(cfg, rng)
        self.dropout = Dropout(cfg.dropout)
        self.encoder = TransformerEncoder(cfg, rotary=self.rotary, rng=rng)

    def forward(self, input_ids: np.ndarray,
                attention_mask: Optional[np.ndarray] = None) -> Tensor:
        kpm = None if attention_mask is None else (attention_mask == 0)
        x = self.token_emb(input_ids)
        if self.pos is not None:
            x = self.pos(x)
        x = self.dropout(x)
        return self.encoder(x, key_padding_mask=kpm)

    def aux_loss(self) -> Optional[Tensor]:
        return _gather_aux_losses(self)


# ============================================================
# Decoder-only (GPT-like) causal language model
# ============================================================


class DecoderOnlyModel(Module):
    """GPT-style causal language model with optional KV-cache decoding."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.token_emb = TokenEmbedding(cfg.vocab_size, cfg.dim,
                                        padding_idx=cfg.pad_token_id, rng=rng)
        self.pos, self.rotary = _make_pos_encoding(cfg, rng)
        self.dropout = Dropout(cfg.dropout)
        # Reuse the decoder stack but disable cross-attention.
        self.decoder = TransformerDecoder(cfg, has_cross_attention=False,
                                          rotary=self.rotary, rng=rng)
        self.tie_embeddings = cfg.tie_embeddings
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.dim, cfg.vocab_size, bias=False, rng=rng)

    def _logits(self, hidden: Tensor) -> Tensor:
        if self.tie_embeddings:
            # Multiply by E^T  (vocab_size, dim) -> dim x vocab.
            return hidden @ self.token_emb.weight.transpose(1, 0)
        return self.lm_head(hidden)

    def forward(self, input_ids: np.ndarray,
                attention_mask: Optional[np.ndarray] = None,
                kv_caches: Optional[List[KVCache]] = None) -> Tensor:
        kpm = None if attention_mask is None else (attention_mask == 0)
        x = self.token_emb(input_ids)
        if self.pos is not None:
            x = self.pos(x)
        x = self.dropout(x)
        h = self.decoder(x, memory=None,
                         tgt_key_padding_mask=kpm,
                         kv_caches=kv_caches)
        return self._logits(h)

    def aux_loss(self) -> Optional[Tensor]:
        return _gather_aux_losses(self)

    # -------------------- generation --------------------

    def make_kv_caches(self) -> List[KVCache]:
        return [KVCache() for _ in range(self.cfg.num_decoder_layers)]

    def generate(self, input_ids: np.ndarray,
                 max_new_tokens: int = 20,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 use_cache: bool = True,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Greedy / sampled autoregressive generation.

        ``input_ids`` shape ``(B, T0)``. Returns shape ``(B, T0 + n)``
        where ``n`` is at most ``max_new_tokens`` (early-stops on EOS).
        """
        rng = rng if rng is not None else np.random.default_rng()
        was_training = self.training
        self.eval()
        try:
            ids = np.asarray(input_ids, dtype=np.int64).copy()
            B = ids.shape[0]
            caches = self.make_kv_caches() if use_cache else None
            # Prefill
            if use_cache:
                logits = self.forward(ids, kv_caches=caches)
            else:
                logits = self.forward(ids)
            done = np.zeros((B,), dtype=bool)
            for _ in range(max_new_tokens):
                step = logits.data[:, -1, :] / max(temperature, 1e-6)
                next_ids = self._sample(step, top_k=top_k, top_p=top_p, rng=rng)
                # Lock finished sequences to a pad token.
                if eos_token_id is not None:
                    next_ids = np.where(done, eos_token_id, next_ids)
                    done = done | (next_ids == eos_token_id)
                ids = np.concatenate([ids, next_ids[:, None]], axis=1)
                if eos_token_id is not None and done.all():
                    break
                if use_cache:
                    logits = self.forward(next_ids[:, None], kv_caches=caches)
                else:
                    logits = self.forward(ids)
            return ids
        finally:
            self.train(was_training)

    @staticmethod
    def _sample(logits: np.ndarray, top_k: Optional[int],
                top_p: Optional[float], rng: np.random.Generator) -> np.ndarray:
        if top_k is None and top_p is None:
            # Greedy
            return logits.argmax(axis=-1)
        # Apply top-k filter
        x = logits.copy()
        if top_k is not None and top_k > 0:
            kth = np.partition(x, -top_k, axis=-1)[:, -top_k:-top_k + 1]
            x = np.where(x < kth, -np.inf, x)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_idx = np.argsort(-x, axis=-1)
            sorted_logits = np.take_along_axis(x, sorted_idx, axis=-1)
            # Stable softmax over each row
            shifted = sorted_logits - sorted_logits.max(axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            cum = np.cumsum(probs, axis=-1)
            mask_sorted = cum - probs > top_p
            sorted_logits = np.where(mask_sorted, -np.inf, sorted_logits)
            inverse = np.argsort(sorted_idx, axis=-1)
            x = np.take_along_axis(sorted_logits, inverse, axis=-1)
        # Sample from softmax
        shifted = x - x.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        out = np.empty(probs.shape[0], dtype=np.int64)
        for i in range(probs.shape[0]):
            out[i] = rng.choice(probs.shape[1], p=probs[i])
        return out


# ============================================================
# Encoder-decoder Seq2Seq (Vaswani 2017)
# ============================================================


class Seq2SeqModel(Module):
    """Full encoder-decoder Transformer with shared / tied output head."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.src_emb = TokenEmbedding(cfg.vocab_size, cfg.dim,
                                      padding_idx=cfg.pad_token_id, rng=rng)
        self.tgt_emb = TokenEmbedding(cfg.vocab_size, cfg.dim,
                                      padding_idx=cfg.pad_token_id, rng=rng)
        self.pos_src, self.rotary = _make_pos_encoding(cfg, rng)
        # The decoder side reuses positional encoding shape; share if learned.
        self.pos_tgt = self.pos_src
        self.dropout_src = Dropout(cfg.dropout)
        self.dropout_tgt = Dropout(cfg.dropout)
        self.encoder = TransformerEncoder(cfg, rotary=self.rotary, rng=rng)
        self.decoder = TransformerDecoder(cfg, has_cross_attention=True,
                                          rotary=self.rotary, rng=rng)
        self.tie_embeddings = cfg.tie_embeddings
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.dim, cfg.vocab_size, bias=False, rng=rng)

    def encode(self, src_ids: np.ndarray,
               src_attention_mask: Optional[np.ndarray] = None) -> Tensor:
        kpm = None if src_attention_mask is None else (src_attention_mask == 0)
        x = self.src_emb(src_ids)
        if self.pos_src is not None:
            x = self.pos_src(x)
        x = self.dropout_src(x)
        return self.encoder(x, key_padding_mask=kpm)

    def decode(self, tgt_ids: np.ndarray, memory: Tensor,
               tgt_attention_mask: Optional[np.ndarray] = None,
               memory_key_padding_mask: Optional[np.ndarray] = None,
               kv_caches: Optional[List[KVCache]] = None) -> Tensor:
        kpm = None if tgt_attention_mask is None else (tgt_attention_mask == 0)
        x = self.tgt_emb(tgt_ids)
        if self.pos_tgt is not None:
            x = self.pos_tgt(x)
        x = self.dropout_tgt(x)
        return self.decoder(x, memory=memory,
                            tgt_key_padding_mask=kpm,
                            memory_key_padding_mask=memory_key_padding_mask,
                            kv_caches=kv_caches)

    def _logits(self, hidden: Tensor) -> Tensor:
        if self.tie_embeddings:
            return hidden @ self.tgt_emb.weight.transpose(1, 0)
        return self.lm_head(hidden)

    def forward(self,
                src_ids: np.ndarray,
                tgt_ids: np.ndarray,
                src_attention_mask: Optional[np.ndarray] = None,
                tgt_attention_mask: Optional[np.ndarray] = None) -> Tensor:
        memory = self.encode(src_ids, src_attention_mask=src_attention_mask)
        memory_kpm = None if src_attention_mask is None else (src_attention_mask == 0)
        h = self.decode(tgt_ids, memory,
                        tgt_attention_mask=tgt_attention_mask,
                        memory_key_padding_mask=memory_kpm)
        return self._logits(h)

    def aux_loss(self) -> Optional[Tensor]:
        return _gather_aux_losses(self)


# Convenience alias matching torch.nn.Transformer naming.
Transformer = Seq2SeqModel
