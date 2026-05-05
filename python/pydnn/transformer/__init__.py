"""
``pydnn.transformer`` — modern Transformer / MoE building blocks for LLMs,
designed to compose with the Dynamic Neural Network philosophy.

Quick start:

    >>> from pydnn import transformer
    >>> cfg = transformer.TransformerConfig(
    ...     vocab_size=1024, dim=128, num_heads=4,
    ...     num_decoder_layers=4, ffn_type="swiglu",
    ...     pos_encoding="rope", norm_type="rmsnorm",
    ... )
    >>> model = transformer.DecoderOnlyModel(cfg)
    >>> import numpy as np
    >>> ids = np.random.randint(0, cfg.vocab_size, size=(2, 16))
    >>> logits = model(ids)             # (2, 16, vocab_size)
    >>> out = model.generate(ids[:, :4], max_new_tokens=10)

Mixture of Experts:

    >>> moe_cfg = transformer.TransformerConfig(
    ...     vocab_size=1024, dim=128, num_heads=4,
    ...     num_decoder_layers=4, ffn_type="moe",
    ...     num_experts=8, moe_top_k=2,
    ... )
    >>> moe_model = transformer.DecoderOnlyModel(moe_cfg)

Dynamic (growing/pruning) transformer:

    >>> dyn = transformer.DynamicTransformer(cfg)
    >>> trainer = transformer.CausalLMTrainer(dyn)
    >>> # ... training loop, then ...
    >>> dyn.adapt(trainer.fit(corpus_ids, steps=200).losses)
    >>> dyn.compact()      # freeze the structural changes for inference
"""

from . import autograd
from . import attention as _attention
from . import embeddings as _embeddings
from . import layers as _layers
from . import model as _model
from . import moe as _moe
from . import tokenizer as _tokenizer
from . import training as _training
from . import dynamic as _dynamic

# Autograd primitives
from .autograd import (
    Tensor,
    Parameter,
    matmul,
    softmax,
    log_softmax,
    gelu,
    silu,
    relu,
    dropout,
    embedding,
    cross_entropy,
    masked_fill,
    concat,
    stack,
)

# Module base
from .module import Module, ModuleList

# Layers
from .layers import (
    Linear,
    LayerNorm,
    RMSNorm,
    Dropout,
    FeedForward,
    SwiGLU,
)

# Embeddings & positional encodings
from .embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryEmbedding,
)

# Attention
from .attention import (
    MultiHeadAttention,
    GroupedQueryAttention,
    KVCache,
    scaled_dot_product_attention,
)

# Mixture of Experts
from .moe import (
    Expert,
    MixtureOfExperts,
)

# Models
from .model import (
    TransformerConfig,
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    EncoderOnlyModel,
    DecoderOnlyModel,
    Seq2SeqModel,
    Transformer,
)

# Tokenizers
from .tokenizer import (
    WhitespaceTokenizer,
    CharTokenizer,
    BPETokenizer,
    TokenizerOutput,
    SPECIAL_TOKENS,
)

# Training
from .training import (
    AdamW,
    SGD,
    cosine_with_warmup,
    linear_with_warmup,
    clip_grad_norm,
    CausalLMTrainer,
    TrainResult,
)

# Dynamic transformer (growth + prune)
from .dynamic import (
    DynamicTransformer,
    AdaptiveDecoderBlock,
    TransformerHealth,
)


__all__ = [
    # Autograd
    "Tensor", "Parameter", "matmul", "softmax", "log_softmax",
    "gelu", "silu", "relu", "dropout", "embedding", "cross_entropy",
    "masked_fill", "concat", "stack",
    # Module
    "Module", "ModuleList",
    # Layers
    "Linear", "LayerNorm", "RMSNorm", "Dropout", "FeedForward", "SwiGLU",
    # Embeddings
    "TokenEmbedding", "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding", "RotaryEmbedding",
    # Attention
    "MultiHeadAttention", "GroupedQueryAttention", "KVCache",
    "scaled_dot_product_attention",
    # MoE
    "Expert", "MixtureOfExperts",
    # Models
    "TransformerConfig",
    "TransformerEncoderLayer", "TransformerEncoder",
    "TransformerDecoderLayer", "TransformerDecoder",
    "EncoderOnlyModel", "DecoderOnlyModel", "Seq2SeqModel", "Transformer",
    # Tokenizers
    "WhitespaceTokenizer", "CharTokenizer", "BPETokenizer",
    "TokenizerOutput", "SPECIAL_TOKENS",
    # Training
    "AdamW", "SGD", "cosine_with_warmup", "linear_with_warmup",
    "clip_grad_norm", "CausalLMTrainer", "TrainResult",
    # Dynamic
    "DynamicTransformer", "AdaptiveDecoderBlock", "TransformerHealth",
]
