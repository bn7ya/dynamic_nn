"""
Pure Python Transformer Components

Implements all transformer building blocks from scratch using only NumPy:
- Embedding
- Positional Encoding
- Layer Normalization
- Multi-Head Self-Attention
- Feed-Forward Network
- Transformer Encoder Layer

Each component includes forward pass, backward pass (gradient computation),
and efficiency tracking for dynamic architecture adaptation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# =============================================================================
# EMBEDDING LAYER
# =============================================================================

class Embedding:
    """
    Learnable word embeddings.
    Maps token IDs to dense vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        np.random.seed(seed)
        # Xavier/Glorot initialization
        self.weights = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / (vocab_size + embed_dim))
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Gradients
        self.grad_weights = np.zeros_like(self.weights)

        # Cache for backward pass
        self._cache: Dict = {}

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            token_ids: (batch_size, seq_len) integer token IDs

        Returns:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        self._cache['token_ids'] = token_ids
        return self.weights[token_ids]

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Compute gradient for embedding weights.

        Args:
            grad_output: (batch_size, seq_len, embed_dim) gradient from next layer
        """
        token_ids = self._cache['token_ids']
        batch_size, seq_len, _ = grad_output.shape

        # Accumulate gradients for each token
        self.grad_weights.fill(0)
        np.add.at(self.grad_weights, token_ids.flatten(),
                  grad_output.reshape(-1, self.embed_dim))

    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.grad_weights.fill(0)

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters"""
        return [self.weights]

    def gradients(self) -> List[np.ndarray]:
        """Return list of gradients"""
        return [self.grad_weights]


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Sinusoidal positional encoding (non-learnable).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        self.encoding = self._create_encoding(max_seq_len, embed_dim)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding matrix"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])

        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to embeddings.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len, embed_dim) with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Positional encoding has no learnable parameters"""
        return grad_output


# =============================================================================
# LAYER NORMALIZATION
# =============================================================================

class LayerNorm:
    """
    Layer Normalization: normalizes across features.

    y = gamma * (x - mean) / (std + eps) + beta
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)  # Scale
        self.beta = np.zeros(normalized_shape)   # Shift
        self.normalized_shape = normalized_shape

        # Gradients
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # Cache for backward pass
        self._cache: Dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (..., normalized_shape)

        Returns:
            Normalized x with same shape
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)

        x_norm = (x - mean) / std

        self._cache = {'x': x, 'mean': mean, 'std': std, 'x_norm': x_norm}

        return self.gamma * x_norm + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients for gamma, beta, and input.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        x = self._cache['x']
        x_norm = self._cache['x_norm']
        std = self._cache['std']
        N = self.normalized_shape

        # Gradient for gamma and beta
        self.grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient for input
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (x - self._cache['mean']) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / std, axis=-1, keepdims=True)

        dx = dx_norm / std + dvar * 2 * (x - self._cache['mean']) / N + dmean / N

        return dx

    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters"""
        return [self.gamma, self.beta]

    def gradients(self) -> List[np.ndarray]:
        """Return list of gradients"""
        return [self.grad_gamma, self.grad_beta]


# =============================================================================
# MULTI-HEAD SELF-ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        seed: int = 42
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.dropout = dropout

        # Weight matrices
        self.W_q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim)

        # Biases
        self.b_q = np.zeros(embed_dim)
        self.b_k = np.zeros(embed_dim)
        self.b_v = np.zeros(embed_dim)
        self.b_o = np.zeros(embed_dim)

        # Gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_o = np.zeros_like(self.b_o)

        # Efficiency tracking for dynamic adaptation
        self.head_efficiency = np.ones(num_heads) * 0.5

        # Cache for backward pass
        self._cache: Dict = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over last axis"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Args:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = query @ self.W_q + self.b_q
        K = key @ self.W_k + self.b_k
        V = value @ self.W_v + self.b_v

        # Reshape for multi-head: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores: (batch, heads, seq, seq)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask * (-1e9)

        # Softmax
        attention_weights = self._softmax(scores)

        # Store for backward pass and efficiency calculation
        self._cache = {
            'query': query, 'key': key, 'value': value,
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'batch_size': batch_size, 'seq_len': seq_len,
            'training': training
        }

        # Apply dropout during training
        if training and self.dropout > 0:
            drop_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights_dropped = attention_weights * drop_mask / (1 - self.dropout)
            self._cache['drop_mask'] = drop_mask
        else:
            attention_weights_dropped = attention_weights

        # Apply attention to values
        context = attention_weights_dropped @ V  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        self._cache['context'] = context

        # Final projection
        output = context @ self.W_o + self.b_o

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients for all weights and return gradient for input.

        Args:
            grad_output: (batch_size, seq_len, embed_dim)

        Returns:
            Gradient with respect to query (input)
        """
        batch_size = self._cache['batch_size']
        seq_len = self._cache['seq_len']
        query = self._cache['query']
        key = self._cache['key']
        value = self._cache['value']
        Q = self._cache['Q']
        K = self._cache['K']
        V = self._cache['V']
        attention_weights = self._cache['attention_weights']
        context = self._cache['context']

        # Gradient through output projection
        self.grad_W_o = context.reshape(-1, self.embed_dim).T @ grad_output.reshape(-1, self.embed_dim)
        self.grad_b_o = grad_output.sum(axis=(0, 1))
        grad_context = grad_output @ self.W_o.T

        # Reshape gradient: (batch, seq, embed_dim) -> (batch, heads, seq, head_dim)
        grad_context = grad_context.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Gradient through attention
        grad_V = attention_weights.transpose(0, 1, 3, 2) @ grad_context

        grad_attn = grad_context @ V.transpose(0, 1, 3, 2)

        # Apply dropout mask if used
        if 'drop_mask' in self._cache:
            grad_attn = grad_attn * self._cache['drop_mask'] / (1 - self.dropout)

        # Gradient through softmax
        softmax_grad = attention_weights * (grad_attn - (grad_attn * attention_weights).sum(axis=-1, keepdims=True))

        # Gradient through scaled dot-product
        grad_Q = (softmax_grad @ K) * self.scale
        grad_K = (softmax_grad.transpose(0, 1, 3, 2) @ Q) * self.scale

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # Gradient through linear projections
        self.grad_W_q = query.reshape(-1, self.embed_dim).T @ grad_Q.reshape(-1, self.embed_dim)
        self.grad_b_q = grad_Q.sum(axis=(0, 1))

        self.grad_W_k = key.reshape(-1, self.embed_dim).T @ grad_K.reshape(-1, self.embed_dim)
        self.grad_b_k = grad_K.sum(axis=(0, 1))

        self.grad_W_v = value.reshape(-1, self.embed_dim).T @ grad_V.reshape(-1, self.embed_dim)
        self.grad_b_v = grad_V.sum(axis=(0, 1))

        # Gradient with respect to input (assuming query = key = value)
        grad_input = grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T

        return grad_input

    def compute_head_efficiency(self) -> np.ndarray:
        """
        Compute efficiency score for each attention head.
        Used by dynamic layer manager to add/remove heads.

        Efficiency based on attention entropy:
        - Low entropy = focused attention = more efficient
        - High entropy = diffuse attention = less efficient
        """
        if 'attention_weights' not in self._cache:
            return self.head_efficiency

        weights = self._cache['attention_weights']  # (batch, heads, seq, seq)

        # Entropy-based efficiency (heads with focused attention are more efficient)
        entropy = -np.sum(weights * np.log(weights + 1e-8), axis=-1)
        avg_entropy = np.mean(entropy, axis=(0, 2))  # (heads,)

        # Normalize to [0, 1]
        max_entropy = np.log(weights.shape[-1])
        efficiency = 1.0 - (avg_entropy / (max_entropy + 1e-8))

        # Exponential moving average for stability
        self.head_efficiency = 0.9 * self.head_efficiency + 0.1 * efficiency

        return self.head_efficiency

    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.grad_W_q.fill(0)
        self.grad_W_k.fill(0)
        self.grad_W_v.fill(0)
        self.grad_W_o.fill(0)
        self.grad_b_q.fill(0)
        self.grad_b_k.fill(0)
        self.grad_b_v.fill(0)
        self.grad_b_o.fill(0)

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters"""
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_q, self.b_k, self.b_v, self.b_o]

    def gradients(self) -> List[np.ndarray]:
        """Return list of gradients"""
        return [self.grad_W_q, self.grad_W_k, self.grad_W_v, self.grad_W_o,
                self.grad_b_q, self.grad_b_k, self.grad_b_v, self.grad_b_o]


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    FFN(x) = activation(xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        seed: int = 42
    ):
        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation

        # Weights (He initialization for ReLU-like activations)
        self.W1 = np.random.randn(embed_dim, hidden_dim) * np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(embed_dim)

        # Gradients
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

        # Node efficiency tracking
        self.node_efficiency = np.ones(hidden_dim) * 0.5

        # Cache for backward pass
        self._cache: Dict = {}

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of GELU"""
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit"""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(x.dtype)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # First linear
        pre_act = x @ self.W1 + self.b1

        # Activation
        if self.activation == "gelu":
            hidden = self._gelu(pre_act)
        else:  # relu
            hidden = self._relu(pre_act)

        self._cache = {
            'x': x,
            'pre_act': pre_act,
            'hidden': hidden,
            'training': training
        }

        # Dropout
        if training and self.dropout > 0:
            drop_mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * drop_mask / (1 - self.dropout)
            self._cache['drop_mask'] = drop_mask

        # Second linear
        output = hidden @ self.W2 + self.b2

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients for all weights and return gradient for input.

        Args:
            grad_output: (batch_size, seq_len, embed_dim)

        Returns:
            Gradient with respect to input
        """
        x = self._cache['x']
        pre_act = self._cache['pre_act']
        hidden = self._cache['hidden']

        # Gradient through second linear
        self.grad_W2 = hidden.reshape(-1, self.hidden_dim).T @ grad_output.reshape(-1, self.embed_dim)
        self.grad_b2 = grad_output.sum(axis=(0, 1))
        grad_hidden = grad_output @ self.W2.T

        # Apply dropout mask if used
        if 'drop_mask' in self._cache:
            grad_hidden = grad_hidden * self._cache['drop_mask'] / (1 - self.dropout)

        # Gradient through activation
        if self.activation == "gelu":
            grad_pre_act = grad_hidden * self._gelu_derivative(pre_act)
        else:  # relu
            grad_pre_act = grad_hidden * self._relu_derivative(pre_act)

        # Gradient through first linear
        self.grad_W1 = x.reshape(-1, self.embed_dim).T @ grad_pre_act.reshape(-1, self.hidden_dim)
        self.grad_b1 = grad_pre_act.sum(axis=(0, 1))
        grad_input = grad_pre_act @ self.W1.T

        return grad_input

    def compute_node_efficiency(self) -> np.ndarray:
        """
        Compute efficiency for each hidden node.
        Based on activation magnitude - inactive nodes are inefficient.
        """
        if 'hidden' not in self._cache:
            return self.node_efficiency

        hidden = self._cache['hidden']

        # Activity-based efficiency
        avg_activation = np.mean(np.abs(hidden), axis=(0, 1))  # (hidden_dim,)
        efficiency = avg_activation / (np.max(avg_activation) + 1e-8)

        # Exponential moving average for stability
        # Handle shape mismatch when hidden_dim changes dynamically
        if self.node_efficiency.shape[0] != efficiency.shape[0]:
            self.node_efficiency = efficiency
        else:
            self.node_efficiency = 0.9 * self.node_efficiency + 0.1 * efficiency

        return self.node_efficiency

    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.grad_W1.fill(0)
        self.grad_b1.fill(0)
        self.grad_W2.fill(0)
        self.grad_b2.fill(0)

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters"""
        return [self.W1, self.b1, self.W2, self.b2]

    def gradients(self) -> List[np.ndarray]:
        """Return list of gradients"""
        return [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]


# =============================================================================
# TRANSFORMER ENCODER LAYER
# =============================================================================

class TransformerEncoderLayer:
    """
    Single Transformer Encoder Layer.

    Output = LayerNorm(x + Attention(x))
    Output = LayerNorm(Output + FFN(Output))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        seed: int = 42
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, seed)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, "gelu", seed + 1)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        # Layer efficiency (for dynamic layer management)
        self.efficiency = 0.5

        # Cache for backward pass
        self._cache: Dict = {}

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_output = self.attention.forward(x, x, x, mask, training)

        # Apply dropout to attention output if training
        if training and self.dropout > 0:
            attn_drop_mask = np.random.binomial(1, 1 - self.dropout, attn_output.shape)
            attn_output = attn_output * attn_drop_mask / (1 - self.dropout)
            self._cache['attn_drop_mask'] = attn_drop_mask

        x1 = self.norm1.forward(x + attn_output)

        # FFN with residual connection
        ffn_output = self.ffn.forward(x1, training)

        # Apply dropout to FFN output if training
        if training and self.dropout > 0:
            ffn_drop_mask = np.random.binomial(1, 1 - self.dropout, ffn_output.shape)
            ffn_output = ffn_output * ffn_drop_mask / (1 - self.dropout)
            self._cache['ffn_drop_mask'] = ffn_drop_mask

        x2 = self.norm2.forward(x1 + ffn_output)

        self._cache['x'] = x
        self._cache['x1'] = x1
        self._cache['attn_output'] = attn_output
        self._cache['ffn_output'] = ffn_output

        return x2

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients and return gradient for input.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        x = self._cache['x']
        x1 = self._cache['x1']

        # Backward through norm2
        grad_norm2 = self.norm2.backward(grad_output)

        # Split gradient for residual connection
        grad_ffn = grad_norm2
        grad_x1 = grad_norm2

        # Apply dropout mask if used
        if 'ffn_drop_mask' in self._cache:
            grad_ffn = grad_ffn * self._cache['ffn_drop_mask'] / (1 - self.dropout)

        # Backward through FFN
        grad_x1 += self.ffn.backward(grad_ffn)

        # Backward through norm1
        grad_norm1 = self.norm1.backward(grad_x1)

        # Split gradient for residual connection
        grad_attn = grad_norm1
        grad_x = grad_norm1

        # Apply dropout mask if used
        if 'attn_drop_mask' in self._cache:
            grad_attn = grad_attn * self._cache['attn_drop_mask'] / (1 - self.dropout)

        # Backward through attention
        grad_x += self.attention.backward(grad_attn)

        return grad_x

    def compute_layer_efficiency(self) -> float:
        """Compute overall layer efficiency"""
        head_eff = np.mean(self.attention.compute_head_efficiency())
        node_eff = np.mean(self.ffn.compute_node_efficiency())
        self.efficiency = (head_eff + node_eff) / 2
        return self.efficiency

    def zero_grad(self) -> None:
        """Reset all gradients to zero"""
        self.attention.zero_grad()
        self.ffn.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        """Return list of all parameters"""
        return (self.attention.parameters() +
                self.ffn.parameters() +
                self.norm1.parameters() +
                self.norm2.parameters())

    def gradients(self) -> List[np.ndarray]:
        """Return list of all gradients"""
        return (self.attention.gradients() +
                self.ffn.gradients() +
                self.norm1.gradients() +
                self.norm2.gradients())

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.size for p in self.parameters())
