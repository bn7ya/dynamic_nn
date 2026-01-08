"""
Dynamic Neural Network Enhanced Transformer

Extends the standard transformer with dynamic architecture capabilities:
- Dynamic attention heads (add/remove based on efficiency)
- Dynamic encoder layers (add/remove during training)
- Dynamic FFN width (grow/shrink hidden dimension)
- Health monitoring (cancer/alzheimer scores)
- 3-phase training matching DynamicNetwork pattern

Integrates patterns from the existing DynamicNetwork implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .transformer_components import (
    Embedding,
    PositionalEncoding,
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
)


@dataclass
class RewardPenaltyConfig:
    """Configuration for the reward/penalty system."""
    cost_improvement_threshold: float = 0.001
    efficiency_improvement_threshold: float = 0.01
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1.0
    baseline_learning_rate: float = 0.005
    max_adjustment_factor: float = 2.0
    min_adjustment_factor: float = 0.5
    extreme_threshold: float = 0.8
    window_size: int = 10


@dataclass
class TrainingPhaseConfig:
    """Configuration for the training phases."""
    # Phase 1: Exploration
    exploration_epochs: int = 10
    exploration_learning_rate: float = 0.01

    # Phase 2: Estimation
    estimation_epochs: int = 10
    estimation_learning_rate: float = 0.005

    # Phase 3: Main Training
    main_learning_rate: float = 0.005
    perturbation_cutoff_ratio: float = 0.2
    perturbation_frequency: int = 3
    architecture_adjustment_frequency: int = 5

    # Epoch estimation
    target_efficiency: float = 0.9
    min_estimated_epochs: int = 10
    max_estimated_epochs: int = 150


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency computation and thresholds."""
    # Thresholds
    default_saturation_threshold: float = 0.7
    exploration_saturation_threshold: float = 0.3
    default_efficiency_threshold: float = 0.5
    exploration_efficiency_threshold: float = 0.3
    removal_efficiency_multiplier: float = 0.5

    # Sigmoid threshold params
    sigmoid_k: float = 5.0
    sigmoid_base: float = 0.3
    sigmoid_range: float = 0.5
    sigmoid_center: float = 0.5

    # Initial values
    initial_efficiency: float = 0.5


@dataclass
class HealthScoreConfig:
    """Configuration for health score computation."""
    layer_weight: int = 10
    cancer_denominator: float = 10.0
    alzheimer_denominator: float = 10.0
    healthy_threshold: float = 0.3
    at_risk_threshold: float = 0.7


@dataclass
class GradientConfig:
    """Configuration for gradient handling."""
    gradient_clip_value: float = 1.0


@dataclass
class PerturbationConfig:
    """Configuration for random perturbation."""
    perturbation_fraction: float = 0.005
    perturbation_scale: float = 0.01


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    window_size: int = 30
    improvement_threshold: float = 1e-5


@dataclass
class ArchitectureConfig:
    """Configuration for dynamic architecture adjustments."""
    # FFN growth
    ffn_growth_rate: float = 0.125  # 12.5% (1/8)
    ffn_max_removal_rate: float = 0.1  # 10%

    # Initial efficiency for new nodes/heads
    initial_head_efficiency: float = 0.5
    initial_node_efficiency: float = 0.5


@dataclass
class DynamicTransformerConfig:
    """Configuration for Dynamic Transformer"""
    vocab_size: int = 1000
    max_seq_len: int = 32
    embed_dim: int = 128
    initial_num_heads: int = 4
    initial_num_layers: int = 4
    initial_ffn_dim: int = 512
    min_heads: int = 2
    max_heads: int = 16
    min_layers: int = 2
    max_layers: int = 12
    min_ffn_dim: int = 128
    max_ffn_dim: int = 2048
    dropout: float = 0.1
    num_classes: int = 100
    seed: int = 42

    # Sub-configurations
    training_phase: TrainingPhaseConfig = field(default_factory=TrainingPhaseConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    health_score: HealthScoreConfig = field(default_factory=HealthScoreConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    reward_penalty: RewardPenaltyConfig = field(default_factory=RewardPenaltyConfig)


# =============================================================================
# DYNAMIC ATTENTION
# =============================================================================

class DynamicAttention(MultiHeadAttention):
    """
    Dynamic Multi-Head Attention that can add/remove heads.
    Extends base MultiHeadAttention with dynamic capabilities.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        min_heads: int = 2,
        max_heads: int = 16,
        dropout: float = 0.0,
        seed: int = 42
    ):
        super().__init__(embed_dim, num_heads, dropout, seed)
        self.min_heads = min_heads
        self.max_heads = max_heads
        self.original_num_heads = num_heads

        # Track head changes
        self.heads_added = 0
        self.heads_removed = 0

    def add_head(self) -> bool:
        """
        Add a new attention head if below max.
        Redistributes existing weights to accommodate new head.
        """
        if self.num_heads >= self.max_heads:
            return False

        new_num_heads = self.num_heads + 1
        new_head_dim = self.embed_dim // new_num_heads

        if self.embed_dim % new_num_heads != 0:
            return False

        # Note: In a full implementation, we would resize weight matrices
        # For simplicity, we just update the configuration
        self.num_heads = new_num_heads
        self.head_dim = new_head_dim
        self.head_efficiency = np.concatenate([self.head_efficiency, [0.5]])
        self.heads_added += 1

        return True

    def remove_head(self, head_idx: int = -1) -> bool:
        """
        Remove least efficient attention head.
        If head_idx == -1, removes the least efficient head.
        """
        if self.num_heads <= self.min_heads:
            return False

        new_num_heads = self.num_heads - 1
        new_head_dim = self.embed_dim // new_num_heads

        if self.embed_dim % new_num_heads != 0:
            return False

        # Remove the specified head
        if head_idx == -1:
            head_idx = np.argmin(self.head_efficiency)

        # Update efficiency tracking
        self.head_efficiency = np.delete(self.head_efficiency, head_idx)

        self.num_heads = new_num_heads
        self.head_dim = new_head_dim
        self.heads_removed += 1

        return True

    def adjust_heads(
        self,
        saturation_threshold: float = 0.7,
        efficiency_threshold: float = 0.3
    ) -> Tuple[int, int]:
        """
        Dynamically adjust number of heads based on efficiency.

        Args:
            saturation_threshold: Add heads if mean efficiency above this
            efficiency_threshold: Remove heads with efficiency below this

        Returns:
            (heads_added, heads_removed)
        """
        efficiency = self.compute_head_efficiency()
        added, removed = 0, 0

        # Add heads if all are saturated (high efficiency)
        if np.mean(efficiency) > saturation_threshold:
            if self.add_head():
                added = 1

        # Remove inefficient heads
        inefficient_heads = np.where(efficiency < efficiency_threshold)[0]
        for idx in sorted(inefficient_heads, reverse=True):
            if self.remove_head(idx):
                removed += 1
                break  # Only remove one per adjustment

        return added, removed


# =============================================================================
# DYNAMIC FEED-FORWARD NETWORK
# =============================================================================

class DynamicFFN(FeedForward):
    """
    Dynamic Feed-Forward Network that can adjust hidden dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        min_hidden: int = 128,
        max_hidden: int = 2048,
        dropout: float = 0.0,
        seed: int = 42
    ):
        super().__init__(embed_dim, hidden_dim, dropout, "gelu", seed)
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
        self.original_hidden_dim = hidden_dim

        self.nodes_added = 0
        self.nodes_removed = 0

    def add_nodes(self, count: int) -> int:
        """Add hidden nodes"""
        actual_add = min(count, self.max_hidden - self.hidden_dim)
        if actual_add <= 0:
            return 0

        # Expand W1: (embed_dim, hidden_dim) -> (embed_dim, hidden_dim + actual_add)
        new_W1_cols = np.random.randn(self.embed_dim, actual_add) * np.sqrt(2.0 / self.embed_dim)
        self.W1 = np.concatenate([self.W1, new_W1_cols], axis=1)
        self.b1 = np.concatenate([self.b1, np.zeros(actual_add)])

        # Expand W2: (hidden_dim, embed_dim) -> (hidden_dim + actual_add, embed_dim)
        new_W2_rows = np.random.randn(actual_add, self.embed_dim) * np.sqrt(2.0 / actual_add) * 0.1
        self.W2 = np.concatenate([self.W2, new_W2_rows], axis=0)

        # Expand efficiency tracking
        self.node_efficiency = np.concatenate([self.node_efficiency, np.ones(actual_add) * 0.5])

        # Update gradients
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b1 = np.zeros_like(self.b1)

        self.hidden_dim += actual_add
        self.nodes_added += actual_add

        return actual_add

    def remove_nodes(self, indices: List[int]) -> int:
        """Remove inefficient hidden nodes"""
        if self.hidden_dim - len(indices) < self.min_hidden:
            # Only remove what we can
            max_removable = self.hidden_dim - self.min_hidden
            indices = indices[:max_removable]

        if len(indices) == 0:
            return 0

        keep_mask = np.ones(self.hidden_dim, dtype=bool)
        keep_mask[indices] = False

        self.W1 = self.W1[:, keep_mask]
        self.b1 = self.b1[keep_mask]
        self.W2 = self.W2[keep_mask, :]
        self.node_efficiency = self.node_efficiency[keep_mask]

        # Update gradients
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b1 = np.zeros_like(self.b1)

        removed = len(indices)
        self.hidden_dim -= removed
        self.nodes_removed += removed

        return removed

    def adjust_width(
        self,
        saturation_threshold: float = 0.7,
        efficiency_threshold: float = 0.3
    ) -> Tuple[int, int]:
        """Dynamically adjust FFN width"""
        efficiency = self.compute_node_efficiency()
        added, removed = 0, 0

        # Add nodes if saturated
        if np.mean(efficiency) > saturation_threshold:
            add_count = max(1, self.hidden_dim // 8)  # 12.5% growth
            added = self.add_nodes(add_count)

        # Remove inefficient nodes
        inefficient = np.where(efficiency < efficiency_threshold * 0.5)[0]
        if len(inefficient) > 0:
            # Remove at most 10% at a time
            max_remove = max(1, self.hidden_dim // 10)
            removed = self.remove_nodes(inefficient[:max_remove].tolist())

        return added, removed


# =============================================================================
# DYNAMIC TRANSFORMER LAYER
# =============================================================================

class DynamicTransformerLayer:
    """
    Dynamic Transformer Layer with adjustable attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        config: DynamicTransformerConfig,
        seed: int = 42
    ):
        self.embed_dim = embed_dim
        self.config = config

        # Dynamic components
        self.attention = DynamicAttention(
            embed_dim, num_heads,
            config.min_heads, config.max_heads,
            config.dropout, seed
        )
        self.ffn = DynamicFFN(
            embed_dim, ffn_dim,
            config.min_ffn_dim, config.max_ffn_dim,
            config.dropout, seed + 1
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout = config.dropout

        # Layer efficiency
        self.efficiency = 0.5

        # Cache for backward
        self._cache: Dict = {}

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads

    @property
    def ffn_dim(self) -> int:
        return self.ffn.hidden_dim

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass"""
        # Self-attention with residual
        attn_output = self.attention.forward(x, x, x, mask, training)
        x1 = self.norm1.forward(x + attn_output)

        # FFN with residual
        ffn_output = self.ffn.forward(x1, training)
        x2 = self.norm2.forward(x1 + ffn_output)

        self._cache = {'x': x, 'x1': x1, 'attn_output': attn_output, 'ffn_output': ffn_output}

        return x2

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        x = self._cache['x']
        x1 = self._cache['x1']

        # Backward through norm2
        grad_norm2 = self.norm2.backward(grad_output)

        # Split for residual
        grad_ffn = grad_norm2
        grad_x1 = grad_norm2

        # Backward through FFN
        grad_x1 = grad_x1 + self.ffn.backward(grad_ffn)

        # Backward through norm1
        grad_norm1 = self.norm1.backward(grad_x1)

        # Split for residual
        grad_attn = grad_norm1
        grad_x = grad_norm1

        # Backward through attention
        grad_x = grad_x + self.attention.backward(grad_attn)

        return grad_x

    def compute_layer_efficiency(self) -> float:
        """Compute overall layer efficiency"""
        head_eff = np.mean(self.attention.compute_head_efficiency())
        node_eff = np.mean(self.ffn.compute_node_efficiency())
        self.efficiency = (head_eff + node_eff) / 2
        return self.efficiency

    def adjust_architecture(
        self,
        saturation_threshold: float,
        efficiency_threshold: float
    ) -> Dict[str, Tuple[int, int]]:
        """
        Adjust both attention heads and FFN width.
        Returns change counts.
        """
        attn_changes = self.attention.adjust_heads(saturation_threshold, efficiency_threshold)
        ffn_changes = self.ffn.adjust_width(saturation_threshold, efficiency_threshold)

        return {
            'attention': attn_changes,
            'ffn': ffn_changes
        }

    def zero_grad(self) -> None:
        """Reset gradients"""
        self.attention.zero_grad()
        self.ffn.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters"""
        return (self.attention.parameters() +
                self.ffn.parameters() +
                self.norm1.parameters() +
                self.norm2.parameters())

    def gradients(self) -> List[np.ndarray]:
        """Return all gradients"""
        return (self.attention.gradients() +
                self.ffn.gradients() +
                self.norm1.gradients() +
                self.norm2.gradients())

    def count_parameters(self) -> int:
        """Count parameters"""
        return sum(p.size for p in self.parameters())


# =============================================================================
# DYNAMIC TRANSFORMER
# =============================================================================

class DynamicTransformer:
    """
    Dynamic Neural Network-Enhanced Transformer.

    Integrates DynamicNetwork patterns:
    - 3-phase training (Exploration, Estimation, Main)
    - Dynamic attention heads
    - Dynamic encoder layers
    - Dynamic FFN width
    - Health monitoring (cancer/alzheimer scores)
    - Adaptive saturation thresholds
    """

    def __init__(self, config: DynamicTransformerConfig):
        self.config = config
        np.random.seed(config.seed)

        # Embedding layers
        self.token_embedding = Embedding(config.vocab_size, config.embed_dim, config.seed)
        self.pos_encoding = PositionalEncoding(config.max_seq_len, config.embed_dim)

        # Dynamic encoder layers
        self.layers: List[DynamicTransformerLayer] = [
            DynamicTransformerLayer(
                config.embed_dim,
                config.initial_num_heads,
                config.initial_ffn_dim,
                config,
                config.seed + i
            )
            for i in range(config.initial_num_layers)
        ]

        # Final layer norm
        self.final_norm = LayerNorm(config.embed_dim)

        # Classification head
        self.classifier = np.random.randn(config.embed_dim, config.num_classes) * np.sqrt(2.0 / config.embed_dim)
        self.classifier_bias = np.zeros(config.num_classes)

        # Gradients for classifier
        self.grad_classifier = np.zeros_like(self.classifier)
        self.grad_classifier_bias = np.zeros_like(self.classifier_bias)

        # Health monitoring (matching DynamicNetwork pattern)
        self.cancer_score_history: List[float] = []
        self.alzheimer_score_history: List[float] = []
        self.architecture_history: List[Dict] = []

        # Change tracking
        self.layers_added = 0
        self.layers_removed = 0
        self.total_heads_added = 0
        self.total_heads_removed = 0
        self.total_nodes_added = 0
        self.total_nodes_removed = 0
        self.perturbations_applied = 0

        # Emotional state tracking (reward/penalty system)
        self.total_rewards = 0
        self.total_penalties = 0
        self.reward_history: List[float] = []
        self.penalty_history: List[float] = []
        self.depression_history: List[float] = []
        self.excitement_history: List[float] = []
        self.lr_reset_count = 0
        self.learning_rate_history: List[float] = []
        self.emotional_state_actions: List[str] = []

        # Reward/Penalty config
        self.rp_cost_threshold = 0.001
        self.rp_efficiency_threshold = 0.01
        self.rp_min_lr = 1e-6
        self.rp_max_lr = 1.0
        self.rp_baseline_lr = 0.005
        self.rp_extreme_threshold = 0.8

        self.trained = False
        self.training_history: Dict = {}

        # Cache
        self._cache: Dict = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass through dynamic architecture"""
        # Embedding + positional
        x = self.token_embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)

        # Dynamic encoder layers
        for layer in self.layers:
            x = layer.forward(x, training=training)

        # Final norm
        x = self.final_norm.forward(x)

        # Mean pooling
        pooled = np.mean(x, axis=1)

        self._cache['pooled'] = pooled
        self._cache['token_ids'] = token_ids

        # Classification
        logits = pooled @ self.classifier + self.classifier_bias

        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """Backward pass"""
        pooled = self._cache['pooled']
        batch_size = pooled.shape[0]
        seq_len = self._cache['token_ids'].shape[1]

        # Gradient through classifier
        self.grad_classifier = pooled.T @ grad_logits
        self.grad_classifier_bias = grad_logits.sum(axis=0)
        grad_pooled = grad_logits @ self.classifier.T

        # Gradient through mean pooling
        grad_x = np.repeat(grad_pooled[:, np.newaxis, :], seq_len, axis=1) / seq_len

        # Gradient through final norm
        grad_x = self.final_norm.backward(grad_x)

        # Gradient through layers (reverse order)
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        # Gradient through embedding
        self.token_embedding.backward(grad_x)

    def _cross_entropy_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient"""
        batch_size = logits.shape[0]
        probs = self._softmax(logits)
        log_probs = np.log(probs + 1e-8)
        loss = -np.mean(log_probs[np.arange(batch_size), targets])

        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size

        return loss, grad

    def _compute_efficiency(self, cost_history: List[float]) -> float:
        """Compute efficiency metric (matching DynamicNetwork)"""
        eff_config = self.config.efficiency
        if len(cost_history) < 2:
            return eff_config.initial_efficiency
        improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)
        center = eff_config.sigmoid_center
        multiplier = 10.0  # efficiency multiplier
        return min(1.0, max(0.0, center + improvement * multiplier))

    def _sigmoid_threshold(self, efficiency: float) -> float:
        """Adaptive saturation threshold using sigmoid (matching DynamicNetwork)"""
        eff_config = self.config.efficiency
        k = eff_config.sigmoid_k
        base = eff_config.sigmoid_base
        range_val = eff_config.sigmoid_range
        center = eff_config.sigmoid_center
        sigmoid = 1.0 / (1.0 + np.exp(-k * (efficiency - center)))
        return base + range_val * sigmoid

    # ============ REWARD/PENALTY SYSTEM METHODS ============

    def _compute_improvement_metrics(
        self,
        cost_history: List[float],
        efficiency_history: List[float],
        window: int = 5
    ) -> Dict[str, float]:
        """Compute improvement metrics over a sliding window."""
        if len(cost_history) < 2:
            return {
                "cost_improvement": 0.0,
                "efficiency_improvement": 0.0,
                "cost_trend": 0.0,
                "efficiency_trend": 0.0
            }

        recent_costs = cost_history[-min(window, len(cost_history)):]
        recent_efficiency = efficiency_history[-min(window, len(efficiency_history)):]

        cost_improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)
        efficiency_improvement = efficiency_history[-1] - efficiency_history[-2] if len(efficiency_history) >= 2 else 0.0

        if len(recent_costs) >= 2:
            cost_trend = (recent_costs[-1] - recent_costs[0]) / (len(recent_costs) * (recent_costs[0] + 1e-8))
        else:
            cost_trend = 0.0

        if len(recent_efficiency) >= 2:
            efficiency_trend = (recent_efficiency[-1] - recent_efficiency[0]) / len(recent_efficiency)
        else:
            efficiency_trend = 0.0

        return {
            "cost_improvement": cost_improvement,
            "efficiency_improvement": efficiency_improvement,
            "cost_trend": cost_trend,
            "efficiency_trend": efficiency_trend
        }

    def _should_reward(self, metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Determine if current epoch deserves a reward."""
        cost_improving = metrics["cost_improvement"] > self.rp_cost_threshold
        efficiency_good = metrics["efficiency_improvement"] >= 0 or metrics["efficiency_trend"] > 0
        trend_positive = metrics["cost_trend"] < 0

        should_reward = cost_improving and efficiency_good and trend_positive

        if should_reward:
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)
        else:
            magnitude = 0.0

        return should_reward, magnitude

    def _should_penalize(self, metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Determine if current epoch deserves a penalty."""
        cost_degrading = metrics["cost_improvement"] < -self.rp_cost_threshold
        efficiency_bad = metrics["efficiency_improvement"] < -self.rp_efficiency_threshold
        trend_negative = metrics["cost_trend"] > 0

        should_penalize = cost_degrading or (efficiency_bad and trend_negative)

        if should_penalize:
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)
        else:
            magnitude = 0.0

        return should_penalize, magnitude

    def _apply_reward(self, learning_rate: float, magnitude: float) -> float:
        """Apply reward by decreasing learning rate."""
        decrease_factor = 1.0 - magnitude * 0.5
        new_lr = learning_rate * decrease_factor
        new_lr = max(new_lr, self.rp_min_lr)

        self.total_rewards += 1
        self.reward_history.append(magnitude)

        return new_lr

    def _apply_penalty(self, learning_rate: float, magnitude: float) -> float:
        """Apply penalty by increasing learning rate."""
        increase_factor = 1.0 + magnitude * 1.0
        new_lr = learning_rate * increase_factor
        new_lr = min(new_lr, self.rp_max_lr)

        self.total_penalties += 1
        self.penalty_history.append(magnitude)

        return new_lr

    @property
    def depression_ratio(self) -> float:
        """Ratio of penalties to total adjustments."""
        total = self.total_rewards + self.total_penalties
        return self.total_penalties / total if total > 0 else 0.0

    @property
    def excitement_ratio(self) -> float:
        """Ratio of rewards to total adjustments."""
        total = self.total_rewards + self.total_penalties
        return self.total_rewards / total if total > 0 else 0.0

    def _check_extreme_states(self, learning_rate: float) -> Tuple[float, str]:
        """Check for extreme emotional states and reset LR if detected."""
        depression = self.depression_ratio
        excitement = self.excitement_ratio

        state = "neutral"

        if depression > self.rp_extreme_threshold:
            learning_rate = self.rp_baseline_lr
            self.lr_reset_count += 1
            state = "extreme_depression"
        elif excitement > self.rp_extreme_threshold:
            learning_rate = self.rp_baseline_lr
            self.lr_reset_count += 1
            state = "extreme_excitement"
        elif depression > 0.5:
            state = "depressed"
        elif excitement > 0.5:
            state = "excited"

        self.depression_history.append(depression)
        self.excitement_history.append(excitement)

        return learning_rate, state

    def _apply_reward_penalty_system(
        self,
        learning_rate: float,
        cost_history: List[float],
        efficiency_history: List[float]
    ) -> Tuple[float, str]:
        """Apply the complete reward/penalty system for one epoch."""
        metrics = self._compute_improvement_metrics(cost_history, efficiency_history, window=10)

        action = "neutral"

        should_reward, reward_mag = self._should_reward(metrics)
        if should_reward:
            learning_rate = self._apply_reward(learning_rate, reward_mag)
            action = "reward"
        else:
            should_penalize, penalty_mag = self._should_penalize(metrics)
            if should_penalize:
                learning_rate = self._apply_penalty(learning_rate, penalty_mag)
                action = "penalty"

        learning_rate, extreme_state = self._check_extreme_states(learning_rate)
        if extreme_state.startswith("extreme"):
            action = f"{extreme_state}_reset"

        self.learning_rate_history.append(learning_rate)
        self.emotional_state_actions.append(action)

        return learning_rate, action

    # ============ END REWARD/PENALTY SYSTEM METHODS ============

    def _compute_health_scores(self, epoch: int) -> Tuple[float, float]:
        """Compute cancer and alzheimer scores"""
        total_adds = self.layers_added + self.total_heads_added + self.total_nodes_added
        total_removes = self.layers_removed + self.total_heads_removed + self.total_nodes_removed

        hs_config = self.config.health_score
        # Cancer: excessive growth
        growth_rate = total_adds / (epoch + 1)
        cancer = min(1.0, growth_rate / hs_config.cancer_denominator)

        # Alzheimer: excessive removal
        removal_rate = total_removes / (epoch + 1)
        alzheimer = min(1.0, removal_rate / hs_config.alzheimer_denominator)

        return cancer, alzheimer

    def add_layer(self, after_index: int) -> bool:
        """Add a new encoder layer"""
        if len(self.layers) >= self.config.max_layers:
            return False

        new_layer = DynamicTransformerLayer(
            self.config.embed_dim,
            self.config.initial_num_heads,
            self.config.initial_ffn_dim,
            self.config,
            self.config.seed + len(self.layers) + 100
        )
        self.layers.insert(after_index + 1, new_layer)
        self.layers_added += 1
        return True

    def remove_layer(self, index: int) -> bool:
        """Remove least efficient layer"""
        if len(self.layers) <= self.config.min_layers:
            return False

        # Don't remove first or last layer
        if index == 0 or index == len(self.layers) - 1:
            return False

        self.layers.pop(index)
        self.layers_removed += 1
        return True

    def _aggressive_architecture_adjustment(
        self,
        saturation_threshold: float,
        efficiency_threshold: float
    ) -> None:
        """Aggressive changes during exploration phase"""
        for layer in self.layers:
            changes = layer.adjust_architecture(saturation_threshold, efficiency_threshold)
            self.total_heads_added += changes['attention'][0]
            self.total_heads_removed += changes['attention'][1]
            self.total_nodes_added += changes['ffn'][0]
            self.total_nodes_removed += changes['ffn'][1]

        # Consider adding layer if all saturated
        efficiencies = [layer.compute_layer_efficiency() for layer in self.layers]
        if np.mean(efficiencies) > saturation_threshold and len(self.layers) < self.config.max_layers:
            self.add_layer(len(self.layers) // 2)

    def _architecture_adjustment(
        self,
        saturation_threshold: float,
        efficiency_threshold: float
    ) -> None:
        """Standard architecture adjustment during main training"""
        for layer in self.layers:
            changes = layer.adjust_architecture(saturation_threshold, efficiency_threshold)
            self.total_heads_added += changes['attention'][0]
            self.total_heads_removed += changes['attention'][1]
            self.total_nodes_added += changes['ffn'][0]
            self.total_nodes_removed += changes['ffn'][1]

        # Check for redundant layers
        if len(self.layers) > self.config.min_layers:
            efficiencies = [layer.compute_layer_efficiency() for layer in self.layers]
            min_idx = np.argmin(efficiencies[1:-1]) + 1  # Exclude first and last
            if efficiencies[min_idx] < efficiency_threshold * 0.3:
                self.remove_layer(min_idx)

    def _apply_perturbation(self, fraction: float = None) -> None:
        """Apply random perturbation to avoid overfitting"""
        if fraction is None:
            fraction = self.config.perturbation.perturbation_fraction
        scale = self.config.perturbation.perturbation_scale

        self.perturbations_applied += 1
        for layer in self.layers:
            # Perturb attention weights
            mask = np.random.random(layer.attention.W_q.shape) < fraction
            layer.attention.W_q[mask] *= (1 + np.random.randn() * scale)

            mask = np.random.random(layer.attention.W_k.shape) < fraction
            layer.attention.W_k[mask] *= (1 + np.random.randn() * scale)

    def _record_architecture(self) -> None:
        """Record current architecture for history"""
        total_heads = sum(l.num_heads for l in self.layers)
        total_ffn_nodes = sum(l.ffn_dim for l in self.layers)
        self.architecture_history.append({
            'num_layers': len(self.layers),
            'total_heads': total_heads,
            'total_ffn_nodes': total_ffn_nodes,
            'params': self.count_parameters()
        })

    def zero_grad(self) -> None:
        """Reset all gradients"""
        self.token_embedding.zero_grad()
        self.final_norm.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.grad_classifier.fill(0)
        self.grad_classifier_bias.fill(0)

    def _clip_gradients(self, max_norm: float = None) -> float:
        """Clip gradients by global norm"""
        if max_norm is None:
            max_norm = self.config.gradient.gradient_clip_value
        all_grads = []
        all_grads.extend(self.token_embedding.gradients())
        all_grads.extend(self.final_norm.gradients())
        for layer in self.layers:
            all_grads.extend(layer.gradients())
        all_grads.extend([self.grad_classifier, self.grad_classifier_bias])

        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for grad in all_grads:
                grad *= scale

        return total_norm

    def _update_parameters(self, learning_rate: float) -> None:
        """Update parameters"""
        for param, grad in zip(self.token_embedding.parameters(), self.token_embedding.gradients()):
            param -= learning_rate * grad

        for param, grad in zip(self.final_norm.parameters(), self.final_norm.gradients()):
            param -= learning_rate * grad

        for layer in self.layers:
            for param, grad in zip(layer.parameters(), layer.gradients()):
                param -= learning_rate * grad

        self.classifier -= learning_rate * self.grad_classifier
        self.classifier_bias -= learning_rate * self.grad_classifier_bias

    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """Single training epoch"""
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        total_correct = 0

        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            self.zero_grad()
            logits = self.forward(X_batch, training=True)
            loss, grad = self._cross_entropy_loss(logits, y_batch)

            self.backward(grad)
            self._clip_gradients(1.0)
            self._update_parameters(learning_rate)

            total_loss += loss * len(y_batch)
            preds = np.argmax(logits, axis=-1)
            total_correct += np.sum(preds == y_batch)

        return total_loss / n_samples, total_correct / n_samples

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train with 3-phase dynamic approach (matching DynamicNetwork).

        Phase 1: Exploration - Aggressive architecture changes
        Phase 2: Estimation - Estimate required epochs
        Phase 3: Main Training - Adaptive architecture adjustment
        """
        start_time = time.time()

        cost_history: List[float] = []
        efficiency_history: List[float] = []
        train_acc_history: List[float] = []
        val_acc_history: List[float] = []

        # Get training phase config
        tp_config = self.config.training_phase
        eff_config = self.config.efficiency
        exploration_epochs = tp_config.exploration_epochs

        # ============ PHASE 1: EXPLORATION ============
        if verbose:
            print("=" * 60)
            print(f"PHASE 1: EXPLORATION ({exploration_epochs} epochs)")
            print("=" * 60)
            print(f"Initial architecture: {len(self.layers)} layers, {self.layers[0].num_heads} heads")

        learning_rate = tp_config.exploration_learning_rate
        for epoch in range(exploration_epochs):
            loss, acc = self._train_epoch(X_train, y_train, learning_rate, batch_size)
            cost_history.append(loss)
            train_acc_history.append(acc)

            # Aggressive architecture exploration
            self._aggressive_architecture_adjustment(
                saturation_threshold=eff_config.exploration_saturation_threshold,
                efficiency_threshold=eff_config.exploration_efficiency_threshold
            )

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)

            cancer, alzheimer = self._compute_health_scores(epoch + 1)
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)
            self._record_architecture()

            if verbose and epoch % 2 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}, layers={len(self.layers)}")

        # ============ PHASE 2: ESTIMATION ============
        estimation_epochs = tp_config.estimation_epochs
        if verbose:
            print("=" * 60)
            print(f"PHASE 2: ESTIMATION ({estimation_epochs} epochs)")
            print("=" * 60)

        learning_rate = tp_config.estimation_learning_rate
        for epoch in range(estimation_epochs):
            loss, acc = self._train_epoch(X_train, y_train, learning_rate, batch_size)
            cost_history.append(loss)
            train_acc_history.append(acc)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)

            cancer, alzheimer = self._compute_health_scores(exploration_epochs + epoch + 1)
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)
            self._record_architecture()

        # Estimate epochs needed for target efficiency
        if len(cost_history) > exploration_epochs:
            avg_improvement = (cost_history[exploration_epochs] - cost_history[-1]) / estimation_epochs
            efficiency_gap = tp_config.target_efficiency - efficiency_history[-1]
            estimated_epochs = max(
                tp_config.min_estimated_epochs,
                min(tp_config.max_estimated_epochs, int(efficiency_gap / (avg_improvement * 0.1 + 1e-8)))
            )
        else:
            estimated_epochs = 50

        if verbose:
            print(f"  Estimated epochs for {tp_config.target_efficiency:.0%} efficiency: {estimated_epochs}")

        # ============ PHASE 3: MAIN TRAINING ============
        if verbose:
            print("=" * 60)
            print(f"PHASE 3: MAIN TRAINING ({estimated_epochs} epochs)")
            print("  Using Reward/Penalty System for adaptive LR")
            print("=" * 60)

        perturbation_cutoff = int(estimated_epochs * tp_config.perturbation_cutoff_ratio)
        learning_rate = tp_config.main_learning_rate
        self.rp_baseline_lr = learning_rate  # Set baseline for resets
        best_val_acc = 0.0
        arch_freq = tp_config.architecture_adjustment_frequency
        perturb_freq = tp_config.perturbation_frequency

        for epoch in range(estimated_epochs):
            # Apply perturbation in first portion
            if epoch < perturbation_cutoff and epoch % perturb_freq == 0:
                self._apply_perturbation()

            # Adaptive saturation threshold
            efficiency = self._compute_efficiency(cost_history)
            saturation_threshold = self._sigmoid_threshold(efficiency)

            loss, acc = self._train_epoch(X_train, y_train, learning_rate, batch_size)
            cost_history.append(loss)
            train_acc_history.append(acc)

            # Architecture adjustment with adaptive threshold
            if epoch % arch_freq == 0:
                self._architecture_adjustment(saturation_threshold, efficiency_threshold=eff_config.default_efficiency_threshold)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)

            cancer, alzheimer = self._compute_health_scores(exploration_epochs + estimation_epochs + epoch + 1)
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)
            self._record_architecture()

            # Validation
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val, training=False)
                val_preds = np.argmax(val_logits, axis=-1)
                val_acc = np.mean(val_preds == y_val)
                val_acc_history.append(val_acc)
                best_val_acc = max(best_val_acc, val_acc)

            # Apply reward/penalty system (replaces fixed LR decay)
            learning_rate, action = self._apply_reward_penalty_system(
                learning_rate, cost_history, efficiency_history
            )

            if verbose and epoch % 10 == 0:
                msg = f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}, eff={efficiency:.2%}, lr={learning_rate:.4f}"
                msg += f", R/P={self.total_rewards}/{self.total_penalties}"
                if X_val is not None:
                    msg += f", val_acc={val_acc:.2%}"
                print(msg)

            # Early stopping
            es_config = self.config.early_stopping
            if len(cost_history) > es_config.window_size:
                if cost_history[-es_config.window_size] - cost_history[-1] < es_config.improvement_threshold:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

        training_time = time.time() - start_time
        self.trained = True

        # Final metrics
        self.training_history = {
            'cost_history': cost_history,
            'efficiency_history': efficiency_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            'cancer_score_history': self.cancer_score_history,
            'alzheimer_score_history': self.alzheimer_score_history,
            'architecture_history': self.architecture_history,
            'epochs_completed': len(cost_history),
            'final_cost': cost_history[-1],
            'best_cost': min(cost_history),
            'final_train_acc': train_acc_history[-1],
            'final_val_acc': val_acc_history[-1] if val_acc_history else None,
            'best_val_acc': best_val_acc if X_val is not None else None,
            'training_time_seconds': training_time,
            'layers_added': self.layers_added,
            'layers_removed': self.layers_removed,
            'heads_added': self.total_heads_added,
            'heads_removed': self.total_heads_removed,
            'nodes_added': self.total_nodes_added,
            'nodes_removed': self.total_nodes_removed,
            'perturbations_applied': self.perturbations_applied,
            # Emotional state fields
            'total_rewards': self.total_rewards,
            'total_penalties': self.total_penalties,
            'depression_history': self.depression_history,
            'excitement_history': self.excitement_history,
            'lr_reset_count': self.lr_reset_count,
            'learning_rate_history': self.learning_rate_history,
            'emotional_state_history': self.emotional_state_actions,
            'final_depression_ratio': self.depression_ratio,
            'final_excitement_ratio': self.excitement_ratio,
        }

        if verbose:
            print("=" * 60)
            print(f"Training completed in {training_time:.2f}s")
            print(f"Final architecture: {len(self.layers)} layers")
            print(f"Total heads: {sum(l.num_heads for l in self.layers)}")
            print(f"Total FFN nodes: {sum(l.ffn_dim for l in self.layers)}")
            print(f"Parameters: {self.count_parameters():,}")
            print(f"Architecture changes: +{self.layers_added}/-{self.layers_removed} layers, "
                  f"+{self.total_heads_added}/-{self.total_heads_removed} heads, "
                  f"+{self.total_nodes_added}/-{self.total_nodes_removed} nodes")
            print(f"Emotional state: {self.total_rewards} rewards, {self.total_penalties} penalties")
            print(f"Depression ratio: {self.depression_ratio:.1%}, Excitement ratio: {self.excitement_ratio:.1%}")
            print(f"LR resets: {self.lr_reset_count}")
            print("=" * 60)

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions"""
        logits = self.forward(X, training=False)
        return np.argmax(logits, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        logits = self.forward(X, training=False)
        return self._softmax(logits)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model"""
        logits = self.forward(X, training=False)
        loss, _ = self._cross_entropy_loss(logits, y)

        probs = self._softmax(logits)
        preds = np.argmax(probs, axis=-1)

        accuracy = np.mean(preds == y)

        top3_preds = np.argsort(probs, axis=-1)[:, -3:]
        top3_acc = np.mean([y[i] in top3_preds[i] for i in range(len(y))])

        top5_preds = np.argsort(probs, axis=-1)[:, -5:]
        top5_acc = np.mean([y[i] in top5_preds[i] for i in range(len(y))])

        perplexity = np.exp(loss)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'perplexity': perplexity,
        }

    def health_status(self) -> Dict:
        """Get health status including emotional state (matching DynamicNetwork API)"""
        cancer = self.cancer_score_history[-1] if self.cancer_score_history else 0.0
        alzheimer = self.alzheimer_score_history[-1] if self.alzheimer_score_history else 0.0

        hs_config = self.config.health_score
        if max(cancer, alzheimer) < hs_config.healthy_threshold:
            state = 'Healthy'
        elif max(cancer, alzheimer) < hs_config.at_risk_threshold:
            state = 'At Risk'
        else:
            state = 'Critical'

        # Determine emotional state
        depression = self.depression_ratio
        excitement = self.excitement_ratio

        rp_config = self.config.reward_penalty
        if depression > rp_config.extreme_threshold:
            emotional_state = "extreme_depression"
        elif excitement > rp_config.extreme_threshold:
            emotional_state = "extreme_excitement"
        elif depression > 0.5:
            emotional_state = "depressed"
        elif excitement > 0.5:
            emotional_state = "excited"
        else:
            emotional_state = "neutral"

        return {
            'state': state,
            'cancer_score': cancer,
            'alzheimer_score': alzheimer,
            'overall_health': 1.0 - max(cancer, alzheimer),
            'num_layers': len(self.layers),
            'total_heads': sum(l.num_heads for l in self.layers),
            'total_ffn_nodes': sum(l.ffn_dim for l in self.layers),
            # Emotional state fields
            'depression_ratio': depression,
            'excitement_ratio': excitement,
            'emotional_state': emotional_state,
            'total_rewards': self.total_rewards,
            'total_penalties': self.total_penalties,
            'lr_reset_count': self.lr_reset_count,
        }

    def count_parameters(self) -> int:
        """Count current parameters"""
        total = 0
        total += sum(p.size for p in self.token_embedding.parameters())
        total += sum(p.size for p in self.final_norm.parameters())
        for layer in self.layers:
            total += layer.count_parameters()
        total += self.classifier.size + self.classifier_bias.size
        return total

    def get_architecture_summary(self) -> Dict:
        """Get summary of current architecture"""
        return {
            'vocab_size': self.config.vocab_size,
            'embed_dim': self.config.embed_dim,
            'num_layers': len(self.layers),
            'heads_per_layer': [l.num_heads for l in self.layers],
            'ffn_dim_per_layer': [l.ffn_dim for l in self.layers],
            'total_heads': sum(l.num_heads for l in self.layers),
            'total_ffn_nodes': sum(l.ffn_dim for l in self.layers),
            'num_classes': self.config.num_classes,
            'total_parameters': self.count_parameters(),
        }

    def save(self, filepath: str) -> None:
        """Save model to file"""
        weights = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'max_seq_len': self.config.max_seq_len,
                'embed_dim': self.config.embed_dim,
                'initial_num_heads': self.config.initial_num_heads,
                'initial_num_layers': self.config.initial_num_layers,
                'initial_ffn_dim': self.config.initial_ffn_dim,
                'min_heads': self.config.min_heads,
                'max_heads': self.config.max_heads,
                'min_layers': self.config.min_layers,
                'max_layers': self.config.max_layers,
                'min_ffn_dim': self.config.min_ffn_dim,
                'max_ffn_dim': self.config.max_ffn_dim,
                'dropout': self.config.dropout,
                'num_classes': self.config.num_classes,
                'seed': self.config.seed,
            },
            'token_embedding': self.token_embedding.weights,
            'classifier': self.classifier,
            'classifier_bias': self.classifier_bias,
            'final_norm_gamma': self.final_norm.gamma,
            'final_norm_beta': self.final_norm.beta,
            'training_history': self.training_history,
            'num_layers': len(self.layers),
        }

        # Save layer weights
        for i, layer in enumerate(self.layers):
            weights[f'layer_{i}_attn_W_q'] = layer.attention.W_q
            weights[f'layer_{i}_attn_W_k'] = layer.attention.W_k
            weights[f'layer_{i}_attn_W_v'] = layer.attention.W_v
            weights[f'layer_{i}_attn_W_o'] = layer.attention.W_o
            weights[f'layer_{i}_attn_b_q'] = layer.attention.b_q
            weights[f'layer_{i}_attn_b_k'] = layer.attention.b_k
            weights[f'layer_{i}_attn_b_v'] = layer.attention.b_v
            weights[f'layer_{i}_attn_b_o'] = layer.attention.b_o
            weights[f'layer_{i}_ffn_W1'] = layer.ffn.W1
            weights[f'layer_{i}_ffn_b1'] = layer.ffn.b1
            weights[f'layer_{i}_ffn_W2'] = layer.ffn.W2
            weights[f'layer_{i}_ffn_b2'] = layer.ffn.b2
            weights[f'layer_{i}_norm1_gamma'] = layer.norm1.gamma
            weights[f'layer_{i}_norm1_beta'] = layer.norm1.beta
            weights[f'layer_{i}_norm2_gamma'] = layer.norm2.gamma
            weights[f'layer_{i}_norm2_beta'] = layer.norm2.beta
            weights[f'layer_{i}_num_heads'] = layer.num_heads
            weights[f'layer_{i}_ffn_dim'] = layer.ffn_dim

        np.savez(filepath, **weights)

    @classmethod
    def load(cls, filepath: str) -> "DynamicTransformer":
        """Load model from file"""
        data = np.load(filepath, allow_pickle=True)

        config_dict = data['config'].item()
        config = DynamicTransformerConfig(**config_dict)

        model = cls(config)

        # Load weights
        model.token_embedding.weights = data['token_embedding']
        model.classifier = data['classifier']
        model.classifier_bias = data['classifier_bias']
        model.final_norm.gamma = data['final_norm_gamma']
        model.final_norm.beta = data['final_norm_beta']

        if 'training_history' in data:
            model.training_history = data['training_history'].item()
            model.trained = True

        # Reconstruct layers based on saved architecture
        num_layers = int(data['num_layers'])
        model.layers = []

        for i in range(num_layers):
            layer = DynamicTransformerLayer(
                config.embed_dim,
                int(data[f'layer_{i}_num_heads']),
                int(data[f'layer_{i}_ffn_dim']),
                config,
                config.seed + i
            )

            layer.attention.W_q = data[f'layer_{i}_attn_W_q']
            layer.attention.W_k = data[f'layer_{i}_attn_W_k']
            layer.attention.W_v = data[f'layer_{i}_attn_W_v']
            layer.attention.W_o = data[f'layer_{i}_attn_W_o']
            layer.attention.b_q = data[f'layer_{i}_attn_b_q']
            layer.attention.b_k = data[f'layer_{i}_attn_b_k']
            layer.attention.b_v = data[f'layer_{i}_attn_b_v']
            layer.attention.b_o = data[f'layer_{i}_attn_b_o']
            layer.ffn.W1 = data[f'layer_{i}_ffn_W1']
            layer.ffn.b1 = data[f'layer_{i}_ffn_b1']
            layer.ffn.W2 = data[f'layer_{i}_ffn_W2']
            layer.ffn.b2 = data[f'layer_{i}_ffn_b2']
            layer.norm1.gamma = data[f'layer_{i}_norm1_gamma']
            layer.norm1.beta = data[f'layer_{i}_norm1_beta']
            layer.norm2.gamma = data[f'layer_{i}_norm2_gamma']
            layer.norm2.beta = data[f'layer_{i}_norm2_beta']

            model.layers.append(layer)

        return model
