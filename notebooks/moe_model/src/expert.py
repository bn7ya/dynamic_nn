"""
Expert Network implementation for MoE.

Each expert is a mini feedforward network with efficiency tracking
similar to DynamicNetwork nodes.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    from .moe_config import ExpertConfig
except ImportError:
    from moe_config import ExpertConfig


class Expert:
    """
    Individual expert network - a mini feedforward network.

    Tracks efficiency metrics like DynamicNetwork nodes:
    - Activation count and utilization rate
    - Node-level efficiency scores
    - Variance and gradient tracking
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: ExpertConfig,
        expert_id: int,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.expert_id = expert_id

        # Efficiency tracking (from DNN pattern)
        self.efficiency = config.initial_efficiency
        self.efficiency_history: List[float] = []

        # Usage statistics
        self.activation_count = 0
        self.total_count = 0
        self.recent_activations = 0
        self.recent_total = 0

        # Gradient tracking
        self.gradient_magnitude = 0.0
        self.gradient_history: List[float] = []

        # Activation statistics
        self.activation_mean = 0.0
        self.activation_variance = 0.0

        # Loss tracking for quality measurement
        self.recent_loss = 0.0
        self.loss_history: List[float] = []

        # Initialize layers
        self._init_layers(seed)

        # Cache for backward pass
        self._cache = {}

    def _init_layers(self, seed: int) -> None:
        """Initialize expert layers with He initialization."""
        np.random.seed(seed + self.expert_id)
        hidden = self.config.hidden_dim

        # Two-layer FFN (standard transformer FFN structure)
        # Layer 1: input_dim -> hidden_dim
        self.W1 = np.random.randn(self.input_dim, hidden) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden)

        # Layer 2: hidden_dim -> output_dim
        self.W2 = np.random.randn(hidden, self.output_dim) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(self.output_dim)

        # Gradient accumulators
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # Node-level efficiency (for hidden layer)
        self.node_efficiency = np.ones(hidden) * self.config.initial_efficiency
        self.node_activation_counts = np.zeros(hidden)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through expert.

        Args:
            x: Input tensor (batch, input_dim)
            training: Whether in training mode

        Returns:
            Output tensor (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Layer 1: Linear + ReLU
        z1 = x @ self.W1 + self.b1  # (batch, hidden)

        if self.config.activation == 'relu':
            h = np.maximum(0, z1)
        elif self.config.activation == 'gelu':
            h = 0.5 * z1 * (1 + np.tanh(np.sqrt(2/np.pi) * (z1 + 0.044715 * z1**3)))
        else:
            h = np.maximum(0, z1)  # Default to ReLU

        # Track activation statistics for efficiency
        if training:
            self._update_activation_stats(h)
            self._cache = {'x': x, 'z1': z1, 'h': h}

        # Layer 2: Linear (no activation - will be combined by router)
        out = h @ self.W2 + self.b2  # (batch, output_dim)

        # Update usage statistics
        if training:
            self.activation_count += batch_size
            self.total_count += batch_size
            self.recent_activations += batch_size
            self.recent_total += batch_size

        return out

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through expert.

        Args:
            grad_output: Gradient of loss w.r.t. output (batch, output_dim)
            learning_rate: Learning rate for parameter updates

        Returns:
            Gradient of loss w.r.t. input (batch, input_dim)
        """
        x = self._cache['x']
        z1 = self._cache['z1']
        h = self._cache['h']
        batch_size = x.shape[0]

        # Gradient through Layer 2
        self.dW2 = h.T @ grad_output / batch_size
        self.db2 = np.mean(grad_output, axis=0)
        grad_h = grad_output @ self.W2.T  # (batch, hidden)

        # Gradient through activation
        if self.config.activation == 'relu':
            grad_z1 = grad_h * (z1 > 0)
        else:
            grad_z1 = grad_h * (z1 > 0)  # Default ReLU derivative

        # Gradient through Layer 1
        self.dW1 = x.T @ grad_z1 / batch_size
        self.db1 = np.mean(grad_z1, axis=0)
        grad_input = grad_z1 @ self.W1.T  # (batch, input_dim)

        # Track gradient magnitude
        grad_mag = np.sqrt(np.mean(self.dW1**2) + np.mean(self.dW2**2))
        self.gradient_magnitude = grad_mag
        self.gradient_history.append(grad_mag)

        # Update parameters
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.db2 -= learning_rate * self.db2

        # Update node-level efficiency based on gradients
        self._update_node_efficiency(grad_z1)

        return grad_input

    def _update_activation_stats(self, h: np.ndarray) -> None:
        """Update activation statistics for efficiency computation."""
        # Per-node activation rates (how often each node is active)
        active_mask = h > 0
        node_activation_rate = np.mean(active_mask, axis=0)
        self.node_activation_counts += np.sum(active_mask, axis=0)

        # Update running statistics
        batch_mean = np.mean(h)
        batch_var = np.var(h)

        # Exponential moving average
        alpha = 0.1
        self.activation_mean = (1 - alpha) * self.activation_mean + alpha * batch_mean
        self.activation_variance = (1 - alpha) * self.activation_variance + alpha * batch_var

    def _update_node_efficiency(self, grad_z1: np.ndarray) -> None:
        """Update node-level efficiency based on gradient flow."""
        # Nodes with higher gradient magnitude are more "alive"
        node_grad_mag = np.mean(np.abs(grad_z1), axis=0)

        # Efficiency based on gradient flow (dead neurons have zero gradient)
        grad_efficiency = np.clip(node_grad_mag / (np.mean(node_grad_mag) + 1e-8), 0, 2)

        # Update with exponential moving average
        decay = self.config.efficiency_decay
        scale = self.config.efficiency_update_scale
        self.node_efficiency = decay * self.node_efficiency + scale * grad_efficiency

    def compute_efficiency(self) -> float:
        """
        Compute expert efficiency using DNN-style formula.

        Efficiency is based on:
        1. Utilization rate (how often expert is used)
        2. Node-level health (gradient flow)
        3. Output quality (inverse of recent loss)
        """
        if self.total_count == 0:
            return self.config.initial_efficiency

        # Utilization score
        utilization = self.activation_count / max(1, self.total_count)
        S_util = min(1.0, utilization * 4)  # Scale to [0, 1], expect ~25% usage

        # Node health score (mean of node efficiencies)
        S_node = np.mean(self.node_efficiency)

        # Gradient flow score
        if len(self.gradient_history) > 0:
            recent_grad = np.mean(self.gradient_history[-10:])
            S_grad = min(1.0, recent_grad / 0.1)  # Normalize to expected range
        else:
            S_grad = 0.5

        # Quality score (lower loss = higher quality)
        if len(self.loss_history) > 0:
            recent_loss = np.mean(self.loss_history[-10:])
            S_quality = np.exp(-recent_loss)  # Exponential decay
        else:
            S_quality = 0.5

        # Weighted combination (like DNN efficiency weights)
        w_util = 0.25
        w_node = 0.25
        w_grad = 0.25
        w_quality = 0.25

        efficiency = (w_util * S_util + w_node * S_node +
                      w_grad * S_grad + w_quality * S_quality)

        self.efficiency = np.clip(efficiency, 0, 1)
        self.efficiency_history.append(self.efficiency)

        return self.efficiency

    def reset_recent_stats(self) -> None:
        """Reset recent statistics (called periodically during training)."""
        self.recent_activations = 0
        self.recent_total = 0
        self.recent_loss = 0.0

    def get_stats(self) -> dict:
        """Get expert statistics for monitoring."""
        return {
            'expert_id': self.expert_id,
            'efficiency': self.efficiency,
            'utilization': self.activation_count / max(1, self.total_count),
            'activation_mean': self.activation_mean,
            'activation_variance': self.activation_variance,
            'gradient_magnitude': self.gradient_magnitude,
            'node_efficiency_mean': np.mean(self.node_efficiency),
            'node_efficiency_std': np.std(self.node_efficiency),
            'dead_nodes': np.sum(self.node_efficiency < 0.1),
        }

    def compute_importance(
        self,
        utilization_weight: float = 0.3,
        gradient_weight: float = 0.3,
        loss_weight: float = 0.4
    ) -> float:
        """
        Compute importance score for pruning/merging decisions.

        Lower score = less important = candidate for removal.

        Components:
        1. Utilization score: How often this expert is selected
        2. Gradient magnitude score: Is this expert still learning?
        3. Loss contribution score: How much does this expert reduce loss?

        Args:
            utilization_weight: Weight for utilization component
            gradient_weight: Weight for gradient magnitude component
            loss_weight: Weight for loss contribution component

        Returns:
            float: Importance score in [0, 1], lower = more removable
        """
        # Component 1: Utilization (higher = more important)
        if self.total_count > 0:
            utilization = self.activation_count / self.total_count
            # Scale expecting ~25% usage with 4 experts (1/num_experts)
            S_util = min(1.0, utilization * 4.0)
        else:
            S_util = 0.0  # Never used = not important

        # Component 2: Gradient magnitude (higher gradients = still learning = important)
        if len(self.gradient_history) > 0:
            recent_grads = self.gradient_history[-10:]
            avg_grad = np.mean(recent_grads)
            # Normalize - expect gradients around 0.01-0.1 during active learning
            S_grad = min(1.0, avg_grad / 0.05)
        else:
            S_grad = 0.5  # No gradient history = neutral

        # Component 3: Loss contribution (lower recent loss = more important)
        if len(self.loss_history) > 0:
            recent_loss = np.mean(self.loss_history[-10:])
            # Invert: lower loss = higher importance
            S_loss = np.exp(-recent_loss)  # Bounded to (0, 1]
        else:
            S_loss = 0.5  # No loss history = neutral

        # Weighted combination
        importance = (
            utilization_weight * S_util +
            gradient_weight * S_grad +
            loss_weight * S_loss
        )

        return float(np.clip(importance, 0.0, 1.0))

    def get_weight_vector(self) -> np.ndarray:
        """
        Flatten all expert weights into a single vector for similarity computation.

        Used by merging logic to compute weight-space similarity between experts.

        Returns:
            np.ndarray: Flattened weight vector
        """
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    @classmethod
    def clone_with_noise(
        cls,
        source_expert: 'Expert',
        new_expert_id: int,
        noise_scale: float = 0.1,
        seed: Optional[int] = None
    ) -> 'Expert':
        """
        Create a new expert by cloning an existing one and adding noise.

        This implements the "clone from most-loaded expert" strategy.
        The noise breaks symmetry so the new expert can specialize differently.

        Args:
            source_expert: Expert to clone from
            new_expert_id: ID for the new expert
            noise_scale: Standard deviation of Gaussian noise to add (relative to weight std)
            seed: Random seed for reproducibility

        Returns:
            Expert: New expert with cloned + noised weights
        """
        if seed is not None:
            np.random.seed(seed)

        # Create new expert with same config
        new_expert = cls(
            input_dim=source_expert.input_dim,
            output_dim=source_expert.output_dim,
            config=source_expert.config,
            expert_id=new_expert_id,
            seed=seed if seed else new_expert_id
        )

        # Clone weights with noise
        # Noise magnitude is scaled relative to weight standard deviation
        def add_scaled_noise(weights: np.ndarray) -> np.ndarray:
            weight_std = np.std(weights) + 1e-8
            noise = np.random.randn(*weights.shape) * noise_scale * weight_std
            return weights + noise

        new_expert.W1 = add_scaled_noise(source_expert.W1.copy())
        new_expert.b1 = add_scaled_noise(source_expert.b1.copy())
        new_expert.W2 = add_scaled_noise(source_expert.W2.copy())
        new_expert.b2 = add_scaled_noise(source_expert.b2.copy())

        # Copy efficiency state (start similar to parent)
        new_expert.efficiency = source_expert.efficiency * 0.9  # Slightly lower initially
        new_expert.node_efficiency = source_expert.node_efficiency.copy() * 0.9

        # Reset statistics - new expert starts fresh
        new_expert.activation_count = 0
        new_expert.total_count = 0
        new_expert.recent_activations = 0
        new_expert.recent_total = 0
        new_expert.gradient_magnitude = 0.0
        new_expert.gradient_history = []
        new_expert.loss_history = []
        new_expert.efficiency_history = []

        return new_expert

    @classmethod
    def merge_experts(
        cls,
        expert1: 'Expert',
        expert2: 'Expert',
        merged_expert_id: int,
        efficiency_weight: float = 0.5,
        seed: Optional[int] = None
    ) -> 'Expert':
        """
        Merge two experts into one by averaging their weights.

        The merge is weighted by the efficiency of each expert, so the
        more efficient expert contributes more to the merged weights.

        Args:
            expert1: First expert to merge
            expert2: Second expert to merge
            merged_expert_id: ID for the merged expert
            efficiency_weight: Blend factor based on efficiency (0=average, 1=fully efficiency-weighted)
            seed: Random seed

        Returns:
            Expert: New merged expert
        """
        if seed is not None:
            np.random.seed(seed)

        # Compute merge weights based on efficiency
        eff1 = expert1.compute_efficiency()
        eff2 = expert2.compute_efficiency()

        # Interpolate between equal weighting and efficiency-based weighting
        equal_w1, equal_w2 = 0.5, 0.5
        if eff1 + eff2 > 0:
            eff_w1 = eff1 / (eff1 + eff2)
            eff_w2 = eff2 / (eff1 + eff2)
        else:
            eff_w1, eff_w2 = 0.5, 0.5

        w1 = (1 - efficiency_weight) * equal_w1 + efficiency_weight * eff_w1
        w2 = (1 - efficiency_weight) * equal_w2 + efficiency_weight * eff_w2

        # Create merged expert
        merged = cls(
            input_dim=expert1.input_dim,
            output_dim=expert1.output_dim,
            config=expert1.config,
            expert_id=merged_expert_id,
            seed=seed if seed else merged_expert_id
        )

        # Weighted average of weights
        merged.W1 = w1 * expert1.W1 + w2 * expert2.W1
        merged.b1 = w1 * expert1.b1 + w2 * expert2.b1
        merged.W2 = w1 * expert1.W2 + w2 * expert2.W2
        merged.b2 = w1 * expert1.b2 + w2 * expert2.b2

        # Merged efficiency is weighted combination
        merged.efficiency = w1 * eff1 + w2 * eff2
        merged.node_efficiency = w1 * expert1.node_efficiency + w2 * expert2.node_efficiency

        # Reset statistics
        merged.activation_count = 0
        merged.total_count = 0
        merged.gradient_history = []
        merged.loss_history = []
        merged.efficiency_history = [merged.efficiency]

        return merged
