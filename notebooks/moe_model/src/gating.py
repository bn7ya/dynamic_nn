"""
Gating Network (Router) for MoE.

Implements top-k routing with load balancing and exploration noise.
"""

import numpy as np
from typing import Tuple, Dict, List

try:
    from .moe_config import GatingConfig, LoadBalanceConfig
except ImportError:
    from moe_config import GatingConfig, LoadBalanceConfig


class GatingNetwork:
    """
    Gating network (router) for MoE.

    Uses top-k routing with:
    - Noise injection for exploration during training
    - Load tracking for balancing auxiliary loss
    - Health metrics per expert
    """

    def __init__(
        self,
        input_dim: int,
        config: GatingConfig,
        load_balance_config: LoadBalanceConfig,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.noise_std = config.noise_std
        self.temperature = config.temperature
        self.config = config
        self.load_balance_config = load_balance_config

        # Router weights
        np.random.seed(seed)
        self.W_gate = np.random.randn(input_dim, self.num_experts) * 0.01
        self.b_gate = np.zeros(self.num_experts)

        # Gradient accumulators
        self.dW_gate = np.zeros_like(self.W_gate)
        self.db_gate = np.zeros_like(self.b_gate)

        # Load tracking for balancing
        self.expert_loads = np.zeros(self.num_experts)
        self.total_tokens = 0
        self.expert_load_history: List[np.ndarray] = []

        # Whether routing is frozen (Phase 4)
        self.trainable = True

        # Cache for backward
        self._cache = {}

    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute routing weights for input.

        Args:
            x: Input tensor (batch, input_dim)
            training: Whether in training mode

        Returns:
            router_weights: (batch, top_k) - weights for selected experts
            expert_indices: (batch, top_k) - indices of selected experts
            router_probs: (batch, num_experts) - full probability distribution
        """
        batch_size = x.shape[0]

        # Compute logits
        logits = x @ self.W_gate + self.b_gate  # (batch, num_experts)

        # Add noise during training for exploration (load balancing)
        if training and self.noise_std > 0:
            noise = np.random.randn(*logits.shape) * self.noise_std
            logits = logits + noise

        # Apply temperature and softmax
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)  # (batch, num_experts)

        # Top-k selection
        expert_indices = np.argsort(probs, axis=-1)[:, -self.top_k:][:, ::-1]  # (batch, top_k), descending

        # Gather top-k weights
        router_weights = np.zeros((batch_size, self.top_k))
        for b in range(batch_size):
            router_weights[b] = probs[b, expert_indices[b]]

        # Renormalize so weights sum to 1
        router_weights = router_weights / (router_weights.sum(axis=-1, keepdims=True) + 1e-8)

        # Update load tracking
        if training:
            self._update_load_stats(probs, expert_indices)
            self._cache = {
                'x': x,
                'logits': logits,
                'probs': probs,
                'expert_indices': expert_indices,
                'router_weights': router_weights
            }

        return router_weights, expert_indices, probs

    def backward(
        self,
        grad_weights: np.ndarray,
        learning_rate: float
    ) -> np.ndarray:
        """
        Backward pass through gating network.

        Args:
            grad_weights: Gradient of loss w.r.t. router weights (batch, top_k)
            learning_rate: Learning rate

        Returns:
            Gradient of loss w.r.t. input (batch, input_dim)
        """
        if not self.trainable:
            return np.zeros((grad_weights.shape[0], self.input_dim))

        x = self._cache['x']
        probs = self._cache['probs']
        expert_indices = self._cache['expert_indices']
        router_weights = self._cache['router_weights']
        batch_size = x.shape[0]

        # Gradient w.r.t. full probability distribution
        grad_probs = np.zeros_like(probs)

        # Scatter gradients from top-k back to full distribution
        for b in range(batch_size):
            # Account for renormalization
            sum_weights = router_weights[b].sum()
            for k in range(self.top_k):
                idx = expert_indices[b, k]
                # d(w_k / sum) / d(p_idx) where w_k = p_idx
                grad_probs[b, idx] += grad_weights[b, k] / (sum_weights + 1e-8)

        # Softmax backward: d_logits = probs * (grad_probs - sum(probs * grad_probs))
        sum_grad = np.sum(probs * grad_probs, axis=-1, keepdims=True)
        grad_logits = probs * (grad_probs - sum_grad) / self.temperature

        # Gradient w.r.t. gate weights
        self.dW_gate = x.T @ grad_logits / batch_size
        self.db_gate = np.mean(grad_logits, axis=0)

        # Gradient w.r.t. input
        grad_input = grad_logits @ self.W_gate.T

        # Update parameters
        self.W_gate -= learning_rate * self.dW_gate
        self.b_gate -= learning_rate * self.db_gate

        return grad_input

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def _update_load_stats(
        self,
        probs: np.ndarray,
        expert_indices: np.ndarray
    ) -> None:
        """Update expert load statistics."""
        batch_size = probs.shape[0]

        # Count how many tokens go to each expert (based on top-k selection)
        for b in range(batch_size):
            for k in range(self.top_k):
                self.expert_loads[expert_indices[b, k]] += 1

        self.total_tokens += batch_size * self.top_k

    def compute_load_balance_loss(self) -> float:
        """
        Compute auxiliary loss for load balancing.

        Encourages uniform expert utilization (from Switch Transformer).
        """
        if self.total_tokens == 0:
            return 0.0

        # Current load distribution
        load_dist = self.expert_loads / max(1, self.total_tokens)

        # Target is uniform distribution
        target_load = 1.0 / self.num_experts

        # Squared difference from uniform
        balance_loss = np.sum((load_dist - target_load) ** 2) * self.num_experts

        return balance_loss * self.load_balance_config.load_balance_weight

    def compute_importance_loss(self) -> float:
        """
        Compute importance-weighted load balance loss.

        This is the standard MoE auxiliary loss from literature.
        """
        if 'probs' not in self._cache or self.total_tokens == 0:
            return 0.0

        probs = self._cache['probs']  # (batch, num_experts)

        # Fraction of tokens routed to each expert
        load_dist = self.expert_loads / max(1, self.total_tokens)

        # Importance: average probability mass per expert
        importance = np.mean(probs, axis=0)

        # Auxiliary loss: N * sum(load * importance)
        aux_loss = self.num_experts * np.sum(load_dist * importance)

        return aux_loss * self.load_balance_config.load_balance_weight

    def get_expert_health(self) -> Dict[str, float]:
        """
        Get health metrics per expert (cancer/alzheimer analogy).

        Cancer: Expert domination (one expert getting too much traffic)
        Alzheimer: Expert death (expert not being used)
        """
        if self.total_tokens == 0:
            return {}

        load_dist = self.expert_loads / max(1, self.total_tokens)

        health = {}
        for i in range(self.num_experts):
            load = load_dist[i]
            if load > self.load_balance_config.max_expert_load:
                # Expert getting too much traffic (cancer)
                health[f'expert_{i}_cancer'] = load
            elif load < self.load_balance_config.min_expert_load:
                # Expert not getting enough traffic (alzheimer)
                health[f'expert_{i}_alzheimer'] = 1.0 - load

        return health

    def compute_cancer_score(self) -> float:
        """Compute overall cancer score (expert domination)."""
        if self.total_tokens == 0:
            return 0.0

        load_dist = self.expert_loads / max(1, self.total_tokens)
        max_load = np.max(load_dist)

        threshold = self.load_balance_config.max_expert_load
        if max_load > threshold:
            return (max_load - threshold) / (1.0 - threshold)
        return 0.0

    def compute_alzheimer_score(self) -> float:
        """Compute overall alzheimer score (expert death)."""
        if self.total_tokens == 0:
            return 0.0

        load_dist = self.expert_loads / max(1, self.total_tokens)
        threshold = self.load_balance_config.min_expert_load

        # Count experts below threshold
        dead_experts = np.sum(load_dist < threshold)
        return dead_experts / self.num_experts

    def get_load_distribution(self) -> np.ndarray:
        """Get current load distribution across experts."""
        if self.total_tokens == 0:
            return np.ones(self.num_experts) / self.num_experts
        return self.expert_loads / max(1, self.total_tokens)

    def reset_load_stats(self) -> None:
        """Reset load statistics (called periodically)."""
        # Store history before reset
        if self.total_tokens > 0:
            self.expert_load_history.append(self.get_load_distribution().copy())

        self.expert_loads = np.zeros(self.num_experts)
        self.total_tokens = 0

    def set_noise_for_phase(self, phase: int) -> None:
        """Set noise level based on training phase."""
        if phase == 1:  # Exploration
            self.noise_std = self.config.exploration_noise_std
        elif phase == 3:  # Main training
            self.noise_std = self.config.main_noise_std
        else:  # Phase 4 (fine-tuning)
            self.noise_std = self.config.finetune_noise_std

    def freeze(self) -> None:
        """Freeze gating network (for Phase 4)."""
        self.trainable = False
        self.noise_std = 0.0

    def unfreeze(self) -> None:
        """Unfreeze gating network."""
        self.trainable = True

    def get_stats(self) -> dict:
        """Get gating statistics for monitoring."""
        load_dist = self.get_load_distribution()
        return {
            'expert_loads': load_dist.tolist(),
            'load_entropy': -np.sum(load_dist * np.log(load_dist + 1e-8)),
            'load_balance_loss': self.compute_load_balance_loss(),
            'cancer_score': self.compute_cancer_score(),
            'alzheimer_score': self.compute_alzheimer_score(),
            'total_tokens_routed': self.total_tokens,
        }
