"""
Dynamic Neural Network for Emotional Classification

An adaptive-architecture neural network that dynamically adjusts its structure
during training, featuring the reward/penalty system for learning rate adaptation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time


@dataclass
class EmotionalState:
    """Tracks the network's emotional state during training."""
    total_rewards: int = 0
    total_penalties: int = 0
    reward_history: List[float] = field(default_factory=list)
    penalty_history: List[float] = field(default_factory=list)
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0

    @property
    def total_adjustments(self) -> int:
        return self.total_rewards + self.total_penalties

    @property
    def depression_ratio(self) -> float:
        if self.total_adjustments == 0:
            return 0.0
        return self.total_penalties / self.total_adjustments

    @property
    def excitement_ratio(self) -> float:
        if self.total_adjustments == 0:
            return 0.0
        return self.total_rewards / self.total_adjustments


@dataclass
class RewardPenaltyConfig:
    """Configuration for the reward/penalty system."""
    cost_improvement_threshold: float = 0.001
    efficiency_improvement_threshold: float = 0.01
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1.0
    baseline_learning_rate: float = 0.01
    max_adjustment_factor: float = 2.0
    min_adjustment_factor: float = 0.5
    extreme_threshold: float = 0.8
    window_size: int = 10


@dataclass
class TrainingPhaseConfig:
    """Configuration for the training phases."""
    # Phase 1: Exploration
    exploration_epochs: int = 10

    # Phase 2: Estimation
    min_estimated_epochs: int = 20
    max_estimated_epochs: int = 200
    default_estimated_epochs: int = 100

    # Architecture change frequency
    architecture_change_interval: int = 10

    # Node changes
    nodes_to_add: int = 8
    nodes_to_remove: int = 4
    add_probability: float = 0.5
    remove_probability: float = 0.3


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency computation and thresholds."""
    # Sigmoid threshold params
    sigmoid_k: float = 5.0
    sigmoid_base: float = 0.3
    sigmoid_range: float = 0.5
    sigmoid_center: float = 0.5

    # Efficiency computation
    efficiency_window: int = 5
    efficiency_multiplier: float = 10.0
    initial_efficiency: float = 0.5

    # Threshold multiplier for low efficiency
    low_efficiency_multiplier: float = 0.5


@dataclass
class HealthScoreConfig:
    """Configuration for health score computation."""
    layer_weight: int = 10
    cancer_denominator: float = 5.0
    alzheimer_denominator: float = 5.0
    healthy_threshold: float = 0.3
    at_risk_threshold: float = 0.7


@dataclass
class GradientConfig:
    """Configuration for gradient handling."""
    momentum: float = 0.9
    gradient_clip_value: float = 1.0


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    window_size: int = 30
    improvement_threshold: float = 1e-5
    max_consecutive_increases: int = 5


@dataclass
class FineTuningConfig:
    """Configuration for Phase 4 fine-tuning."""
    enabled: bool = True
    excitement_min: float = 0.3
    excitement_max: float = 0.7
    lr_multiplier: float = 0.1
    min_epochs: int = 5
    max_epochs: int = 50
    worthiness_threshold: float = 0.3
    improvement_weight: float = 0.4
    balance_weight: float = 0.4
    cost_potential_weight: float = 0.2


@dataclass
class DynamicNNConfig:
    """Configuration for Dynamic Neural Network."""
    input_size: int = 1000
    initial_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 7  # 7 emotions
    min_nodes: int = 16
    max_nodes: int = 512
    min_layers: int = 1
    max_layers: int = 8
    dropout: float = 0.2
    activation: str = 'relu'
    seed: int = 42

    # Sub-configurations
    training_phase: TrainingPhaseConfig = field(default_factory=TrainingPhaseConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    health_score: HealthScoreConfig = field(default_factory=HealthScoreConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    reward_penalty: RewardPenaltyConfig = field(default_factory=RewardPenaltyConfig)


@dataclass
class DynamicTrainingResult:
    """Training result for Dynamic NN."""
    success: bool
    epochs_completed: int
    final_cost: float
    final_accuracy: float
    best_cost: float
    best_accuracy: float
    cost_history: List[float]
    accuracy_history: List[float]
    efficiency_history: List[float]
    val_cost_history: List[float]
    val_accuracy_history: List[float]
    training_time: float
    # Architecture changes
    layers_added: int = 0
    layers_removed: int = 0
    nodes_added: int = 0
    nodes_removed: int = 0
    architecture_history: List[Tuple[int, int]] = field(default_factory=list)
    # Health scores
    cancer_score_history: List[float] = field(default_factory=list)
    alzheimer_score_history: List[float] = field(default_factory=list)
    # Emotional state
    total_rewards: int = 0
    total_penalties: int = 0
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0
    learning_rate_history: List[float] = field(default_factory=list)
    emotional_state_actions: List[str] = field(default_factory=list)


class DynamicNeuralNetwork:
    """
    Dynamic Neural Network with adaptive architecture and reward/penalty system.

    Features:
    - Dynamic layer/node addition and removal
    - 3-phase training (Exploration, Estimation, Main Training)
    - Reward/Penalty system for adaptive learning rate
    - Health monitoring (cancer/alzheimer scores)
    - Depression/Excitement ratio tracking
    """

    def __init__(self, config: DynamicNNConfig):
        """Initialize network with config."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize layers
        self._layers: List[Dict[str, np.ndarray]] = []
        self._init_architecture()

        # Architecture change tracking
        self.layers_added = 0
        self.layers_removed = 0
        self.nodes_added = 0
        self.nodes_removed = 0

        # Health tracking
        self.cancer_score_history: List[float] = []
        self.alzheimer_score_history: List[float] = []

        self._trained = False
        self.training_history: Optional[DynamicTrainingResult] = None

    def _init_architecture(self) -> None:
        """Initialize network architecture."""
        sizes = [self.config.input_size] + self.config.initial_hidden_sizes + [self.config.output_size]

        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]

            std = np.sqrt(2.0 / fan_in)
            W = self.rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)

            self._layers.append({'W': W, 'b': b})

    def _count_nodes(self) -> int:
        """Count total hidden nodes."""
        return sum(l['W'].shape[1] for l in self._layers[:-1])

    def _activation(self, z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == 'relu':
            if derivative:
                return (z > 0).astype(np.float32)
            return np.maximum(0, z)
        elif self.config.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z) ** 2
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-8)

    def _dropout_mask(self, shape: Tuple, training: bool) -> np.ndarray:
        """Generate dropout mask."""
        if training and self.config.dropout > 0:
            mask = (self.rng.random(shape) > self.config.dropout).astype(np.float32)
            mask /= (1 - self.config.dropout)
            return mask
        return np.ones(shape, dtype=np.float32)

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Forward pass through network."""
        cache = []
        a = X.astype(np.float32)

        for i, layer in enumerate(self._layers[:-1]):
            z = a @ layer['W'] + layer['b']
            a_pre = a.copy()
            a = self._activation(z)

            mask = self._dropout_mask(a.shape, training)
            a = a * mask

            cache.append({
                'a_prev': a_pre,
                'z': z,
                'a': a,
                'mask': mask
            })

        # Output layer
        z_out = a @ self._layers[-1]['W'] + self._layers[-1]['b']
        output = self._softmax(z_out)

        cache.append({
            'a_prev': a,
            'z': z_out,
            'a': output,
            'mask': np.ones_like(output)
        })

        return output, cache

    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_onehot

        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
        return loss

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, cache: List[Dict]) -> List[Dict]:
        """Backward pass to compute gradients."""
        batch_size = y_pred.shape[0]

        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_onehot

        gradients = []
        dz = (y_pred - y_true) / batch_size

        for i in range(len(self._layers) - 1, -1, -1):
            layer_cache = cache[i]
            a_prev = layer_cache['a_prev']

            dW = a_prev.T @ dz
            db = np.sum(dz, axis=0)

            gradients.insert(0, {'dW': dW, 'db': db})

            if i > 0:
                da = dz @ self._layers[i]['W'].T
                da = da * cache[i-1]['mask']
                dz = da * self._activation(cache[i-1]['z'], derivative=True)

        return gradients

    def _update_weights(self, gradients: List[Dict], learning_rate: float,
                        velocities: Optional[List[Dict]] = None) -> List[Dict]:
        """Update weights using SGD with momentum."""
        if velocities is None:
            velocities = [
                {'vW': np.zeros_like(l['W']), 'vb': np.zeros_like(l['b'])}
                for l in self._layers
            ]

        momentum = self.config.gradient.momentum
        for i, (layer, grad, vel) in enumerate(zip(self._layers, gradients, velocities)):
            vel['vW'] = momentum * vel['vW'] - learning_rate * grad['dW']
            vel['vb'] = momentum * vel['vb'] - learning_rate * grad['db']

            layer['W'] += vel['vW']
            layer['b'] += vel['vb']

        return velocities

    # ============ DYNAMIC ARCHITECTURE METHODS ============

    def _compute_efficiency(self, cost_history: List[float], window: int = None) -> float:
        """Compute training efficiency."""
        if window is None:
            window = self.config.efficiency.efficiency_window
        if len(cost_history) < 2:
            return self.config.efficiency.initial_efficiency

        recent = cost_history[-min(window, len(cost_history)):]
        if len(recent) < 2:
            return self.config.efficiency.initial_efficiency

        improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-8)
        center = self.config.efficiency.sigmoid_center
        multiplier = self.config.efficiency.efficiency_multiplier
        return min(1.0, max(0.0, center + improvement * multiplier))

    def _sigmoid_threshold(self, efficiency: float) -> float:
        """Adaptive saturation threshold."""
        k = self.config.efficiency.sigmoid_k
        base = self.config.efficiency.sigmoid_base
        range_val = self.config.efficiency.sigmoid_range
        center = self.config.efficiency.sigmoid_center
        sigmoid = 1.0 / (1.0 + np.exp(-k * (efficiency - center)))
        return base + range_val * sigmoid

    def _add_nodes(self, layer_idx: int, num_nodes: int) -> None:
        """Add nodes to a layer."""
        if layer_idx >= len(self._layers) - 1:
            return

        layer = self._layers[layer_idx]
        current_nodes = layer['W'].shape[1]
        new_nodes = min(num_nodes, self.config.max_nodes - current_nodes)

        if new_nodes <= 0:
            return

        # Expand current layer output
        std = np.sqrt(2.0 / layer['W'].shape[0])
        new_W = np.hstack([
            layer['W'],
            self.rng.normal(0, std, (layer['W'].shape[0], new_nodes)).astype(np.float32)
        ])
        new_b = np.append(layer['b'], np.zeros(new_nodes, dtype=np.float32))

        layer['W'] = new_W
        layer['b'] = new_b

        # Expand next layer input
        next_layer = self._layers[layer_idx + 1]
        std = np.sqrt(2.0 / (current_nodes + new_nodes))
        next_W = np.vstack([
            next_layer['W'],
            self.rng.normal(0, std, (new_nodes, next_layer['W'].shape[1])).astype(np.float32)
        ])
        next_layer['W'] = next_W

        self.nodes_added += new_nodes

    def _remove_nodes(self, layer_idx: int, num_nodes: int) -> None:
        """Remove nodes from a layer."""
        if layer_idx >= len(self._layers) - 1:
            return

        layer = self._layers[layer_idx]
        current_nodes = layer['W'].shape[1]
        nodes_to_remove = min(num_nodes, current_nodes - self.config.min_nodes)

        if nodes_to_remove <= 0:
            return

        # Find least important nodes (lowest L2 norm)
        node_importance = np.sum(layer['W'] ** 2, axis=0)
        keep_indices = np.argsort(node_importance)[-current_nodes + nodes_to_remove:]
        keep_indices = np.sort(keep_indices)

        # Remove from current layer
        layer['W'] = layer['W'][:, keep_indices]
        layer['b'] = layer['b'][keep_indices]

        # Remove from next layer input
        next_layer = self._layers[layer_idx + 1]
        next_layer['W'] = next_layer['W'][keep_indices, :]

        self.nodes_removed += nodes_to_remove

    def _add_layer(self) -> None:
        """Add a new hidden layer."""
        if len(self._layers) >= self.config.max_layers:
            return

        # Insert before output layer
        insert_idx = len(self._layers) - 1
        prev_size = self._layers[insert_idx - 1]['W'].shape[1] if insert_idx > 0 else self.config.input_size
        next_size = self._layers[insert_idx]['W'].shape[1]

        new_size = (prev_size + next_size) // 2
        new_size = max(self.config.min_nodes, min(self.config.max_nodes, new_size))

        # Create new layer
        std = np.sqrt(2.0 / prev_size)
        new_W = self.rng.normal(0, std, (prev_size, new_size)).astype(np.float32)
        new_b = np.zeros(new_size, dtype=np.float32)

        # Update next layer input size
        std = np.sqrt(2.0 / new_size)
        self._layers[insert_idx]['W'] = self.rng.normal(
            0, std, (new_size, self._layers[insert_idx]['W'].shape[1])
        ).astype(np.float32)

        self._layers.insert(insert_idx, {'W': new_W, 'b': new_b})
        self.layers_added += 1

    def _remove_layer(self) -> None:
        """Remove a hidden layer."""
        if len(self._layers) <= self.config.min_layers + 1:
            return

        # Find least important layer (lowest average activation)
        # For simplicity, remove second-to-last hidden layer
        remove_idx = len(self._layers) - 2

        if remove_idx <= 0:
            return

        # Connect previous to next
        prev_size = self._layers[remove_idx - 1]['W'].shape[1]
        next_size = self._layers[remove_idx + 1]['W'].shape[1]

        std = np.sqrt(2.0 / prev_size)
        self._layers[remove_idx + 1]['W'] = self.rng.normal(
            0, std, (prev_size, next_size)
        ).astype(np.float32)

        del self._layers[remove_idx]
        self.layers_removed += 1

    def _compute_health_scores(self, epoch: int) -> Tuple[float, float]:
        """Compute cancer and alzheimer scores."""
        layer_weight = self.config.health_score.layer_weight
        total_adds = self.nodes_added + self.layers_added * layer_weight
        total_removes = self.nodes_removed + self.layers_removed * layer_weight

        # Cancer: excessive growth
        growth_rate = total_adds / (epoch + 1)
        cancer = min(1.0, growth_rate / self.config.health_score.cancer_denominator)

        # Alzheimer: excessive pruning
        prune_rate = total_removes / (epoch + 1)
        alzheimer = min(1.0, prune_rate / self.config.health_score.alzheimer_denominator)

        return cancer, alzheimer

    # ============ REWARD/PENALTY SYSTEM METHODS ============

    def _compute_improvement_metrics(self, cost_history: List[float],
                                      efficiency_history: List[float],
                                      window: int = 5) -> Dict[str, float]:
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

    def _should_reward(self, metrics: Dict[str, float], config: RewardPenaltyConfig) -> Tuple[bool, float]:
        """Determine if current epoch deserves a reward."""
        cost_improving = metrics["cost_improvement"] > config.cost_improvement_threshold
        efficiency_good = metrics["efficiency_improvement"] >= 0 or metrics["efficiency_trend"] > 0
        trend_positive = metrics["cost_trend"] < 0

        should_reward = cost_improving and efficiency_good and trend_positive

        if should_reward:
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)
        else:
            magnitude = 0.0

        return should_reward, magnitude

    def _should_penalize(self, metrics: Dict[str, float], config: RewardPenaltyConfig) -> Tuple[bool, float]:
        """Determine if current epoch deserves a penalty."""
        cost_degrading = metrics["cost_improvement"] < -config.cost_improvement_threshold
        efficiency_bad = metrics["efficiency_improvement"] < -config.efficiency_improvement_threshold
        trend_negative = metrics["cost_trend"] > 0

        should_penalize = cost_degrading or (efficiency_bad and trend_negative)

        if should_penalize:
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)
        else:
            magnitude = 0.0

        return should_penalize, magnitude

    def _apply_reward(self, learning_rate: float, magnitude: float,
                      config: RewardPenaltyConfig, emotional_state: EmotionalState) -> float:
        """Apply reward by decreasing learning rate."""
        decrease_factor = 1.0 - magnitude * (1.0 - config.min_adjustment_factor)
        new_lr = learning_rate * decrease_factor
        new_lr = max(new_lr, config.min_learning_rate)

        emotional_state.total_rewards += 1
        emotional_state.reward_history.append(magnitude)

        return new_lr

    def _apply_penalty(self, learning_rate: float, magnitude: float,
                       config: RewardPenaltyConfig, emotional_state: EmotionalState) -> float:
        """Apply penalty by increasing learning rate."""
        increase_factor = 1.0 + magnitude * (config.max_adjustment_factor - 1.0)
        new_lr = learning_rate * increase_factor
        new_lr = min(new_lr, config.max_learning_rate)

        emotional_state.total_penalties += 1
        emotional_state.penalty_history.append(magnitude)

        return new_lr

    def _check_extreme_states(self, learning_rate: float, config: RewardPenaltyConfig,
                               emotional_state: EmotionalState) -> Tuple[float, str]:
        """Check for extreme emotional states and reset LR if detected."""
        depression = emotional_state.depression_ratio
        excitement = emotional_state.excitement_ratio

        state = "neutral"

        if depression > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_depression"
        elif excitement > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_excitement"
        elif depression > 0.5:
            state = "depressed"
        elif excitement > 0.5:
            state = "excited"

        emotional_state.depression_history.append(depression)
        emotional_state.excitement_history.append(excitement)

        return learning_rate, state

    def _apply_reward_penalty_system(self, learning_rate: float, cost_history: List[float],
                                      efficiency_history: List[float], config: RewardPenaltyConfig,
                                      emotional_state: EmotionalState) -> Tuple[float, str]:
        """Apply the complete reward/penalty system for one epoch."""
        metrics = self._compute_improvement_metrics(cost_history, efficiency_history, config.window_size)

        action = "neutral"

        should_reward, reward_mag = self._should_reward(metrics, config)
        if should_reward:
            learning_rate = self._apply_reward(learning_rate, reward_mag, config, emotional_state)
            action = "reward"
        else:
            should_penalize, penalty_mag = self._should_penalize(metrics, config)
            if should_penalize:
                learning_rate = self._apply_penalty(learning_rate, penalty_mag, config, emotional_state)
                action = "penalty"

        learning_rate, extreme_state = self._check_extreme_states(learning_rate, config, emotional_state)
        if extreme_state.startswith("extreme"):
            action = f"{extreme_state}_reset"

        return learning_rate, action

    def _estimate_additional_epochs(
        self,
        cost_history: List[float],
        efficiency_history: List[float],
        emotional_state: EmotionalState,
        min_epochs: int = None,
        max_epochs: int = None
    ) -> int:
        """
        Estimate additional epochs needed for fine-tuning.

        Factors:
        - Recent cost improvement rate (configurable weight)
        - Excitement ratio distance from optimal 0.5 (configurable weight)
        - Cost potential - room for improvement (configurable weight)

        Returns:
            int: Estimated epochs (0 if not worthwhile)
        """
        ft_config = self.config.fine_tuning
        if min_epochs is None:
            min_epochs = ft_config.min_epochs
        if max_epochs is None:
            max_epochs = ft_config.max_epochs

        if len(cost_history) < 10:
            return 0

        # Factor 1: Recent cost improvement rate (last 20 epochs)
        window = min(20, len(cost_history))
        recent_costs = cost_history[-window:]
        cost_improvement_rate = (recent_costs[0] - recent_costs[-1]) / (window * (recent_costs[0] + 1e-8))

        # Normalize improvement rate to [0, 1] range
        improvement_score = min(1.0, max(0.0, cost_improvement_rate * 100))

        # Factor 2: Excitement ratio distance from optimal (0.5)
        # Closer to 0.5 = more balanced learning = more potential for refinement
        excitement = emotional_state.excitement_ratio
        balance_score = 1.0 - 2.0 * abs(excitement - 0.5)  # 1.0 at 0.5, 0.0 at 0 or 1

        # Factor 3: Current cost level (lower cost = less room for improvement)
        final_cost = cost_history[-1]
        cost_potential = min(1.0, final_cost * 2)  # More epochs if cost still high

        # Combined worthiness score with configurable weights
        worthiness = (
            improvement_score * ft_config.improvement_weight +
            balance_score * ft_config.balance_weight +
            cost_potential * ft_config.cost_potential_weight
        )

        # Threshold: only fine-tune if worthiness > threshold
        if worthiness < ft_config.worthiness_threshold:
            return 0

        # Scale epochs by worthiness
        estimated = int(min_epochs + worthiness * (max_epochs - min_epochs))
        return max(min_epochs, min(max_epochs, estimated))

    def _should_fine_tune(
        self,
        cost_history: List[float],
        efficiency_history: List[float],
        emotional_state: EmotionalState,
        excitement_min: float = None,
        excitement_max: float = None
    ) -> Tuple[bool, int]:
        """
        Determine if conditional fine-tuning should be performed.

        Entry conditions:
        1. Excitement ratio must be in balanced range
        2. Estimation function must return > 0 epochs

        Returns:
            Tuple[bool, int]: (should_fine_tune, estimated_epochs)
        """
        ft_config = self.config.fine_tuning
        if not ft_config.enabled:
            return (False, 0)

        if excitement_min is None:
            excitement_min = ft_config.excitement_min
        if excitement_max is None:
            excitement_max = ft_config.excitement_max

        excitement = emotional_state.excitement_ratio

        # Check excitement ratio is in balanced range
        if not (excitement_min <= excitement <= excitement_max):
            return (False, 0)

        # Get estimation
        estimated_epochs = self._estimate_additional_epochs(
            cost_history, efficiency_history, emotional_state
        )

        if estimated_epochs <= 0:
            return (False, 0)

        return (True, estimated_epochs)

    # ============ TRAINING ============

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> DynamicTrainingResult:
        """
        Train with 4-phase approach and reward/penalty system.

        Phase 1: Exploration (10 epochs) - Initial learning
        Phase 2: Estimation - Estimate required epochs
        Phase 3: Main Training - Dynamic architecture + reward/penalty
        Phase 4: Conditional Fine-Tuning - Low LR refinement if conditions met
            Entry: excitement_ratio in [0.3, 0.7], estimation returns > 0
            Uses 10% of Phase 3 final LR, frozen architecture
        """
        start_time = time.time()

        cost_history = []
        accuracy_history = []
        efficiency_history = []
        val_cost_history = []
        val_accuracy_history = []
        architecture_history = []
        learning_rate_history = []

        best_cost = float('inf')
        best_accuracy = 0.0
        velocities = None
        n_samples = len(X)

        # Emotional state tracking
        emotional_state = EmotionalState()
        rp_config = self.config.reward_penalty
        # Override baseline learning rate if needed
        rp_config = RewardPenaltyConfig(
            cost_improvement_threshold=rp_config.cost_improvement_threshold,
            efficiency_improvement_threshold=rp_config.efficiency_improvement_threshold,
            min_learning_rate=rp_config.min_learning_rate,
            max_learning_rate=rp_config.max_learning_rate,
            baseline_learning_rate=learning_rate,
            max_adjustment_factor=rp_config.max_adjustment_factor,
            min_adjustment_factor=rp_config.min_adjustment_factor,
            extreme_threshold=rp_config.extreme_threshold,
            window_size=rp_config.window_size
        )
        emotional_state_actions = []

        # Get training phase config
        tp_config = self.config.training_phase
        exploration_epochs = tp_config.exploration_epochs

        if verbose:
            print("=" * 60)
            print("DYNAMIC NEURAL NETWORK TRAINING")
            print("=" * 60)
            print(f"Initial architecture: {[l['W'].shape for l in self._layers]}")
            print(f"Initial parameters: {self.count_parameters():,}")
            print(f"Training samples: {n_samples}")
            print("=" * 60)

        # ============ PHASE 1: EXPLORATION ============
        if verbose:
            print(f"\nPHASE 1: EXPLORATION ({exploration_epochs} epochs)")

        exploration_costs = []
        for epoch in range(exploration_epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred, cache = self.forward(X_batch, training=True)
                loss = self._cross_entropy_loss(y_pred, y_batch)
                epoch_losses.append(loss)

                gradients = self.backward(y_pred, y_batch, cache)
                velocities = self._update_weights(gradients, learning_rate, velocities)

            epoch_cost = np.mean(epoch_losses)
            exploration_costs.append(epoch_cost)
            cost_history.append(epoch_cost)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)

            y_pred_train, _ = self.forward(X, training=False)
            train_acc = np.mean(np.argmax(y_pred_train, axis=-1) == y)
            accuracy_history.append(train_acc)

            if X_val is not None:
                y_pred_val, _ = self.forward(X_val, training=False)
                val_cost = self._cross_entropy_loss(y_pred_val, y_val)
                val_acc = np.mean(np.argmax(y_pred_val, axis=-1) == y_val)
                val_cost_history.append(val_cost)
                val_accuracy_history.append(val_acc)

            architecture_history.append((len(self._layers), self._count_nodes()))

            if verbose:
                print(f"  Epoch {epoch}: cost={epoch_cost:.4f}, acc={train_acc:.2%}")

        # ============ PHASE 2: ESTIMATION ============
        if verbose:
            print("\nPHASE 2: ESTIMATION")

        if len(exploration_costs) >= 2:
            avg_improvement = (exploration_costs[0] - exploration_costs[-1]) / len(exploration_costs)
            if avg_improvement > 0:
                remaining = exploration_costs[-1] / avg_improvement
                estimated_epochs = int(min(max(remaining, tp_config.min_estimated_epochs), tp_config.max_estimated_epochs))
            else:
                estimated_epochs = tp_config.default_estimated_epochs
        else:
            estimated_epochs = tp_config.default_estimated_epochs

        if verbose:
            print(f"  Estimated epochs for convergence: {estimated_epochs}")

        # ============ PHASE 3: MAIN TRAINING ============
        if verbose:
            print(f"\nPHASE 3: MAIN TRAINING ({estimated_epochs} epochs)")
            print("  Using Reward/Penalty System for adaptive LR")

        for epoch in range(estimated_epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred, cache = self.forward(X_batch, training=True)
                loss = self._cross_entropy_loss(y_pred, y_batch)
                epoch_losses.append(loss)

                gradients = self.backward(y_pred, y_batch, cache)
                velocities = self._update_weights(gradients, learning_rate, velocities)

            epoch_cost = np.mean(epoch_losses)
            cost_history.append(epoch_cost)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)

            y_pred_train, _ = self.forward(X, training=False)
            train_acc = np.mean(np.argmax(y_pred_train, axis=-1) == y)
            accuracy_history.append(train_acc)

            if X_val is not None:
                y_pred_val, _ = self.forward(X_val, training=False)
                val_cost = self._cross_entropy_loss(y_pred_val, y_val)
                val_acc = np.mean(np.argmax(y_pred_val, axis=-1) == y_val)
                val_cost_history.append(val_cost)
                val_accuracy_history.append(val_acc)

                if val_cost < best_cost:
                    best_cost = val_cost
                    best_accuracy = val_acc

            # Dynamic architecture adaptation
            threshold = self._sigmoid_threshold(efficiency)
            architecture_changed = False
            arch_interval = tp_config.architecture_change_interval
            if efficiency > threshold and epoch % arch_interval == 0:
                # High efficiency - consider pruning
                if self.rng.random() < tp_config.remove_probability:
                    self._remove_nodes(self.rng.integers(0, len(self._layers) - 1), tp_config.nodes_to_remove)
                    architecture_changed = True
            elif efficiency < threshold * self.config.efficiency.low_efficiency_multiplier and epoch % arch_interval == 0:
                # Low efficiency - consider growing
                if self.rng.random() < tp_config.add_probability:
                    self._add_nodes(self.rng.integers(0, len(self._layers) - 1), tp_config.nodes_to_add)
                    architecture_changed = True

            # Reset velocities after architecture change to match new weight shapes
            if architecture_changed:
                velocities = None

            # Health scores
            cancer, alzheimer = self._compute_health_scores(len(cost_history))
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)

            architecture_history.append((len(self._layers), self._count_nodes()))

            # Apply reward/penalty system
            learning_rate, action = self._apply_reward_penalty_system(
                learning_rate, cost_history, efficiency_history, rp_config, emotional_state
            )
            learning_rate_history.append(learning_rate)
            emotional_state_actions.append(action)

            if verbose and epoch % 10 == 0:
                msg = f"  Epoch {epoch}: cost={epoch_cost:.4f}, acc={train_acc:.2%}, eff={efficiency:.2%}"
                msg += f", lr={learning_rate:.6f}, R/P={emotional_state.total_rewards}/{emotional_state.total_penalties}"
                if X_val is not None:
                    msg += f", val_acc={val_acc:.2%}"
                print(msg)

            # Early stopping
            es_config = self.config.early_stopping
            if len(cost_history) > es_config.window_size:
                if cost_history[-es_config.window_size] - cost_history[-1] < es_config.improvement_threshold:
                    if verbose:
                        print(f"\n  Early stopping at epoch {epoch}")
                    break

        # ============ PHASE 4: CONDITIONAL FINE-TUNING ============
        should_fine_tune, fine_tune_epochs = self._should_fine_tune(
            cost_history, efficiency_history, emotional_state
        )
        fine_tune_completed = 0

        if should_fine_tune:
            ft_config = self.config.fine_tuning
            if verbose:
                print(f"\nPHASE 4: CONDITIONAL FINE-TUNING ({fine_tune_epochs} epochs)")
                print(f"  Entry condition met: excitement_ratio={emotional_state.excitement_ratio:.2%} in [{ft_config.excitement_min}, {ft_config.excitement_max}]")

            # Use configured multiplier of final Phase 3 learning rate
            fine_tune_lr = learning_rate * ft_config.lr_multiplier
            fine_tune_lr = max(fine_tune_lr, rp_config.min_learning_rate)

            # Track consecutive cost increases for early stopping
            consecutive_increases = 0
            max_consecutive_increases = self.config.early_stopping.max_consecutive_increases
            fine_tune_best_cost = cost_history[-1]

            for ft_epoch in range(fine_tune_epochs):
                indices = self.rng.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                epoch_losses = []
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    y_pred, cache = self.forward(X_batch, training=True)
                    loss = self._cross_entropy_loss(y_pred, y_batch)
                    epoch_losses.append(loss)

                    gradients = self.backward(y_pred, y_batch, cache)
                    velocities = self._update_weights(gradients, fine_tune_lr, velocities)

                epoch_cost = np.mean(epoch_losses)
                cost_history.append(epoch_cost)

                efficiency = self._compute_efficiency(cost_history)
                efficiency_history.append(efficiency)

                y_pred_train, _ = self.forward(X, training=False)
                train_acc = np.mean(np.argmax(y_pred_train, axis=-1) == y)
                accuracy_history.append(train_acc)

                if X_val is not None:
                    y_pred_val, _ = self.forward(X_val, training=False)
                    val_cost = self._cross_entropy_loss(y_pred_val, y_val)
                    val_acc = np.mean(np.argmax(y_pred_val, axis=-1) == y_val)
                    val_cost_history.append(val_cost)
                    val_accuracy_history.append(val_acc)

                    if val_cost < best_cost:
                        best_cost = val_cost
                        best_accuracy = val_acc

                # No architecture changes during fine-tuning (frozen)
                architecture_history.append((len(self._layers), self._count_nodes()))

                # Health scores (continue tracking)
                cancer, alzheimer = self._compute_health_scores(len(cost_history))
                self.cancer_score_history.append(cancer)
                self.alzheimer_score_history.append(alzheimer)

                # Track LR (fixed during fine-tuning)
                learning_rate_history.append(fine_tune_lr)
                emotional_state_actions.append("fine_tune")

                # Early stopping: check for consecutive cost increases
                if epoch_cost < fine_tune_best_cost:
                    fine_tune_best_cost = epoch_cost
                    consecutive_increases = 0
                else:
                    consecutive_increases += 1

                if consecutive_increases >= max_consecutive_increases:
                    if verbose:
                        print(f"  Fine-tuning early stop at epoch {ft_epoch}: "
                              f"cost increased for {consecutive_increases} consecutive epochs")
                    fine_tune_completed = ft_epoch + 1
                    break

                if verbose and ft_epoch % 5 == 0:
                    msg = f"  FT Epoch {ft_epoch}: cost={epoch_cost:.4f}, acc={train_acc:.2%}, lr={fine_tune_lr:.6f}"
                    if X_val is not None:
                        msg += f", val_acc={val_acc:.2%}"
                    print(msg)

                fine_tune_completed = ft_epoch + 1

            if verbose:
                print(f"  Fine-tuning completed: {fine_tune_completed} epochs")
        else:
            if verbose:
                ft_config = self.config.fine_tuning
                excitement = emotional_state.excitement_ratio
                if not ft_config.enabled:
                    print(f"\nPHASE 4: SKIPPED (fine-tuning disabled in config)")
                elif not (ft_config.excitement_min <= excitement <= ft_config.excitement_max):
                    print(f"\nPHASE 4: SKIPPED (excitement_ratio={excitement:.2%} outside [{ft_config.excitement_min}, {ft_config.excitement_max}])")
                else:
                    print(f"\nPHASE 4: SKIPPED (estimation function returned 0 epochs)")

        training_time = time.time() - start_time
        self._trained = True

        if verbose:
            print("=" * 60)
            print(f"Training completed in {training_time:.2f}s")
            print(f"Final architecture: {[l['W'].shape for l in self._layers]}")
            print(f"Final parameters: {self.count_parameters():,}")
            print(f"Best cost: {best_cost:.4f}, Best accuracy: {best_accuracy:.2%}")
            print(f"Emotional state: {emotional_state.total_rewards} rewards, {emotional_state.total_penalties} penalties")
            print(f"Depression ratio: {emotional_state.depression_ratio:.1%}, Excitement ratio: {emotional_state.excitement_ratio:.1%}")
            print(f"LR resets: {emotional_state.lr_reset_count}")
            if fine_tune_completed > 0:
                print(f"Fine-tuning: {fine_tune_completed} epochs completed")
            else:
                print(f"Fine-tuning: skipped")
            print("=" * 60)

        self.training_history = DynamicTrainingResult(
            success=True,
            epochs_completed=len(cost_history),
            final_cost=cost_history[-1],
            final_accuracy=accuracy_history[-1],
            best_cost=best_cost,
            best_accuracy=best_accuracy,
            cost_history=cost_history,
            accuracy_history=accuracy_history,
            efficiency_history=efficiency_history,
            val_cost_history=val_cost_history,
            val_accuracy_history=val_accuracy_history,
            training_time=training_time,
            layers_added=self.layers_added,
            layers_removed=self.layers_removed,
            nodes_added=self.nodes_added,
            nodes_removed=self.nodes_removed,
            architecture_history=architecture_history,
            cancer_score_history=self.cancer_score_history,
            alzheimer_score_history=self.alzheimer_score_history,
            total_rewards=emotional_state.total_rewards,
            total_penalties=emotional_state.total_penalties,
            depression_history=emotional_state.depression_history,
            excitement_history=emotional_state.excitement_history,
            lr_reset_count=emotional_state.lr_reset_count,
            learning_rate_history=learning_rate_history,
            emotional_state_actions=emotional_state_actions
        )

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        y_pred, _ = self.forward(X, training=False)
        return np.argmax(y_pred, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        y_pred, _ = self.forward(X, training=False)
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        y_pred_proba, _ = self.forward(X, training=False)
        y_pred = np.argmax(y_pred_proba, axis=-1)

        accuracy = np.mean(y_pred == y)
        loss = self._cross_entropy_loss(y_pred_proba, y)

        top3_acc = np.mean([y[i] in np.argsort(y_pred_proba[i])[-3:] for i in range(len(y))])
        top5_acc = np.mean([y[i] in np.argsort(y_pred_proba[i])[-5:] for i in range(len(y))])

        class_acc = {}
        for c in range(self.config.output_size):
            mask = y == c
            if np.sum(mask) > 0:
                class_acc[c] = np.mean(y_pred[mask] == c)

        return {
            'accuracy': accuracy,
            'loss': loss,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'class_accuracy': class_acc
        }

    def count_parameters(self) -> int:
        """Count total parameters."""
        total = 0
        for layer in self._layers:
            total += layer['W'].size + layer['b'].size
        return total

    def get_architecture_summary(self) -> Dict:
        """Get architecture summary."""
        return {
            'layer_sizes': [l['W'].shape for l in self._layers],
            'num_layers': len(self._layers),
            'total_nodes': self._count_nodes(),
            'total_parameters': self.count_parameters(),
            'layers_added': self.layers_added,
            'layers_removed': self.layers_removed,
            'nodes_added': self.nodes_added,
            'nodes_removed': self.nodes_removed
        }

    def health_status(self) -> Dict:
        """Get health status including emotional state."""
        cancer = self.cancer_score_history[-1] if self.cancer_score_history else 0.0
        alzheimer = self.alzheimer_score_history[-1] if self.alzheimer_score_history else 0.0

        hs_config = self.config.health_score
        if max(cancer, alzheimer) < hs_config.healthy_threshold:
            state = 'Healthy'
        elif max(cancer, alzheimer) < hs_config.at_risk_threshold:
            state = 'At Risk'
        else:
            state = 'Critical'

        # Emotional state
        depression = 0.0
        excitement = 0.0
        emotional_state = "neutral"

        rp_config = self.config.reward_penalty
        if self.training_history:
            total = self.training_history.total_rewards + self.training_history.total_penalties
            if total > 0:
                depression = self.training_history.total_penalties / total
                excitement = self.training_history.total_rewards / total

            if depression > rp_config.extreme_threshold:
                emotional_state = "extreme_depression"
            elif excitement > rp_config.extreme_threshold:
                emotional_state = "extreme_excitement"
            elif depression > 0.5:
                emotional_state = "depressed"
            elif excitement > 0.5:
                emotional_state = "excited"

        return {
            'state': state,
            'cancer_score': cancer,
            'alzheimer_score': alzheimer,
            'overall_health': 1.0 - max(cancer, alzheimer),
            'depression_ratio': depression,
            'excitement_ratio': excitement,
            'emotional_state': emotional_state,
            'total_rewards': self.training_history.total_rewards if self.training_history else 0,
            'total_penalties': self.training_history.total_penalties if self.training_history else 0,
            'lr_reset_count': self.training_history.lr_reset_count if self.training_history else 0
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            'config': {
                'input_size': self.config.input_size,
                'initial_hidden_sizes': self.config.initial_hidden_sizes,
                'output_size': self.config.output_size,
                'min_nodes': self.config.min_nodes,
                'max_nodes': self.config.max_nodes,
                'min_layers': self.config.min_layers,
                'max_layers': self.config.max_layers,
                'dropout': self.config.dropout,
                'activation': self.config.activation,
                'seed': self.config.seed
            },
            'num_layers': len(self._layers)
        }

        for i, layer in enumerate(self._layers):
            data[f'layer_{i}_W'] = layer['W']
            data[f'layer_{i}_b'] = layer['b']

        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> 'DynamicNeuralNetwork':
        """Load model from file."""
        data = np.load(path, allow_pickle=True)

        config = DynamicNNConfig(
            input_size=int(data['config'].item()['input_size']),
            initial_hidden_sizes=list(data['config'].item()['initial_hidden_sizes']),
            output_size=int(data['config'].item()['output_size']),
            min_nodes=int(data['config'].item()['min_nodes']),
            max_nodes=int(data['config'].item()['max_nodes']),
            min_layers=int(data['config'].item()['min_layers']),
            max_layers=int(data['config'].item()['max_layers']),
            dropout=float(data['config'].item()['dropout']),
            activation=str(data['config'].item()['activation']),
            seed=int(data['config'].item()['seed'])
        )

        model = cls(config)
        model._layers = []

        num_layers = int(data['num_layers'])
        for i in range(num_layers):
            model._layers.append({
                'W': data[f'layer_{i}_W'],
                'b': data[f'layer_{i}_b']
            })

        model._trained = True
        return model
